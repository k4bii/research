import ast
import inspect
import json
import math
import re
from difflib import SequenceMatcher
from io import StringIO
from typing import Any, Callable, Dict

import nbformat
import numpy as np
import pandas as pd
import polars as pl
from bs4 import BeautifulSoup
from nbclient import NotebookClient

from final_type_analysis import compare_polars_with_confidence


def parse_dataframe_html(
    df: pd.DataFrame,
    source_col: str,
    parser: Callable[[BeautifulSoup], Dict[str, Any]],
    html_parser: str = "html.parser",
) -> pd.DataFrame:
    """Kobi's helper for expanding HTML fragments inside DataFrame columns."""
    def _extract(html):
        if pd.isna(html) or html is None:
            return {}
        soup = BeautifulSoup(str(html), html_parser)
        return parser(soup) or {}

    parsed_df = pd.json_normalize(df[source_col].apply(_extract))
    return pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)


def extract_title_and_text(soup: BeautifulSoup) -> Dict[str, Any]:
    return {
        "title": soup.select_one("h1, h2, title").get_text(strip=True)
        if soup.select_one("h1, h2, title")
        else None,
        "text": soup.select_one("p").get_text(strip=True)
        if soup.select_one("p")
        else None,
    }


class NBRepro:
    def __init__(
        self,
        notebook_path,
        output_path=None,
        kernel_name="python3",
        write_output=True,
    ):
        self.notebook_path = notebook_path
        if write_output:
            if output_path is None:
                output_path = notebook_path.replace(".ipynb", "-out.ipynb")
            self.output_path = output_path
        else:
            self.output_path = None
        self.kernel_name = kernel_name
        self._nb = None
        self._rerun_nb = None

        self.registration_code = ""
        self.type_analyzers = {}
        self.output_comparators = {}
        self.numeric_tolerance = 1e-9

        self.register_output_comparator(np.ndarray, self._compare_numpy_arrays)
        self.register_output_comparator(float, self._compare_floats)

    def read_nb(self):
        try:
            with open(self.notebook_path, encoding="utf-8") as f:
                return nbformat.read(f, as_version=4)
        except UnicodeDecodeError:
            with open(self.notebook_path, encoding="latin-1") as f:
                return nbformat.read(f, as_version=4)

    @property
    def nb(self):
        if self._nb is None:
            self._nb = self.read_nb()
        return self._nb

    @property
    def rerun_nb(self):
        if self._rerun_nb is None:
            self._rerun_nb = self.rerun()
        return self._rerun_nb

    def register_type_analyzer(self, type_or_predicate, analyzer_func):
        self.type_analyzers[type_or_predicate] = analyzer_func

    def register_output_comparator(self, type_or_predicate, comparator_func):
        self.output_comparators[type_or_predicate] = comparator_func

    def rerun(self):
        setup_code = (
            "import final_type_analysis as type_analysis\n"
            "type_analysis.setup_type_formatter()\n"
        )
        nb = self.read_nb()
        client = NotebookClient(nb, timeout=120, kernel_name=self.kernel_name)

        with client.setup_kernel():
            client.kc.execute(setup_code, silent=False, store_history=False)
            self.register_analyzer_functions(client)

            for idx, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                original_source = cell.source
                var_name = self._trailing_assignment_var(cell.source)
                if var_name is not None:
                    cell.source = cell.source + "\n" + var_name
                client.execute_cell(cell, cell_index=idx)
                cell.source = original_source

        if self.output_path is not None:
            with open(self.output_path, "w", encoding="utf-8", newline="\n") as f:
                nbformat.write(nb, f)

        return nb

    _SIMPLE_ASSIGNMENT_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*=[^=]")

    @staticmethod
    def _trailing_assignment_var(source):
        lines = source.splitlines()
        for line in reversed(lines):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            if stripped.startswith(("def ", "class ", "@")):
                return None
            m = NBRepro._SIMPLE_ASSIGNMENT_RE.match(line)
            if m:
                return m.group(1)
            return None
        return None

    @staticmethod
    def _normalize_text(value):
        if isinstance(value, list):
            return "".join(str(part) for part in value)
        return "" if value is None else str(value)

    @staticmethod
    def _extract_mime_data(output):
        return output.get("data", {}) if output.output_type in ("execute_result", "display_data") else {}

    @staticmethod
    def _load_json(raw):
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _type_name_from_data(self, before_data, after_data):
        for data in (after_data, before_data):
            type_info = self._load_json(data.get("application/x-python-type"))
            if type_info:
                return type_info.get("type_name")
        return None

    def _dataframe_kind_from_data(self, before_data, after_data):
        for data in (after_data, before_data):
            df_info = self._load_json(data.get("application/x-python-dataframe"))
            if df_info.get("dataframe_kind"):
                return df_info["dataframe_kind"]

        for data in (after_data, before_data):
            type_info = self._load_json(data.get("application/x-python-type"))
            module = type_info.get("module", "") or ""
            if type_info.get("type_name") == "DataFrame" and "polars" in module:
                return "polars"
            if type_info.get("type_name") == "DataFrame" and "pandas" in module:
                return "pandas"
            if type_info.get("type_name") == "ndarray" and "numpy" in module:
                return "numpy"

        return None

    def _compare_numpy_arrays(self, a, b):
        try:
            return bool(np.allclose(a, b))
        except Exception:
            return False

    def _compare_floats(self, a, b):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
        except Exception:
            return False

    def _literal_or_raw(self, val):
        if not isinstance(val, str):
            return val

        stripped = val.strip()
        if stripped.startswith("array("):
            try:
                numbers = re.findall(r"[+-]?\d*\.\d+|[+-]?\d+", stripped)
                floats = [float(n) for n in numbers]
                shape_match = re.search(r"shape=\((\d+),\s*(\d+)\)", stripped)
                if shape_match:
                    rows = int(shape_match.group(1))
                    cols = int(shape_match.group(2))
                    if len(floats) == rows * cols:
                        return np.array(floats).reshape((rows, cols))
                if floats:
                    return np.array(floats)
            except Exception:
                pass
            return val

        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return stripped

    def _find_comparator(self, a, b):
        for key, cmpfunc in self.output_comparators.items():
            if isinstance(key, type):
                if isinstance(a, key) and isinstance(b, key):
                    return cmpfunc
            elif callable(key):
                try:
                    if key(a) and key(b):
                        return cmpfunc
                except Exception:
                    pass
        return None

    def auto_parse_dataframe_html(self, df: pd.DataFrame, source_col: str = "fragment") -> pd.DataFrame:
        if source_col in df.columns:
            try:
                return parse_dataframe_html(df, source_col, extract_title_and_text)
            except Exception:
                return df
        return df

    def _parse_dataframe_output_html(self, html_text: str):
        html_text = self._normalize_text(html_text)
        soup = BeautifulSoup(html_text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            raise ValueError("No table elements found in DataFrame HTML output")

        dfs = pd.read_html(StringIO(str(tables[0])))
        if not dfs:
            raise ValueError("No tables parsed from DataFrame HTML output")

        df = dfs[0]
        artifact_tokens = {"...", "\u2026", "\u22ef", "\ufffd"}
        artifact_rows = df.apply(
            lambda row: row.notna().any()
            and all(str(val).strip().strip('"') in artifact_tokens for val in row if pd.notna(val)),
            axis=1,
        )
        return df.loc[~artifact_rows].reset_index(drop=True)

    def _parse_pandas_from_output_data(self, data):
        if "text/html" in data:
            df = self._parse_dataframe_output_html(data["text/html"])
            return self.auto_parse_dataframe_html(df)

        df_info = self._load_json(data.get("application/x-python-dataframe"))
        if df_info.get("dataframe_kind") == "pandas":
            columns = df_info.get("columns")
            return pd.DataFrame(df_info.get("array", []), columns=columns)

        raise ValueError("No pandas DataFrame representation found")

    def _extract_output_value(self, output):
        if output.output_type == "stream":
            return self._normalize_text(output.get("text", ""))

        if output.output_type in ("execute_result", "display_data"):
            data = output.get("data", {})
            if "text/plain" in data:
                return self._normalize_text(data["text/plain"])
            if "text/html" in data:
                return self._normalize_text(data["text/html"])

        return str(output)

    def _approx_similarity_percent(self, before_obj, after_obj, match: bool) -> float:
        if match:
            return 100.0

        if isinstance(before_obj, (int, float, np.number)) and isinstance(after_obj, (int, float, np.number)):
            if pd.isna(before_obj) or pd.isna(after_obj):
                return 0.0
            denom = max(abs(float(before_obj)), abs(float(after_obj)), self.numeric_tolerance)
            delta = abs(float(before_obj) - float(after_obj))
            return round(max(0.0, 1.0 - (delta / denom)) * 100.0, 2)

        score = SequenceMatcher(None, str(before_obj), str(after_obj)).ratio()
        return round(float(score) * 100.0, 2)

    def _compare_basic_output(self, before_out, after_out, out_idx, type_name):
        before_val = self._extract_output_value(before_out)
        after_val = self._extract_output_value(after_out)
        before_obj = self._literal_or_raw(before_val)
        after_obj = self._literal_or_raw(after_val)

        comp = self._find_comparator(before_obj, after_obj)
        if comp is not None:
            match = bool(comp(before_obj, after_obj))
            reason = "custom comparator matched" if match else "custom comparator mismatch"
        else:
            try:
                equality = before_obj == after_obj
                if isinstance(equality, np.ndarray):
                    match = bool(equality.all())
                else:
                    match = bool(equality)
                reason = "direct equality matched" if match else "direct equality mismatch"
            except Exception:
                match = str(before_val).strip() == str(after_val).strip()
                reason = "string-normalized equality matched" if match else "string-normalized equality mismatch"

        approx_percent = self._approx_similarity_percent(before_obj, after_obj, match)
        result = {
            "output_index": out_idx,
            "type_name": type_name,
            "match": match,
            "comparison_path": "kobi_basic",
            "confidence": round(approx_percent / 100.0, 4),
            "details": {
                "reason": reason,
                "approx_match_percent": approx_percent,
            },
        }
        if not match:
            result["before"] = before_val
            result["after"] = after_val
        return result

    def _compare_pandas_dataframes(self, before_df, after_df, out_idx, type_name):
        before_aligned, after_aligned = before_df.align(after_df, join="outer", axis=None)
        comparison = before_aligned.eq(after_aligned) | (before_aligned.isnull() & after_aligned.isnull())
        true_count = int(comparison.to_numpy().sum())
        total_count = int(comparison.size)
        false_count = total_count - true_count
        confidence = round(true_count / total_count, 4) if total_count else 1.0

        return {
            "output_index": out_idx,
            "type_name": type_name,
            "dataframe_kind": "pandas",
            "match": confidence == 1.0,
            "comparison_path": "kobi_pandas_html",
            "coverage": "visible_vs_visible",
            "confidence": confidence,
            "details": {
                "shape_before": list(before_df.shape),
                "shape_after": list(after_df.shape),
                "true_count": true_count,
                "false_count": false_count,
                "confidence_score": confidence,
                "approx_match_percent": round(confidence * 100.0, 2),
            },
        }

    def _compare_pandas_output(self, before_data, after_data, out_idx, type_name):
        try:
            before_df = self._parse_pandas_from_output_data(before_data)
            after_df = self._parse_pandas_from_output_data(after_data)
            return self._compare_pandas_dataframes(before_df, after_df, out_idx, type_name)
        except Exception as exc:
            return {
                "output_index": out_idx,
                "type_name": type_name,
                "dataframe_kind": "pandas",
                "match": False,
                "comparison_path": "kobi_pandas_html",
                "coverage": None,
                "confidence": 0.0,
                "message": f"Pandas comparison failed: {exc}",
            }

    def _compare_polars_full_vs_repr(self, before_data, after_data, out_idx, type_name):
        def _coverage_label(visible_rows, declared_rows):
            return "full" if visible_rows == declared_rows else "truncated"

        def _make_result(path, coverage, confidence):
            confidence = round(confidence, 4)
            return {
                "output_index": out_idx,
                "type_name": type_name,
                "dataframe_kind": "polars",
                "comparison_path": path,
                "coverage": coverage,
                "confidence": confidence,
                "match": confidence == 1.0,
            }

        df_mime = "application/x-python-dataframe"
        after_df_info = self._load_json(after_data.get(df_mime))
        if not after_df_info:
            return None

        after_shape = after_df_info["shape"]
        after_array = after_df_info["array"]

        try:
            before_df = pl.from_repr(before_data["text/plain"])
        except Exception:
            return None

        declared_m = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", before_data["text/plain"])
        before_shape = (
            [int(declared_m.group(1)), int(declared_m.group(2))]
            if declared_m
            else [before_df.height, before_df.width]
        )

        coverage = (
            _coverage_label(before_df.height, before_shape[0])
            + "_vs_"
            + _coverage_label(after_shape[0], after_shape[0])
        )
        shape_match = before_shape == after_shape
        shape_score = 0.2 if shape_match else 0.0
        num_cols = before_shape[1] if before_shape[1] else before_df.width
        comparable_rows = min(before_df.height, after_shape[0])
        total_cells = comparable_rows * 2 * num_cols

        if not shape_match or total_cells == 0:
            data_ratio = 1.0 if total_cells == 0 and shape_match else 0.0
            return _make_result("polars_full_vs_repr", coverage, shape_score + 0.8 * data_ratio)

        sorted_cols = sorted(before_df.columns)
        before_sorted = before_df.select(sorted_cols).sort(sorted_cols)
        before_array = before_sorted.to_numpy(allow_copy=True).tolist()

        def _sanitize(val):
            if isinstance(val, float) and math.isnan(val):
                return None
            return val

        before_array = [[_sanitize(v) for v in row] for row in before_array]

        if before_df.height == after_shape[0]:
            matched = sum(
                1
                for r in range(len(before_array))
                for c in range(num_cols)
                if before_array[r][c] == after_array[r][c]
            )
            return _make_result("polars_full_vs_repr", coverage, shape_score + 0.8 * ((matched * 2.0) / total_cells))

        after_set = {tuple(row) for row in after_array}
        matched_rows = sum(1 for row in before_array if tuple(row) in after_set)
        matched_cells = matched_rows * num_cols * 2.0
        return _make_result("polars_full_vs_repr", coverage, shape_score + 0.8 * (matched_cells / total_cells))

    def _compare_dataframe_mime(self, before_data, after_data, out_idx, type_name, kind):
        df_mime = "application/x-python-dataframe"
        before_df = self._load_json(before_data.get(df_mime))
        after_df = self._load_json(after_data.get(df_mime))
        match = before_df.get("array") == after_df.get("array") and before_df.get("shape") == after_df.get("shape")
        result = {
            "output_index": out_idx,
            "type_name": type_name,
            "dataframe_kind": kind,
            "match": match,
            "comparison_path": "dataframe_mime",
            "coverage": "full_vs_full",
            "confidence": 1.0 if match else 0.0,
        }
        if not match:
            result["message"] = "DataFrame/array content differs"
            result["shape_before"] = before_df.get("shape")
            result["shape_after"] = after_df.get("shape")
        return result

    def _compare_output(self, before_out, after_out, out_idx):
        if before_out.output_type == "stream" or after_out.output_type == "stream":
            if before_out.output_type != after_out.output_type:
                return {
                    "output_index": out_idx,
                    "match": False,
                    "comparison_path": "stream",
                    "confidence": 0.0,
                    "message": f"Output type differs: {before_out.output_type} vs {after_out.output_type}",
                }
            match = before_out.get("text", "") == after_out.get("text", "")
            return {
                "output_index": out_idx,
                "match": match,
                "comparison_path": "stream",
                "confidence": 1.0 if match else 0.0,
            }

        before_data = self._extract_mime_data(before_out)
        after_data = self._extract_mime_data(after_out)
        type_name = self._type_name_from_data(before_data, after_data)
        dataframe_kind = self._dataframe_kind_from_data(before_data, after_data)
        df_mime = "application/x-python-dataframe"

        if dataframe_kind == "polars":
            if "text/plain" in before_data and df_mime in after_data:
                result = self._compare_polars_full_vs_repr(before_data, after_data, out_idx, type_name)
                if result is not None:
                    return result
            if "text/plain" in before_data and "text/plain" in after_data:
                result = compare_polars_with_confidence(
                    self._normalize_text(before_data["text/plain"]),
                    self._normalize_text(after_data["text/plain"]),
                    before_html=before_data.get("text/html"),
                    after_html=after_data.get("text/html"),
                )
                result.update({
                    "output_index": out_idx,
                    "type_name": type_name,
                    "dataframe_kind": "polars",
                })
                return result

        if dataframe_kind == "pandas":
            return self._compare_pandas_output(before_data, after_data, out_idx, type_name)

        if df_mime in before_data and df_mime in after_data:
            return self._compare_dataframe_mime(before_data, after_data, out_idx, type_name, dataframe_kind)

        return self._compare_basic_output(before_out, after_out, out_idx, type_name)

    def compare(self):
        before_nb = self.nb
        after_nb = self.rerun_nb
        comparison_results = []

        for idx, (before_cell, after_cell) in enumerate(zip(before_nb.cells, after_nb.cells)):
            if before_cell.cell_type != "code":
                continue

            before_outputs = before_cell.get("outputs", [])
            after_outputs = after_cell.get("outputs", [])
            cell_result = {"cell_index": idx, "match": True, "outputs": []}

            if len(before_outputs) != len(after_outputs):
                cell_result["match"] = False
                cell_result["outputs"].append({
                    "match": False,
                    "comparison_path": "output_count",
                    "confidence": 0.0,
                    "message": f"Output count differs: {len(before_outputs)} vs {len(after_outputs)}",
                })
                comparison_results.append(cell_result)
                continue

            for out_idx, (before_out, after_out) in enumerate(zip(before_outputs, after_outputs)):
                output_result = self._compare_output(before_out, after_out, out_idx)
                cell_result["outputs"].append(output_result)
                if not output_result.get("match", True):
                    cell_result["match"] = False

            comparison_results.append(cell_result)

        return comparison_results

    def get_cell_type_info(self, cell_or_idx):
        if isinstance(cell_or_idx, int):
            cell = self.rerun_nb.cells[cell_or_idx]
        else:
            cell = cell_or_idx
        type_infos = []
        if cell.cell_type == "code":
            for output in cell.outputs:
                if output.output_type == "execute_result":
                    type_info = self._load_json(output.get("data", {}).get("application/x-python-type"))
                    if type_info:
                        type_infos.append(type_info)
        return type_infos

    def get_type_info(self):
        type_infos = {}
        for idx, cell in enumerate(self.rerun_nb.cells):
            if cell.cell_type == "code":
                type_infos[idx] = self.get_cell_type_info(cell)
        return type_infos

    def register_analyzer_functions(self, client):
        for type_or_predicate, analyzer_func in self.type_analyzers.items():
            analyzer_name = analyzer_func.__name__
            analyzer_source = inspect.getsource(analyzer_func)

            if inspect.isclass(type_or_predicate):
                type_name = type_or_predicate.__name__
                registration_code = f"""
{analyzer_source}
type_analysis.register_analyzer({type_name}, {analyzer_name})
"""
            else:
                predicate_name = type_or_predicate.__name__
                predicate_source = inspect.getsource(type_or_predicate)
                registration_code = f"""
{predicate_source}
{analyzer_source}
type_analysis.register_analyzer({predicate_name}, {analyzer_name})
"""
            client.kc.execute(registration_code, silent=False, store_history=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python final_nb_repro_start.py <notebook_path>")
        sys.exit(1)

    repro = NBRepro(sys.argv[1])
    repro.rerun()
    print(json.dumps(repro.compare(), indent=2, default=str))
