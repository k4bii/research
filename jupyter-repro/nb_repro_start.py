import nbformat
from nbclient import NotebookClient
import inspect
import json
import ast
import re
from io import StringIO
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import polars as pl
import math
from bs4 import BeautifulSoup
from typing import Dict, Any, Callable


def parse_dataframe_html(
    df: pd.DataFrame,
    source_col: str,
    parser: Callable[[BeautifulSoup], Dict[str, Any]],
    html_parser: str = "html.parser"
) -> pd.DataFrame:
    """
    Parse HTML fragments in a DataFrame column and append extracted fields as new columns.
    """
    def _extract(html):
        if pd.isna(html) or html is None:
            return {}
        soup = BeautifulSoup(str(html), html_parser)
        return parser(soup) or {}

    parsed_df = pd.json_normalize(df[source_col].apply(_extract))
    return pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)


def extract_title_and_text(soup: BeautifulSoup) -> Dict[str, Any]:
    return {
        "title": soup.select_one("h1, h2, title").get_text(strip=True) if soup.select_one("h1, h2, title") else None,
        "text": soup.select_one("p").get_text(strip=True) if soup.select_one("p") else None,
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
        self.skip_rows = True
        
        # register built-in comparators for common data types
        try:
            def _np_equal(a, b):
                try:
                    result = np.allclose(a, b)
                    return result
                except Exception:
                    return False

            self.register_output_comparator(np.ndarray, _np_equal)
        except ImportError:
            pass

        try:
            def _pd_equal(a, b):
                try:
                    df1 = pd.read_html(StringIO(str(a)))[0]
                    df2 = pd.read_html(StringIO(str(b)))[0]

                    df1, df2 = df1.align(df2, join="outer", axis=None)
                    comparison = df1.eq(df2) | (df1.isnull() & df2.isnull())
                    return comparison.all().all()
                except Exception:
                    return False

            self.register_output_comparator(pd.DataFrame, _pd_equal)
        except ImportError:
            pass
            
        try:
            def _pl_equal(a, b):
                try:
                    c = pl.from_repr(a)
                    d = pl.from_repr(b)
                    return c.frame_equal(d)
                except Exception:
                    return False

            self.register_output_comparator(pl.DataFrame, _pl_equal)
        except ImportError:
            pass

        def _float_close(a, b):
            try:
                return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
            except Exception:
                return False

        self.register_output_comparator(float, _float_close)


    def read_nb(self):
        try:
            with open(self.notebook_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except UnicodeDecodeError:
            with open(self.notebook_path, encoding="latin-1") as f:
                nb = nbformat.read(f, as_version=4)
        return nb

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

    def register_output_comparator(self, type_name, comparator_func):
        self.output_comparators[type_name] = comparator_func

    def rerun(self):
        # Read setup code
        setup_code = "import type_analysis\ntype_analysis.setup_type_formatter()\n"

        nb = self.read_nb()

        client = NotebookClient(nb, timeout=60, kernel_name=self.kernel_name)

        with client.setup_kernel():
            # execute setup code
            client.kc.execute(setup_code, silent=False, store_history=False)
            self.register_analyzer_functions(client)

            # execute cell by cell
            for idx, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    client.execute_cell(cell, cell_index=idx)

        # write output if needed (ensure UTF-8 on Windows)
        if self.output_path is not None:
            with open(self.output_path, "w", encoding="utf-8", newline="\n") as f:
                nbformat.write(nb, f)

            return nb

    def _literal_or_raw(self, val):
        if isinstance(val, str):
            if val.strip().startswith('array('):
                try:
                    # Extract numbers using regex
                    numbers = re.findall(r'[+-]?\d*\.\d+|[+-]?\d+', val)
                    floats = [float(n) for n in numbers]
                    
                    # Extract shape from the string, e.g., 'shape=(5, 5)'
                    shape_match = re.search(r'shape=\((\d+),\s*(\d+)\)', val)
                    if shape_match:
                        rows = int(shape_match.group(1))
                        cols = int(shape_match.group(2))
                        if len(floats) == rows * cols:
                            result = np.array(floats).reshape((rows, cols))
                            return result
                    
                    # If no shape or wrong length, try to infer as 1D
                    if len(floats) > 0:
                        result = np.array(floats)
                        return result
                        
                except Exception:
                    pass
                return val
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val.strip()
        else:
            return val

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
        """Auto-parse HTML in a source DataFrame column when present."""
        if source_col in df.columns:
            try:
                return parse_dataframe_html(df, source_col, extract_title_and_text)
            except Exception:
                return df
        return df

    def _parse_dataframe_output_html(self, html_text: str):
        """Parse notebook DataFrame HTML through BeautifulSoup before pandas table extraction."""
        if isinstance(html_text, list):
            html_text = "".join(str(part) for part in html_text)

        soup = BeautifulSoup(str(html_text), "html.parser")
        tables = soup.find_all("table")
        if not tables:
            raise ValueError("No table elements found in DataFrame HTML output")

        table_html = str(tables[0])
        dfs = pd.read_html(StringIO(table_html))
        if not dfs:
            raise ValueError("No tables parsed from DataFrame HTML output")

        df = dfs[0]
        artifact_tokens = {"...", "…", "⋯", "�"}
        artifact_rows = df.apply(
            lambda row: row.notna().any()
            and all(str(val).strip().strip('"') in artifact_tokens for val in row if pd.notna(val)),
            axis=1,
        )
        return df.loc[~artifact_rows].reset_index(drop=True)

    def extract_output_value(self, output):
        if output.output_type == "execute_result":
            data = output.get("data", {})
            
            # Check for DataFrame types and parse accordingly
            if "application/x-python-type" in data:
                try:
                    type_info = json.loads(data["application/x-python-type"])
                    obj_type = type_info.get("type_name")
                    module = type_info.get("module", "")
                    
                    if obj_type == "DataFrame":
                        if "text/html" in data:
                            try:
                                df = self._parse_dataframe_output_html(data["text/html"])
                                df = self.auto_parse_dataframe_html(df)
                                if "polars" in module:
                                    return pl.from_pandas(df)
                                return df
                            except Exception:
                                pass  # Fall back
                        elif "text/plain" in data:
                            # Try parsing text table for Polars
                            if "polars" in module:
                                df_dict = self.parse_polars_table(data["text/plain"])
                                if df_dict:
                                    df = pl.DataFrame(df_dict)
                                    # no pandas HTML parse path for text/plain here
                                    return df
                            # For pandas, could add similar parsing if needed
                except Exception:
                    pass  # Fall back to default
            
            if "text/html" in data:
                try:
                    df = self._parse_dataframe_output_html(data["text/html"])
                    df = self.auto_parse_dataframe_html(df)
                    if df is not None and not df.empty:
                        return df
                except Exception:
                    pass

            if "text/plain" in data:
                return data["text/plain"]
            elif "text/html" in data:
                return data["text/html"]
            else:
                return str(output)
        else:
            return str(output)


    def _approx_similarity_percent(self, before_obj, after_obj, match: bool) -> float:
        """Estimate an approximate similarity percentage for generic outputs."""
        if match:
            return 100.0

        # Numeric closeness as a percentage.
        if isinstance(before_obj, (int, float, np.number)) and isinstance(after_obj, (int, float, np.number)):
            if pd.isna(before_obj) or pd.isna(after_obj):
                return 0.0
            denom = max(abs(float(before_obj)), abs(float(after_obj)), self.numeric_tolerance)
            delta = abs(float(before_obj) - float(after_obj))
            score = max(0.0, 1.0 - (delta / denom))
            return round(float(score) * 100.0, 2)

        # Text-like closeness for strings and other objects via repr fallback.
        before_text = str(before_obj)
        after_text = str(after_obj)
        score = SequenceMatcher(None, before_text, after_text).ratio()
        return round(float(score) * 100.0, 2)

    def compare(self):
        before_nb = self.nb
        after_nb = self.rerun_nb

        comparison_results = []

        # Iterate through corresponding cells of both notebooks
        for cell_idx, (before_cell, after_cell) in enumerate(zip(before_nb.cells, after_nb.cells)):
            if before_cell.cell_type != "code":
                continue
            
            # Extract outputs from both cells
            before_outputs = before_cell.get("outputs", [])
            after_outputs = after_cell.get("outputs", [])
            
            # Compare each corresponding output
            for output_idx, (before_out, after_out) in enumerate(zip(before_outputs, after_outputs)):
                if before_out.output_type != after_out.output_type:
                    comparison_results.append({
                        "cell": cell_idx,
                        "output": output_idx,
                        "match": False,
                        "reason": "output type mismatch"
                    })
                    continue
                
                # Extract text/data from outputs
                before_val = self.extract_output_value(before_out)
                after_val = self.extract_output_value(after_out)
                
                # convert text to objects when possible
                before_obj = self._literal_or_raw(before_val)
                after_obj = self._literal_or_raw(after_val)

                comp = self._find_comparator(before_obj, after_obj)
                if comp is not None:
                    comp_result = comp(before_obj, after_obj)
                    match = bool(comp_result)
                    reason = "custom comparator matched" if match else "custom comparator mismatch"
                    details = {}
                else:
                    try:
                        match = before_obj == after_obj
                        reason = "direct equality matched" if match else "direct equality mismatch"
                    except Exception:
                        match = str(before_val).strip() == str(after_val).strip()
                        reason = "string-normalized equality matched" if match else "string-normalized equality mismatch"
                    details = {}

                if isinstance(before_obj, pd.DataFrame) and isinstance(after_obj, pd.DataFrame):
                    try:
                        df1, df2 = before_obj.align(after_obj, join="outer", axis=None)
                        comparison = df1.eq(df2) | (df1.isnull() & df2.isnull())
                        true_count = int(comparison.to_numpy().sum())
                        total_count = int(comparison.size)
                        false_count = total_count - true_count
                        confidence_score = round(true_count / total_count, 2) if total_count else 1.0

                        details["true_count"] = true_count
                        details["false_count"] = false_count
                        details["confidence_score"] = confidence_score
                        details["approx_match_percent"] = round(confidence_score * 100.0, 2)

                        if 7 <= cell_idx <= 10:
                            output_lines = []
                            for col in comparison.columns:
                                if "Unnamed: 0" in str(col):
                                    continue
                                bool_list = [str(val) for val in comparison[col].tolist()]
                                output_lines.append(f"{col}: {bool_list}")
                            output_lines.append(
                                f"confidence: {confidence_score} ({true_count} true, {false_count} false)"
                            )
                            print("\n" + "\n".join(output_lines) + "\n", flush=True)
                    except Exception:
                        pass

                comparison_results.append({
                    "cell": cell_idx,
                    "output": output_idx,
                    "match": match,
                    "reason": reason,
                    "details": details,
                    "before": before_val,
                    "after": after_val,
                })

                # Ensure approximate percentages are always available in result details.
                if not isinstance(comparison_results[-1].get("details"), dict):
                    comparison_results[-1]["details"] = {}
                if "approx_match_percent" not in comparison_results[-1]["details"]:
                    comparison_results[-1]["details"]["approx_match_percent"] = self._approx_similarity_percent(
                        before_obj, after_obj, match
                    )

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
                    type_info = json.loads(
                        output.get("data", {}).get("application/x-python-type")
                    )
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
        """Register analyzer by extracting source and name from function objects"""

        # we have to run this directly in the kernel, hence the strings
        for type_or_predicate, analyzer_func in self.type_analyzers.items():
            analyzer_name = analyzer_func.__name__
            analyzer_source = inspect.getsource(analyzer_func)

            if inspect.isclass(type_or_predicate):
                # It's a class - just pass it through
                type_name = type_or_predicate.__name__
                registration_code = f"""
# Define the analyzer function
{analyzer_source}

# Register with class (will be converted to predicate automatically)
type_analysis.register_analyzer({type_name}, {analyzer_name})
"""

            else:
                # It's a predicate function
                predicate_name = type_or_predicate.__name__
                predicate_source = inspect.getsource(type_or_predicate)

                registration_code = f"""
# Define the predicate function
{predicate_source}

# Define the analyzer function  
{analyzer_source}

# Register with predicate function
type_analysis.register_analyzer({predicate_name}, {analyzer_name})
"""
            client.kc.execute(registration_code, silent=False, store_history=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nb_repro_start.py <notebook_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    repro = NBRepro(notebook_path)
    repro.rerun()
    print(repro.compare())
