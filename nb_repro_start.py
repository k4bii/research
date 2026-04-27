import re

import nbformat
from nbclient import NotebookClient
import inspect
import json
import math
from type_analysis import compare_polars_with_confidence


class NBRepro:
    def __init__(
        self, notebook_path, output_path=None, kernel_name="python3", write_output=True
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

    def read_nb(self):
        with open(self.notebook_path, encoding="utf-8") as f:
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
                    original_source = cell.source
                    var_name = self._trailing_assignment_var(cell.source)
                    if var_name is not None:
                        cell.source = cell.source + "\n" + var_name
                    client.execute_cell(cell, cell_index=idx)
                    cell.source = original_source

        # write output if needed
        if self.output_path is not None:
            with open(self.output_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

        return nb

    _SIMPLE_ASSIGNMENT_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*=[^=]")

    @staticmethod
    def _trailing_assignment_var(source):
        """Return the variable name if the last meaningful line is a simple assignment, else None."""
        lines = source.splitlines()
        # Walk backwards past empty and comment-only lines
        for line in reversed(lines):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            # Skip decorator / def / class lines
            if stripped.startswith(("def ", "class ", "@")):
                return None
            m = NBRepro._SIMPLE_ASSIGNMENT_RE.match(line)
            if m:
                return m.group(1)
            return None
        return None

    def _extract_mime_data(self, output):
        """Extract data dict from an execute_result or display_data output."""
        return output.get("data", {})

    def _compare_polars_full_vs_repr(self, before_data, after_data, out_idx, type_name):
        """Compare a saved Polars text repr (before) against full dataframe MIME (after).

        Returns a result dict on success, or None if from_repr fails (caller
        should fall back to text comparison).

        confidence = 0.2 (shape) + 0.8 * (matched_cells / total_comparable_cells)
        where total_comparable_cells = min(b_visible, a_rows) * 2 * num_cols.
        """
        import polars as pl
        import numpy as np

        def _coverage_label(visible_rows, declared_rows):
            return "full" if visible_rows == declared_rows else "truncated"

        def _make_result(path, coverage, confidence):
            confidence = round(confidence, 4)
            return {
                "output_index": out_idx, "type_name": type_name,
                "comparison_path": path,
                "coverage": coverage,
                "confidence": confidence,
                "match": confidence == 1.0,
            }

        df_mime = "application/x-python-dataframe"
        after_df_info = json.loads(after_data[df_mime])
        after_shape = after_df_info["shape"]  # [rows, cols]
        after_array = after_df_info["array"]  # sorted by all cols, NaN→null

        # Reconstruct before DataFrame from text repr
        try:
            b_df = pl.from_repr(before_data["text/plain"])
        except Exception:
            return None  # let caller fall back

        # Check declared shape from the before text repr header
        declared_m = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", before_data["text/plain"])
        if declared_m:
            before_shape = [int(declared_m.group(1)), int(declared_m.group(2))]
        else:
            before_shape = [b_df.height, b_df.width]

        # Coverage: before visible vs declared, after is always full (from MIME)
        coverage = (_coverage_label(b_df.height, before_shape[0])
                    + "_vs_"
                    + _coverage_label(after_shape[0], after_shape[0]))

        shape_match = before_shape == after_shape
        shape_score = 0.2 if shape_match else 0.0

        num_cols = before_shape[1] if before_shape[1] else b_df.width
        comparable_rows = min(b_df.height, after_shape[0])
        total_cells = comparable_rows * 2 * num_cols

        if not shape_match or total_cells == 0:
            data_ratio = 1.0 if (total_cells == 0 and shape_match) else 0.0
            return _make_result("full_vs_repr", coverage,
                                shape_score + 0.8 * data_ratio)

        # Sort the from_repr DataFrame the same way analyze_dataframe does
        sorted_cols = sorted(b_df.columns)
        b_sorted = b_df.select(sorted_cols).sort(sorted_cols)
        b_array = b_sorted.to_numpy(allow_copy=True).tolist()

        # Sanitize NaN → None to match after_array's JSON convention
        def _sanitize(val):
            if isinstance(val, float) and math.isnan(val):
                return None
            return val

        b_array = [[_sanitize(v) for v in row] for row in b_array]

        # If from_repr captured all rows, direct element-wise comparison
        if b_df.height == after_shape[0]:
            matched = sum(1 for r in range(len(b_array))
                          for c in range(num_cols)
                          if b_array[r][c] == after_array[r][c])
            matched_cells = matched * 2.0
            data_ratio = matched_cells / total_cells
            return _make_result("full_vs_repr", coverage,
                                shape_score + 0.8 * data_ratio)

        # from_repr only captured visible rows — check each repr row exists in full DataFrame
        comparable_rows = b_df.height
        after_set = {tuple(row) for row in after_array}
        matched_rows = sum(1 for row in b_array if tuple(row) in after_set)
        matched_cells = matched_rows * num_cols * 2.0
        data_ratio = matched_cells / total_cells
        return _make_result("full_vs_repr", coverage,
                            shape_score + 0.8 * data_ratio)

    def _compare_output(self, before_out, after_out, out_idx):
        """Compare a single output pair, returns a result dict."""
        # Stream outputs: compare text directly
        if before_out.output_type == "stream" or after_out.output_type == "stream":
            if before_out.output_type != after_out.output_type:
                return {"output_index": out_idx, "match": False,
                        "comparison_path": "stream",
                        "message": f"Output type differs: {before_out.output_type} vs {after_out.output_type}"}
            match = before_out.get("text", "") == after_out.get("text", "")
            result = {"output_index": out_idx, "match": match, "comparison_path": "stream"}
            if not match:
                result["before"] = before_out.get("text", "")
                result["after"] = after_out.get("text", "")
            return result

        # For execute_result / display_data: use MIME data
        before_data = self._extract_mime_data(before_out)
        after_data = self._extract_mime_data(after_out)

        # Determine type_name from either side's type info
        type_name = None
        for data in (after_data, before_data):
            raw = data.get("application/x-python-type")
            if raw:
                type_name = json.loads(raw).get("type_name")
                break

        # 1) Registered comparator
        if type_name and type_name in self.output_comparators:
            return self.output_comparators[type_name](before_data, after_data)

        # 2) Dataframe MIME type on both sides
        df_mime = "application/x-python-dataframe"
        if df_mime in before_data and df_mime in after_data:
            before_df = json.loads(before_data[df_mime])
            after_df = json.loads(after_data[df_mime])
            match = before_df["array"] == after_df["array"] and before_df["shape"] == after_df["shape"]
            result = {"output_index": out_idx, "type_name": type_name, "match": match,
                      "comparison_path": "dataframe_mime"}
            if not match:
                result["message"] = "DataFrame/array content differs"
                if before_df["shape"] != after_df["shape"]:
                    result["shape_before"] = before_df["shape"]
                    result["shape_after"] = after_df["shape"]
            return result

        # 3) Polars text/plain comparison (with HTML fallback when truncated)
        if type_name == "DataFrame" and "text/plain" in before_data and "text/plain" in after_data:
            # Check if it's a Polars DataFrame
            is_polars = False
            for data in (after_data, before_data):
                raw = data.get("application/x-python-type")
                if raw:
                    type_info = json.loads(raw)
                    if "polars" in (type_info.get("module") or ""):
                        is_polars = True
                        break
            if is_polars:
                # 3a) If after has full dataframe data, compare directly against
                #     from_repr of the before text — bypasses display config issues
                if df_mime in after_data:
                    result = self._compare_polars_full_vs_repr(
                        before_data, after_data, out_idx, type_name)
                    if result is not None:
                        return result

                # 3b) Fall back to text/plain repr comparison
                result = compare_polars_with_confidence(
                    before_data["text/plain"], after_data["text/plain"],
                    before_html=before_data.get("text/html"),
                    after_html=after_data.get("text/html"),
                )
                result["output_index"] = out_idx
                result["type_name"] = type_name
                return result

        # 4) Fallback: compare text/plain
        before_text = before_data.get("text/plain", "")
        after_text = after_data.get("text/plain", "")
        match = before_text == after_text
        result = {"output_index": out_idx, "type_name": type_name, "match": match,
                  "comparison_path": "text"}
        if not match:
            result["before"] = before_text
            result["after"] = after_text
        return result

    def compare(self):
        before_nb = self.nb
        after_nb = self.rerun_nb

        comparison_results = []

        for idx, (before_cell, after_cell) in enumerate(zip(before_nb.cells, after_nb.cells)):
            if before_cell.cell_type != "code":
                continue

            before_outputs = before_cell.outputs
            after_outputs = after_cell.outputs

            cell_result = {"cell_index": idx, "match": True, "outputs": []}

            if len(before_outputs) != len(after_outputs):
                cell_result["match"] = False

                def _text_of(out):
                    if out.output_type == "stream":
                        raw = out.get("text", "")
                    elif out.output_type in ("execute_result", "display_data"):
                        raw = out.get("data", {}).get("text/plain", "")
                    else:
                        raw = ""
                    if isinstance(raw, list):
                        return "".join(raw)
                    return raw

                def _collect_text(outputs):
                    parts = [_text_of(o) for o in outputs]
                    return "\n".join(p for p in parts if p)

                before_text = _collect_text(before_outputs)
                after_text = _collect_text(after_outputs)
                output_entry = {
                    "match": False,
                    "message": f"Output count differs: {len(before_outputs)} vs {len(after_outputs)}",
                }
                if before_text:
                    output_entry["before"] = before_text
                if after_text:
                    output_entry["after"] = after_text
                cell_result["outputs"].append(output_entry)
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
