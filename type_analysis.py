import inspect
import json
import re
import math

PREDICATE_ANALYZERS = {}


def parse_polars_html(html_str):
    """Parse a Polars DataFrame HTML output into a structured dict."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_str, "html.parser")

    # Parse shape from <small>shape: (R, C)</small>
    small_tag = soup.find("small")
    shape = None
    if small_tag:
        m = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", small_tag.get_text())
        if m:
            shape = (int(m.group(1)), int(m.group(2)))

    # Parse thead rows
    thead = soup.find("thead")
    columns = []
    dtypes = []
    if thead:
        header_rows = thead.find_all("tr")
        if len(header_rows) >= 1:
            columns = [th.get_text(strip=True) for th in header_rows[0].find_all("th")]
        if len(header_rows) >= 2:
            dtypes = [td.get_text(strip=True) for td in header_rows[1].find_all("td")]

    # Parse tbody rows
    tbody = soup.find("tbody")
    head_rows = []
    tail_rows = []
    truncated = False

    # strip the truncated data
    def _clean_cell(text):
        text = text.strip()
        if text == "null":
            return None
        # Strip surrounding quotes from string values (double or single)
        for q in ('"', "'"):
            if len(text) >= 2 and text[0] == q and text[-1] == q:
                return text[1:-1]
            # Strip lone leading quote when closing quote was truncated with the value
            if text.startswith(q) and (text.endswith("…") or text.endswith("...")):
                return text[1:]
        return text

    if tbody:
        found_ellipsis = False
        for tr in tbody.find_all("tr"):
            cells = [td.get_text() for td in tr.find_all("td")]
            # Check if all cells are ellipsis
            if all(c.strip() in ("…", "&hellip;", "\u2026") for c in cells):
                truncated = True
                found_ellipsis = True
                continue
            row_dict = {}
            for col_name, cell_text in zip(columns, cells):
                row_dict[col_name] = _clean_cell(cell_text)
            if found_ellipsis:
                tail_rows.append(row_dict)
            else:
                head_rows.append(row_dict)

    missing_rows = 0
    if truncated and shape:
        missing_rows = shape[0] - len(head_rows) - len(tail_rows)

    return {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "head_rows": head_rows,
        "tail_rows": tail_rows,
        "truncated": truncated,
        "missing_rows": missing_rows,
    }


def compare_polars_html(before_html, after_html):
    """Compare two Polars DataFrame HTML outputs."""
    before = parse_polars_html(before_html)
    after = parse_polars_html(after_html)

    shape_match = before["shape"] == after["shape"]
    schema_match = (before["columns"] == after["columns"] and
                    before["dtypes"] == after["dtypes"])
    data_match = (before["head_rows"] == after["head_rows"] and
                  before["tail_rows"] == after["tail_rows"])

    is_truncated = before["truncated"] or after["truncated"]
    match = shape_match and schema_match and data_match

    result = {
        "match": match,
        "truncated": is_truncated,
        "missing_rows_before": before["missing_rows"],
        "missing_rows_after": after["missing_rows"],
        "shape_match": shape_match,
        "schema_match": schema_match,
        "data_match": data_match,
    }

    if not match:
        details = {}
        if not shape_match:
            details["shape_before"] = before["shape"]
            details["shape_after"] = after["shape"]
        if not schema_match:
            if before["columns"] != after["columns"]:
                details["columns_before"] = before["columns"]
                details["columns_after"] = after["columns"]
            if before["dtypes"] != after["dtypes"]:
                details["dtypes_before"] = before["dtypes"]
                details["dtypes_after"] = after["dtypes"]
        if not data_match:
            if before["head_rows"] != after["head_rows"]:
                details["head_rows_differ"] = True
            if before["tail_rows"] != after["tail_rows"]:
                details["tail_rows_differ"] = True
        result["details"] = details

    return result


def _cells_match(before_val, after_val):
    """Compare two cell values, tolerating Polars truncation.
    Returns: 'full', 'partial' (truncation match), or 'mismatch'.
    """
    if before_val == after_val:
        return "full"
    # Check truncation: one side ends with … or ... and is a substring prefix of the other
    for a, b in [(before_val, after_val), (after_val, before_val)]:
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a.endswith("…"):
            prefix = a[:-1]
        elif a.endswith("..."):
            prefix = a[:-3]
        else:
            continue
        if b.startswith(prefix):
            return "partial"
    return "mismatch"


def compare_polars_with_confidence(before_text, after_text,
                                    before_html=None, after_html=None):
    """Compare two Polars DataFrame text reprs with confidence scoring.

    confidence = 0.2 (shape match) + 0.8 * (matched_cells / total_comparable_cells)
    where total_comparable_cells counts only visible/reconstructed cells, not
    the declared shape.  match = True only when confidence == 1.0.

    Path 1 — from_repr:      total_comparable_cells = (b_df.height + a_df.height) * num_cols
    Path 2 — html_fallback:  total_comparable_cells = (visible_b + visible_a) * num_cols
             Cells scored by _cells_match (full=1.0, partial=0.5, mismatch=0.0).

    Always returns: {comparison_path, coverage, confidence, match}.
    """
    import polars as pl
    import numpy as np

    def _coverage_label(visible_rows, declared_rows):
        return "full" if visible_rows == declared_rows else "truncated"

    def _make_result(path, coverage, confidence):
        confidence = round(confidence, 4)
        return {
            "comparison_path": path,
            "coverage": coverage,
            "confidence": confidence,
            "match": confidence == 1.0,
        }

    try:
        b_df = pl.from_repr(before_text)
        a_df = pl.from_repr(after_text)
    except Exception:
        # ── HTML/BS4 path ─────────────────────────────────────────────────────
        if not before_html or not after_html:
            return {"comparison_path": "error", "coverage": None,
                    "confidence": 0.0, "match": False}

        before = parse_polars_html(before_html)
        after  = parse_polars_html(after_html)

        b_visible = len(before["head_rows"]) + len(before["tail_rows"])
        a_visible = len(after["head_rows"])  + len(after["tail_rows"])
        b_declared = before["shape"][0] if before["shape"] else b_visible
        a_declared = after["shape"][0]  if after["shape"]  else a_visible
        coverage = (_coverage_label(b_visible, b_declared)
                    + "_vs_"
                    + _coverage_label(a_visible, a_declared))

        shapes_match = (before["shape"] is not None
                        and after["shape"] is not None
                        and before["shape"] == after["shape"])
        shape_score = 0.2 if shapes_match else 0.0

        num_cols = len(before["columns"])

        if not shapes_match or before["columns"] != after["columns"]:
            data_ratio = 1.0 if (shapes_match
                                 and before["columns"] == after["columns"]
                                 and num_cols == 0) else 0.0
            return _make_result("html_fallback", coverage,
                                shape_score + 0.8 * data_ratio)

        comparable_head = min(len(before["head_rows"]), len(after["head_rows"]))
        comparable_tail = min(len(before["tail_rows"]), len(after["tail_rows"]))
        total_comparable_cells = (comparable_head + comparable_tail) * 2 * num_cols

        matched_cells = 0.0
        for i in range(comparable_head):
            for col in before["columns"]:
                bv = before["head_rows"][i].get(col)
                av = after["head_rows"][i].get(col)
                result = _cells_match(bv, av)
                if result == "full":
                    matched_cells += 1.0
                elif result == "partial":
                    matched_cells += 0.5

        for i in range(comparable_tail):
            for col in before["columns"]:
                bv = before["tail_rows"][i].get(col)
                av = after["tail_rows"][i].get(col)
                result = _cells_match(bv, av)
                if result == "full":
                    matched_cells += 1.0
                elif result == "partial":
                    matched_cells += 0.5

        data_ratio = matched_cells / total_comparable_cells if total_comparable_cells > 0 else 1.0
        return _make_result("html_fallback", coverage,
                            shape_score + 0.8 * data_ratio)

    # ── from_repr path ────────────────────────────────────────────────────────
    declared_b = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", before_text)
    declared_a = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", after_text)
    b_shape = (int(declared_b.group(1)), int(declared_b.group(2))) if declared_b else (b_df.height, b_df.width)
    a_shape = (int(declared_a.group(1)), int(declared_a.group(2))) if declared_a else (a_df.height, a_df.width)

    b_declared_rows = b_shape[0]
    a_declared_rows = a_shape[0]
    coverage = (_coverage_label(b_df.height, b_declared_rows)
                + "_vs_"
                + _coverage_label(a_df.height, a_declared_rows))

    shapes_match = b_shape == a_shape
    shape_score = 0.2 if shapes_match else 0.0

    num_cols = b_df.width
    total_cells = b_df.height * num_cols + a_df.height * num_cols

    if not shapes_match or b_df.columns != a_df.columns or total_cells == 0:
        data_ratio = 1.0 if (total_cells == 0 and shapes_match
                             and b_df.columns == a_df.columns) else 0.0
        return _make_result("from_repr", coverage,
                            shape_score + 0.8 * data_ratio)

    sorted_cols = sorted(b_df.columns)
    b_sorted = b_df.select(sorted_cols).sort(sorted_cols)
    a_sorted = a_df.select(sorted_cols).sort(sorted_cols)

    try:
        b_arr = b_sorted.to_numpy(allow_copy=True)
        a_arr = a_sorted.to_numpy(allow_copy=True)
        try:
            eq_mask = (b_arr == a_arr)
            # Handle NaN == NaN
            both_nan = np.isnan(b_arr) & np.isnan(a_arr)
            eq_mask = eq_mask | both_nan
        except (TypeError, ValueError):
            b_list = b_arr.tolist()
            a_list = a_arr.tolist()
            min_rows = min(len(b_list), len(a_list))
            matched = sum(1 for r in range(min_rows)
                          for c in range(num_cols)
                          if b_list[r][c] == a_list[r][c])
            matched_cells = matched * 2.0
            data_ratio = matched_cells / total_cells
            return _make_result("from_repr", coverage,
                                shape_score + 0.8 * data_ratio)
    except Exception:
        try:
            all_match = b_sorted.frame_equal(a_sorted, null_equal=True)
        except AttributeError:
            all_match = b_sorted.equals(a_sorted, null_equal=True)
        data_ratio = 1.0 if all_match else 0.0
        return _make_result("from_repr", coverage,
                            shape_score + 0.8 * data_ratio)

    min_rows = min(b_arr.shape[0], a_arr.shape[0])
    matched = int(eq_mask[:min_rows].sum())
    matched_cells = matched * 2.0
    data_ratio = matched_cells / total_cells
    return _make_result("from_repr", coverage,
                        shape_score + 0.8 * data_ratio)


def register_analyzer(type_or_predicate, analyzer_func):
    """Register analyzer for either a type/class or a predicate function"""
    if inspect.isclass(type_or_predicate):
        # It's a class - use isinstance check
        def class_predicate(obj):
            return isinstance(obj, type_or_predicate)
        class_predicate.__name__ = f"is_{type_or_predicate.__name__}"
        PREDICATE_ANALYZERS[class_predicate] = analyzer_func
    else:
        # It's already a predicate function
        PREDICATE_ANALYZERS[type_or_predicate] = analyzer_func


def analyze_basic_type(obj, depth, max_depth, context, recurse_fn):
    """Basic type analyzer that captures core type information"""
    obj_type = type(obj)

    return {
        "type_name": obj_type.__name__,
        "module": getattr(obj_type, "__module__", None),
        "qualname": getattr(obj_type, "__qualname__", obj_type.__name__),
        "mro": [cls.__name__ for cls in obj_type.__mro__],
        "builtin": obj_type.__module__ == "builtins",
    }


def analyze_type(obj, depth=0, max_depth=3, context=None):
    """Main analysis function - only needs to check predicates"""
    if depth > max_depth:
        return {"type_name": type(obj).__name__, "truncated": True}
    
    # Start with basic type info
    base_info = analyze_basic_type(obj, depth, max_depth, context, None)
    
    # Check all registered predicates
    for predicate, analyzer in PREDICATE_ANALYZERS.items():
        if predicate(obj):
            specialized_info = analyzer(obj, depth, max_depth, context, analyze_type)
            return {**base_info, **specialized_info}
    
    return base_info


def setup_type_formatter():
    # Setup type formatter
    ip = get_ipython()  # noqa: F821
    original_format = ip.display_formatter.format

    def format_with_type(obj, include=None, exclude=None):
        format_dict, md_dict = original_format(obj, include, exclude)
        if format_dict:
            format_dict["application/x-python-type"] = json.dumps(analyze_type(obj))
            if _is_dataframe_or_array(obj):
                format_dict["application/x-python-dataframe"] = json.dumps(
                    analyze_dataframe(obj, 0, 3, None, analyze_type)
                )
            # Store parsed Polars HTML for reliable schema comparison
            obj_type = type(obj)
            obj_module = getattr(obj_type, "__module__", "") or ""
            if (obj_type.__name__ == "DataFrame" and "polars" in obj_module
                    and "text/html" in format_dict):
                parsed = parse_polars_html(format_dict["text/html"])
                format_dict["application/x-polars-html-parsed"] = json.dumps(parsed)
        return format_dict, md_dict

    ip.display_formatter.format = format_with_type


def analyze_container(obj, depth, max_depth, context, recurse_fn):
    """Container analyzer that extends basic type info"""
    info = {"length": len(obj), "empty": len(obj) == 0}

    if len(obj) > 0 and depth < max_depth:
        sample = list(obj)[:5]  # Sample first 5
        info["element_types"] = [
            recurse_fn(item, depth + 1, max_depth, context) for item in sample
        ]

        # Summary of element type names for quick inspection
        info["element_type_names"] = list(
            set(elem_info["type_name"] for elem_info in info["element_types"])
        )

    return info


def _is_dataframe_or_array(obj):
    """Predicate to check if obj is a pandas/polars DataFrame or numpy ndarray"""
    type_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "") or ""
    if type_name == "DataFrame" and ("pandas" in module or "polars" in module):
        return True
    if type_name == "ndarray" and "numpy" in module:
        return True
    return False


def analyze_dataframe(obj, depth, max_depth, context, recurse_fn):
    """Analyzer for pandas/polars DataFrames and numpy ndarrays"""
    type_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "") or ""

    if type_name == "DataFrame" and "pandas" in module:
        sorted_df = obj[sorted(obj.columns)]
        sorted_df = sorted_df.sort_values(by=list(sorted_df.columns)).reset_index(drop=True)
        arr = sorted_df.to_numpy()
    elif type_name == "DataFrame" and "polars" in module:
        sorted_cols = sorted(obj.columns)
        sorted_df = obj.select(sorted_cols).sort(sorted_cols)
        arr = sorted_df.to_numpy(allow_copy=True)
    else:
        arr = obj

    arr_list = arr.tolist()

    # Replace NaN with None so they serialize as null in JSON
    def _sanitize(val):
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

    if arr.ndim == 2:
        arr_list = [[_sanitize(v) for v in row] for row in arr_list]
    elif arr.ndim == 1:
        arr_list = [_sanitize(v) for v in arr_list]

    return {
        "array": arr_list,
        "shape": list(arr.shape),
    }


DEFAULT_ANALYZERS = {
    list: analyze_container,
    tuple: analyze_container,
    set: analyze_container,
    frozenset: analyze_container,
    dict: analyze_container,
    lambda obj: hasattr(obj, "__iter__")
    and not isinstance(obj, (str, bytes)): analyze_container,
    _is_dataframe_or_array: analyze_dataframe,
}

for c, analyzer_f in DEFAULT_ANALYZERS.items():
    register_analyzer(c, analyzer_f)
