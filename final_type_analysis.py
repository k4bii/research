import inspect
import json
import math
import re

PREDICATE_ANALYZERS = {}


def register_analyzer(type_or_predicate, analyzer_func):
    """Register analyzer for either a type/class or a predicate function."""
    if inspect.isclass(type_or_predicate):
        def class_predicate(obj):
            return isinstance(obj, type_or_predicate)

        class_predicate.__name__ = f"is_{type_or_predicate.__name__}"
        PREDICATE_ANALYZERS[class_predicate] = analyzer_func
    else:
        PREDICATE_ANALYZERS[type_or_predicate] = analyzer_func


def analyze_basic_type(obj, depth, max_depth, context, recurse_fn):
    """Kobi's basic type analyzer."""
    obj_type = type(obj)
    return {
        "type_name": obj_type.__name__,
        "module": getattr(obj_type, "__module__", None),
        "qualname": getattr(obj_type, "__qualname__", obj_type.__name__),
        "mro": [cls.__name__ for cls in obj_type.__mro__],
        "builtin": obj_type.__module__ == "builtins",
    }


def analyze_type(obj, depth=0, max_depth=3, context=None):
    """Main analysis function."""
    if depth > max_depth:
        return {"type_name": type(obj).__name__, "truncated": True}

    base_info = analyze_basic_type(obj, depth, max_depth, context, None)

    for predicate, analyzer in PREDICATE_ANALYZERS.items():
        if predicate(obj):
            specialized_info = analyzer(obj, depth, max_depth, context, analyze_type)
            return {**base_info, **specialized_info}

    return base_info


def parse_polars_html(html_str):
    """Parse a Polars DataFrame HTML output into structured rows/schema."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_str, "html.parser")

    small_tag = soup.find("small")
    shape = None
    if small_tag:
        m = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", small_tag.get_text())
        if m:
            shape = (int(m.group(1)), int(m.group(2)))

    thead = soup.find("thead")
    columns = []
    dtypes = []
    if thead:
        header_rows = thead.find_all("tr")
        if len(header_rows) >= 1:
            columns = [th.get_text(strip=True) for th in header_rows[0].find_all("th")]
        if len(header_rows) >= 2:
            dtypes = [td.get_text(strip=True) for td in header_rows[1].find_all("td")]

    tbody = soup.find("tbody")
    head_rows = []
    tail_rows = []
    truncated = False

    def _clean_cell(text):
        text = text.strip()
        if text == "null":
            return None
        for q in ('"', "'"):
            if len(text) >= 2 and text[0] == q and text[-1] == q:
                return text[1:-1]
            if text.startswith(q) and (text.endswith("\u2026") or text.endswith("...")):
                return text[1:]
        return text

    if tbody:
        found_ellipsis = False
        for tr in tbody.find_all("tr"):
            cells = [td.get_text() for td in tr.find_all("td")]
            if cells and all(c.strip() in ("...", "\u2026", "&hellip;") for c in cells):
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
    schema_match = (
        before["columns"] == after["columns"]
        and before["dtypes"] == after["dtypes"]
    )
    data_match = (
        before["head_rows"] == after["head_rows"]
        and before["tail_rows"] == after["tail_rows"]
    )

    result = {
        "match": shape_match and schema_match and data_match,
        "truncated": before["truncated"] or after["truncated"],
        "missing_rows_before": before["missing_rows"],
        "missing_rows_after": after["missing_rows"],
        "shape_match": shape_match,
        "schema_match": schema_match,
        "data_match": data_match,
    }

    if not result["match"]:
        details = {}
        if not shape_match:
            details["shape_before"] = before["shape"]
            details["shape_after"] = after["shape"]
        if not schema_match:
            details["columns_before"] = before["columns"]
            details["columns_after"] = after["columns"]
            details["dtypes_before"] = before["dtypes"]
            details["dtypes_after"] = after["dtypes"]
        if not data_match:
            details["head_rows_differ"] = before["head_rows"] != after["head_rows"]
            details["tail_rows_differ"] = before["tail_rows"] != after["tail_rows"]
        result["details"] = details

    return result


def _cells_match(before_val, after_val):
    """Compare visible cell values, tolerating display truncation."""
    if before_val == after_val:
        return "full"

    for a, b in [(before_val, after_val), (after_val, before_val)]:
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a.endswith("\u2026"):
            prefix = a[:-1]
        elif a.endswith("..."):
            prefix = a[:-3]
        else:
            continue
        if b.startswith(prefix):
            return "partial"

    return "mismatch"


def compare_polars_with_confidence(before_text, after_text, before_html=None, after_html=None):
    """Compare two Polars text reprs with the local confidence logic."""
    import numpy as np
    import polars as pl

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
        if not before_html or not after_html:
            return {
                "comparison_path": "polars_error",
                "coverage": None,
                "confidence": 0.0,
                "match": False,
            }

        before = parse_polars_html(before_html)
        after = parse_polars_html(after_html)

        b_visible = len(before["head_rows"]) + len(before["tail_rows"])
        a_visible = len(after["head_rows"]) + len(after["tail_rows"])
        b_declared = before["shape"][0] if before["shape"] else b_visible
        a_declared = after["shape"][0] if after["shape"] else a_visible
        coverage = (
            _coverage_label(b_visible, b_declared)
            + "_vs_"
            + _coverage_label(a_visible, a_declared)
        )

        shapes_match = (
            before["shape"] is not None
            and after["shape"] is not None
            and before["shape"] == after["shape"]
        )
        shape_score = 0.2 if shapes_match else 0.0
        num_cols = len(before["columns"])

        if not shapes_match or before["columns"] != after["columns"]:
            return _make_result("polars_html_fallback", coverage, shape_score)

        comparable_head = min(len(before["head_rows"]), len(after["head_rows"]))
        comparable_tail = min(len(before["tail_rows"]), len(after["tail_rows"]))
        total_cells = (comparable_head + comparable_tail) * 2 * num_cols

        matched_cells = 0.0
        for i in range(comparable_head):
            for col in before["columns"]:
                result = _cells_match(before["head_rows"][i].get(col), after["head_rows"][i].get(col))
                matched_cells += 1.0 if result == "full" else 0.5 if result == "partial" else 0.0

        for i in range(comparable_tail):
            for col in before["columns"]:
                result = _cells_match(before["tail_rows"][i].get(col), after["tail_rows"][i].get(col))
                matched_cells += 1.0 if result == "full" else 0.5 if result == "partial" else 0.0

        data_ratio = matched_cells / total_cells if total_cells else 1.0
        return _make_result("polars_html_fallback", coverage, shape_score + 0.8 * data_ratio)

    declared_b = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", before_text)
    declared_a = re.search(r"shape:\s*\((\d+),\s*(\d+)\)", after_text)
    b_shape = (
        (int(declared_b.group(1)), int(declared_b.group(2)))
        if declared_b
        else (b_df.height, b_df.width)
    )
    a_shape = (
        (int(declared_a.group(1)), int(declared_a.group(2)))
        if declared_a
        else (a_df.height, a_df.width)
    )

    coverage = (
        _coverage_label(b_df.height, b_shape[0])
        + "_vs_"
        + _coverage_label(a_df.height, a_shape[0])
    )
    shapes_match = b_shape == a_shape
    shape_score = 0.2 if shapes_match else 0.0
    num_cols = b_df.width
    total_cells = b_df.height * num_cols + a_df.height * num_cols

    if not shapes_match or b_df.columns != a_df.columns or total_cells == 0:
        data_ratio = 1.0 if total_cells == 0 and shapes_match and b_df.columns == a_df.columns else 0.0
        return _make_result("polars_from_repr", coverage, shape_score + 0.8 * data_ratio)

    sorted_cols = sorted(b_df.columns)
    b_sorted = b_df.select(sorted_cols).sort(sorted_cols)
    a_sorted = a_df.select(sorted_cols).sort(sorted_cols)

    try:
        b_arr = b_sorted.to_numpy(allow_copy=True)
        a_arr = a_sorted.to_numpy(allow_copy=True)
        try:
            eq_mask = b_arr == a_arr
            eq_mask = eq_mask | (np.isnan(b_arr) & np.isnan(a_arr))
        except (TypeError, ValueError):
            b_list = b_arr.tolist()
            a_list = a_arr.tolist()
            min_rows = min(len(b_list), len(a_list))
            matched = sum(
                1
                for r in range(min_rows)
                for c in range(num_cols)
                if b_list[r][c] == a_list[r][c]
            )
            return _make_result("polars_from_repr", coverage, shape_score + 0.8 * ((matched * 2.0) / total_cells))
    except Exception:
        try:
            all_match = b_sorted.frame_equal(a_sorted, null_equal=True)
        except AttributeError:
            all_match = b_sorted.equals(a_sorted, null_equal=True)
        return _make_result("polars_from_repr", coverage, shape_score + 0.8 * (1.0 if all_match else 0.0))

    min_rows = min(b_arr.shape[0], a_arr.shape[0])
    matched = int(eq_mask[:min_rows].sum())
    return _make_result("polars_from_repr", coverage, shape_score + 0.8 * ((matched * 2.0) / total_cells))


def detect_dataframe_kind(obj):
    """Detect DataFrame/array kind for the notebook comparison router."""
    type_name = type(obj).__name__
    module = getattr(type(obj), "__module__", "") or ""
    if type_name == "DataFrame" and "pandas" in module:
        return "pandas"
    if type_name == "DataFrame" and "polars" in module:
        return "polars"
    if type_name == "ndarray" and "numpy" in module:
        return "numpy"
    return None


def _is_dataframe_or_array(obj):
    return detect_dataframe_kind(obj) is not None


def _json_safe_value(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        if hasattr(val, "item"):
            val = val.item()
    except Exception:
        pass
    if isinstance(val, float) and math.isnan(val):
        return None
    if str(type(val)) in {"<class 'pandas._libs.missing.NAType'>", "<class 'pandas._libs.tslibs.nattype.NaTType'>"}:
        return None
    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)


def analyze_dataframe(obj, depth, max_depth, context, recurse_fn):
    """Analyzer for pandas/polars DataFrames and numpy ndarrays."""
    kind = detect_dataframe_kind(obj)

    if kind == "pandas":
        sorted_cols = sorted(obj.columns)
        sorted_df = obj[sorted_cols]
        try:
            sorted_df = sorted_df.sort_values(by=list(sorted_df.columns)).reset_index(drop=True)
        except Exception:
            sorted_df = sorted_df.reset_index(drop=True)
        arr = sorted_df.to_numpy()
        columns = [str(col) for col in sorted_df.columns]
        dtypes = [str(dtype) for dtype in sorted_df.dtypes]
        shape = list(obj.shape)
    elif kind == "polars":
        sorted_cols = sorted(obj.columns)
        sorted_df = obj.select(sorted_cols)
        try:
            sorted_df = sorted_df.sort(sorted_cols)
        except Exception:
            pass
        arr = sorted_df.to_numpy(allow_copy=True)
        columns = [str(col) for col in sorted_df.columns]
        dtypes = [str(dtype) for dtype in sorted_df.dtypes]
        shape = list(obj.shape)
    else:
        arr = obj
        columns = None
        dtypes = None
        shape = list(getattr(arr, "shape", []))

    arr_list = arr.tolist()
    ndim = getattr(arr, "ndim", 0)
    if ndim == 2:
        arr_list = [[_json_safe_value(v) for v in row] for row in arr_list]
    elif ndim == 1:
        arr_list = [_json_safe_value(v) for v in arr_list]
    else:
        arr_list = _json_safe_value(arr_list)

    result = {
        "dataframe_kind": kind,
        "array": arr_list,
        "shape": shape,
    }
    if columns is not None:
        result["columns"] = columns
    if dtypes is not None:
        result["dtypes"] = dtypes
    return result


def setup_type_formatter():
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
            obj_type = type(obj)
            obj_module = getattr(obj_type, "__module__", "") or ""
            if obj_type.__name__ == "DataFrame" and "polars" in obj_module and "text/html" in format_dict:
                format_dict["application/x-polars-html-parsed"] = json.dumps(parse_polars_html(format_dict["text/html"]))
        return format_dict, md_dict

    ip.display_formatter.format = format_with_type


def analyze_container(obj, depth, max_depth, context, recurse_fn):
    """Container analyzer that extends basic type info."""
    info = {"length": len(obj), "empty": len(obj) == 0}

    if len(obj) > 0 and depth < max_depth:
        sample = list(obj)[:5]
        info["element_types"] = [
            recurse_fn(item, depth + 1, max_depth, context) for item in sample
        ]
        info["element_type_names"] = list(
            set(elem_info["type_name"] for elem_info in info["element_types"])
        )

    return info


DEFAULT_ANALYZERS = {
    _is_dataframe_or_array: analyze_dataframe,
    list: analyze_container,
    tuple: analyze_container,
    set: analyze_container,
    frozenset: analyze_container,
    dict: analyze_container,
    lambda obj: hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)): analyze_container,
}

for c, analyzer_f in DEFAULT_ANALYZERS.items():
    register_analyzer(c, analyzer_f)
