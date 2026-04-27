import inspect
import json

PREDICATE_ANALYZERS = {}

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


DEFAULT_ANALYZERS = {
    list: analyze_container,
    tuple: analyze_container,
    set: analyze_container,
    frozenset: analyze_container,
    dict: analyze_container,
    lambda obj: hasattr(obj, "__iter__")
    and not isinstance(obj, (str, bytes)): analyze_container,
}

for c, analyzer_f in DEFAULT_ANALYZERS.items():
    register_analyzer(c, analyzer_f)
