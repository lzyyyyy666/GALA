from typing import Any, Dict, List


def build_compat_fields(image_graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deprecated caption compatibility layer.

    Graph summaries/captions are no longer generated in the main pipeline.
    Keep this helper as a no-op placeholder for backward imports.
    """
    _ = image_graphs
    return {}
