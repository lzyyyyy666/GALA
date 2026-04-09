"""Schema helpers for rooted image-graph extraction."""

from typing import Any, Dict


IMAGE_TYPES = [
    "ui_page",
    "chart_plot",
    "code_screenshot",
    "document_layout",
    "generic_diagram",
]
_IMAGE_TYPE_SET = set(IMAGE_TYPES)


def canonical_image_type(image_type: Any) -> str:
    value = str(image_type or "").strip().lower()
    return value if value in _IMAGE_TYPE_SET else "generic_diagram"


def clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def coerce_confidence(value: Any) -> float:
    return clamp_confidence(value, default=0.5)


def coerce_weight(value: Any, default: float = 1.0) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, weight)


def _norm_str(value: Any) -> str:
    return str(value or "").strip()


def validate_node(item: Any, index: int = 0) -> Dict[str, Any]:
    if not isinstance(item, dict):
        item = {}
    role = _norm_str(item.get("role")).lower()
    if role not in {"root", "supporting"}:
        role = "supporting"
    return {
        "id": _norm_str(item.get("id")) or f"obj_{index}",
        "type": _norm_str(item.get("type")).lower() or "object",
        "text": _norm_str(item.get("text")),
        "role": role,
        "reason": _norm_str(item.get("reason")),
    }


def validate_root_object(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        item = {}
    return {
        "id": _norm_str(item.get("id")),
        "reason": _norm_str(item.get("reason")),
    }


def validate_edge(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        item = {}
    relation = _norm_str(item.get("relation"))
    if not relation:
        relation = _norm_str(item.get("type"))
    relation = relation.lower() if relation else "related_to"
    return {
        "source": _norm_str(item.get("source")),
        "target": _norm_str(item.get("target")),
        "relation": relation,
        "reason": _norm_str(item.get("reason")),
    }


def default_rooted_ir(image_type: Any) -> Dict[str, Any]:
    return {
        "image_type": canonical_image_type(image_type),
        "root_objects": [],
        "nodes": [],
        "edges": [],
    }


def default_simple_ir(image_type: Any) -> Dict[str, Any]:
    return default_rooted_ir(image_type)


# Graph typing helpers used by downstream graph consumers.
GRAPH_TYPE_BY_IMAGE_TYPE = {
    "ui_page": "ui_graph",
    "chart_plot": "chart_graph",
    "code_screenshot": "code_token_graph",
    "document_layout": "document_graph",
    "generic_diagram": "generic_scene_graph",
}

NODE_TYPES_BY_GRAPH_TYPE = {
    "ui_graph": {"region", "component", "text", "icon", "anomaly"},
    "chart_graph": {"chart", "axis", "legend", "series", "mark", "label", "anomaly"},
    "code_token_graph": {"code_block", "line", "token", "anomaly"},
    "document_graph": {"document", "block", "table", "row", "cell", "equation", "anomaly"},
    "generic_scene_graph": {"region", "object", "text", "connector", "anomaly"},
}

EDGE_TYPES_BY_GRAPH_TYPE = {
    "ui_graph": {"part_of", "left_of", "right_of", "above", "below", "overlap", "labeled_by", "has_anomaly"},
    "chart_graph": {"part_of", "corresponds_to", "adjacent_to", "same_style_group", "has_anomaly"},
    "code_token_graph": {"part_of", "next_line", "next_token", "same_line_as", "has_anomaly"},
    "document_graph": {"part_of", "next_block", "next_row", "has_anomaly"},
    "generic_scene_graph": {"part_of", "connected_to", "points_to", "near", "has_anomaly"},
}

DEFAULT_NODE_TYPE_BY_GRAPH_TYPE = {
    "ui_graph": "component",
    "chart_graph": "mark",
    "code_token_graph": "token",
    "document_graph": "block",
    "generic_scene_graph": "object",
}

DEFAULT_EDGE_TYPE_BY_GRAPH_TYPE = {
    "ui_graph": "part_of",
    "chart_graph": "part_of",
    "code_token_graph": "part_of",
    "document_graph": "part_of",
    "generic_scene_graph": "part_of",
}

def graph_type_from_image_type(image_type: Any) -> str:
    return GRAPH_TYPE_BY_IMAGE_TYPE.get(canonical_image_type(image_type), "generic_scene_graph")


def allowed_node_types(graph_type: str):
    return NODE_TYPES_BY_GRAPH_TYPE.get(graph_type, NODE_TYPES_BY_GRAPH_TYPE["generic_scene_graph"])


def allowed_edge_types(graph_type: str):
    return EDGE_TYPES_BY_GRAPH_TYPE.get(graph_type, EDGE_TYPES_BY_GRAPH_TYPE["generic_scene_graph"])
