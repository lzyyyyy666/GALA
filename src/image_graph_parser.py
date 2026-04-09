"""Parser for rooted visual graph extraction output."""

import ast
import json
import re
from typing import Any, Dict, List, Optional

from src.image_graph_schema import (
    IMAGE_TYPES,
    canonical_image_type,
    default_simple_ir,
    validate_edge,
    validate_node,
    validate_root_object,
)


_FENCED_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    match = _FENCED_BLOCK_PATTERN.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = -1
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : idx + 1]
    return None


def _cleanup_json_like_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _try_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _try_literal_eval(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_raw_to_dict(response_text: str) -> Dict[str, Any]:
    raw = _strip_code_fences(response_text or "")
    if not raw:
        return {}

    candidate = _extract_balanced_json_object(raw) or raw
    candidate = _cleanup_json_like_text(candidate)

    parsed = _try_json_loads(candidate)
    if parsed is not None:
        return parsed

    parsed = _try_json_loads(candidate.replace("'", '"'))
    if parsed is not None:
        return parsed

    parsed = _try_literal_eval(candidate)
    if parsed is not None:
        return parsed

    return {}


def parse_image_type(response_text: str) -> str:
    parsed = _parse_raw_to_dict(response_text)
    if "image_type" in parsed:
        return canonical_image_type(parsed.get("image_type"))

    lowered = (response_text or "").lower()
    for image_type in IMAGE_TYPES:
        if image_type in lowered:
            return image_type
    return "generic_diagram"


def _normalize_nodes(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_nodes = parsed.get("nodes")
    if isinstance(raw_nodes, list):
        return [validate_node(item, idx) for idx, item in enumerate(raw_nodes)]
    return []


def _normalize_root_objects(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_root_objects = parsed.get("root_objects")
    if not isinstance(raw_root_objects, list):
        return []
    return [validate_root_object(item) for item in raw_root_objects]


def _normalize_edges(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_edges = parsed.get("edges")
    if isinstance(raw_edges, list):
        return [validate_edge(item) for item in raw_edges]
    return []


def parse_visual_extraction_response(response_text: str, image_type: str) -> Dict[str, Any]:
    """Parse model output into rooted visual IR.

    This function never raises parsing exceptions.
    """
    output = default_simple_ir(image_type=image_type)

    try:
        parsed = _parse_raw_to_dict(response_text)
    except Exception:
        return output

    output["image_type"] = canonical_image_type(parsed.get("image_type", image_type))
    output["root_objects"] = _normalize_root_objects(parsed)
    output["nodes"] = _normalize_nodes(parsed)
    output["edges"] = _normalize_edges(parsed)
    return output

