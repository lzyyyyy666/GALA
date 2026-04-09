"""Normalizer for rooted node/edge visual IR."""

from collections import deque
from typing import Any, Dict, List, Set, Tuple

from src.image_graph_schema import default_simple_ir


def _relation_name(value: Any) -> str:
    relation = str(value or "").strip().lower()
    relation = relation.replace(" ", "_")
    return relation or "related_to"


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_visual_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize rooted node/edge visual IR.

    Rules:
    - Ensure unique node IDs (generate if missing)
    - Preserve root object references from model output
    - Normalize relation names to lowercase snake_case
    - Remove duplicate nodes by normalized type + normalized text
    - Drop edges referencing unknown nodes
    - Keep only nodes reachable from at least one root when roots exist
    """
    output = default_simple_ir(ir.get("image_type") if isinstance(ir, dict) else "generic_diagram")
    if not isinstance(ir, dict):
        return output

    raw_nodes = ir.get("nodes")
    nodes = raw_nodes if isinstance(raw_nodes, list) else []

    raw_root_objects = ir.get("root_objects")
    root_objects = raw_root_objects if isinstance(raw_root_objects, list) else []
    requested_root_reasons: Dict[str, str] = {}
    requested_root_ids: Set[str] = set()
    for root_object in root_objects:
        if not isinstance(root_object, dict):
            continue
        root_id = str(root_object.get("id") or "").strip()
        if not root_id:
            continue
        requested_root_ids.add(root_id)
        requested_root_reasons[root_id] = str(root_object.get("reason") or "").strip()

    id_seen: Set[str] = set()
    signature_to_index: Dict[Tuple[str, str], int] = {}
    normalized_nodes: List[Dict[str, Any]] = []

    auto_id = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue

        node_id = str(node.get("id") or "").strip()
        if not node_id:
            node_id = f"node_{auto_id}"
            auto_id += 1
        if node_id in id_seen:
            suffix = 1
            base = node_id
            while f"{base}_{suffix}" in id_seen:
                suffix += 1
            node_id = f"{base}_{suffix}"
        id_seen.add(node_id)

        node_type = str(node.get("type") or "").strip().lower() or "object"
        node_text = str(node.get("text") or "").strip()
        node_role = str(node.get("role") or "").strip().lower() or "supporting"
        is_root = node_role == "root" or node_id in requested_root_ids

        signature = (node_type, _norm_text(node_text))
        existing_index = signature_to_index.get(signature)
        if existing_index is not None:
            existing_node = normalized_nodes[existing_index]
            if is_root:
                existing_node["role"] = "root"
                if not existing_node.get("reason"):
                    existing_node["reason"] = (
                        requested_root_reasons.get(node_id)
                        or str(node.get("reason") or "").strip()
                    )
            elif not existing_node.get("reason"):
                existing_node["reason"] = str(node.get("reason") or "").strip()
            continue

        signature_to_index[signature] = len(normalized_nodes)
        normalized_nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "text": node_text,
                "role": "root" if is_root else ("supporting" if node_role not in {"root", "supporting"} else node_role),
                "reason": requested_root_reasons.get(node_id) or str(node.get("reason") or "").strip(),
            }
        )

    valid_ids = {node["id"] for node in normalized_nodes}

    normalized_root_objects: List[Dict[str, Any]] = []
    for node in normalized_nodes:
        if node.get("role") != "root":
            continue
        node_id = node["id"]
        if node_id not in valid_ids:
            continue
        normalized_root_objects.append(
            {
                "id": node_id,
                "reason": requested_root_reasons.get(node_id) or str(node.get("reason") or "").strip(),
            }
        )

    root_id_set = {root_object["id"] for root_object in normalized_root_objects}
    for node in normalized_nodes:
        if node["id"] in root_id_set:
            node["role"] = "root"
        elif node.get("role") not in {"root", "supporting"}:
            node["role"] = "supporting"

    raw_edges = ir.get("edges")
    edges = raw_edges if isinstance(raw_edges, list) else []
    normalized_edges: List[Dict[str, Any]] = []
    relation_keys: Set[Tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if source not in valid_ids or target not in valid_ids:
            continue
        relation_name = _relation_name(edge.get("relation") or edge.get("type"))
        relation_key = (source, target, relation_name)
        if relation_key in relation_keys:
            continue
        relation_keys.add(relation_key)
        normalized_edges.append(
            {
                "source": source,
                "target": target,
                "relation": relation_name,
                "reason": str(edge.get("reason") or "").strip(),
            }
        )

    if normalized_root_objects:
        adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in valid_ids}
        for edge in normalized_edges:
            source = edge["source"]
            target = edge["target"]
            adjacency[source].add(target)
            adjacency[target].add(source)

        reachable: Set[str] = set()
        queue = deque(root["id"] for root in normalized_root_objects)
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            reachable.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in reachable:
                    queue.append(neighbor)

        normalized_nodes = [node for node in normalized_nodes if node["id"] in reachable]
        valid_ids = {node["id"] for node in normalized_nodes}
        normalized_root_objects = [
            root_object for root_object in normalized_root_objects if root_object["id"] in valid_ids
        ]
        normalized_edges = [
            edge
            for edge in normalized_edges
            if edge["source"] in valid_ids and edge["target"] in valid_ids
        ]

    output["root_objects"] = normalized_root_objects
    output["nodes"] = normalized_nodes
    output["edges"] = normalized_edges
    return output
