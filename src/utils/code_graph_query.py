#!/usr/bin/env python3
import argparse
import json
import sys
from collections import deque
from typing import Any, Dict, List, Tuple

EDGE_PRIORITY = {
    "calls": 0,
    "uses": 1,
    "imports": 2,
    "reverse_import": 3,
    "extends": 4,
    "semantic": 5,
    "component": 6,
    "config": 7,
    "style": 8,
    "contains": 9,
}

NODE_PRIORITY = {
    "file": 0,
    "class": 1,
    "function": 2,
    "method": 2,
}


def _normalize_rel_path(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _load_code_graph(graph_path: str) -> Dict[str, Any]:
    with open(graph_path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, dict):
        raise ValueError("graph payload is not a JSON object")
    code_graph = payload.get("code_graph", payload)
    if not isinstance(code_graph, dict):
        raise ValueError("missing code_graph object")
    return code_graph


def _build_indexes(code_graph: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    nodes = code_graph.get("nodes", [])
    edges = code_graph.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    node_by_id: Dict[str, Dict[str, Any]] = {}
    adjacency: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        node_by_id[node_id] = node
        adjacency.setdefault(node_id, [])

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        edge_type = str(edge.get("type") or "").strip()
        if not source or not target or not edge_type:
            continue
        if source in adjacency:
            adjacency[source].append({
                "direction": "out",
                "type": edge_type,
                "edge": edge,
                "other": target,
            })
        if target in adjacency:
            adjacency[target].append({
                "direction": "in",
                "type": edge_type,
                "edge": edge,
                "other": source,
            })
    return node_by_id, adjacency


def _node_summary(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": node.get("id"),
        "type": node.get("type"),
        "label": node.get("label") or node.get("name"),
        "file": _normalize_rel_path(str(node.get("file") or "")),
        "start_line": node.get("start_line"),
        "end_line": node.get("end_line"),
    }


def _resolve_start_nodes(
    node_by_id: Dict[str, Dict[str, Any]],
    *,
    file_path: str = "",
    symbol_label: str = "",
    node_id: str = "",
) -> List[Dict[str, Any]]:
    if node_id:
        node = node_by_id.get(node_id)
        return [node] if isinstance(node, dict) else []

    matches: List[Dict[str, Any]] = []
    if file_path:
        normalized_file = _normalize_rel_path(file_path)
        for node in node_by_id.values():
            if _normalize_rel_path(str(node.get("file") or "")) == normalized_file:
                matches.append(node)
        matches.sort(
            key=lambda node: (
                NODE_PRIORITY.get(str(node.get("type") or "").strip(), 99),
                str(node.get("label") or node.get("name") or ""),
            )
        )
        return matches

    if symbol_label:
        label = str(symbol_label).strip()
        for node in node_by_id.values():
            node_label = str(node.get("label") or node.get("name") or "").strip()
            if node_label == label:
                matches.append(node)
        return matches

    return []


def _dedupe_edges(edge_results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen = set()
    ranked: List[Dict[str, Any]] = []
    for edge in sorted(
        edge_results,
        key=lambda item: (
            item.get("depth", 0),
            EDGE_PRIORITY.get(str(item.get("type") or "").strip(), 99),
            str(item.get("from") or ""),
            str(item.get("to") or ""),
        ),
    ):
        key = (edge.get("from"), edge.get("to"), edge.get("type"), edge.get("depth"))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(edge)
        if len(ranked) >= limit:
            break
    return ranked


def _walk_neighbors(
    start_nodes: List[Dict[str, Any]],
    adjacency: Dict[str, List[Dict[str, Any]]],
    node_by_id: Dict[str, Dict[str, Any]],
    *,
    hops: int,
    direction: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if hops <= 0:
        return [], []

    queue = deque()
    visited = set()
    node_results: List[Dict[str, Any]] = []
    edge_results: List[Dict[str, Any]] = []

    for node in start_nodes:
        node_id = str(node.get("id") or "").strip()
        if node_id:
            visited.add(node_id)
            queue.append((node_id, 0))

    while queue and len(node_results) < limit:
        current_id, depth = queue.popleft()
        if depth >= hops:
            continue
        for rel in adjacency.get(current_id, []):
            rel_direction = rel.get("direction")
            if direction != "both" and rel_direction != direction:
                continue
            other_id = str(rel.get("other") or "").strip()
            other_node = node_by_id.get(other_id)
            if not isinstance(other_node, dict):
                continue
            edge_payload = {
                "from": current_id if rel_direction == "out" else other_id,
                "to": other_id if rel_direction == "out" else current_id,
                "type": rel.get("type"),
                "direction_from_start": rel_direction,
                "depth": depth + 1,
            }
            edge_results.append(edge_payload)
            if other_id in visited:
                continue
            visited.add(other_id)
            node_results.append(_node_summary(other_node))
            queue.append((other_id, depth + 1))
            if len(node_results) >= limit:
                break

    node_results.sort(
        key=lambda node: (
            node.get("depth", 0),
            EDGE_PRIORITY.get(str(node.get("via_edge_type") or "").strip(), 99),
            NODE_PRIORITY.get(str(node.get("type") or "").strip(), 99),
            str(node.get("label") or ""),
        )
    )
    return node_results[:limit], _dedupe_edges(edge_results, max(1, limit))


def _summarize_neighbors(
    start_nodes: List[Dict[str, Any]],
    neighbor_nodes: List[Dict[str, Any]],
    neighbor_edges: List[Dict[str, Any]],
    limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    start_ids = {str(node.get("id") or "").strip() for node in start_nodes}
    summarized_nodes: List[Dict[str, Any]] = []
    seen_nodes = set()
    for edge in neighbor_edges:
        other_id = str(edge.get("to") or "").strip()
        if other_id in start_ids:
            other_id = str(edge.get("from") or "").strip()
        if not other_id or other_id in start_ids or other_id in seen_nodes:
            continue
        node = next((item for item in neighbor_nodes if str(item.get("id") or "").strip() == other_id), None)
        if not isinstance(node, dict):
            continue
        merged = dict(node)
        merged["via_edge_type"] = edge.get("type")
        merged["depth"] = edge.get("depth")
        seen_nodes.add(other_id)
        summarized_nodes.append(merged)
        if len(summarized_nodes) >= limit:
            break
    summarized_nodes.sort(
        key=lambda node: (
            node.get("depth", 0),
            EDGE_PRIORITY.get(str(node.get("via_edge_type") or "").strip(), 99),
            NODE_PRIORITY.get(str(node.get("type") or "").strip(), 99),
            str(node.get("label") or ""),
        )
    )
    return summarized_nodes[:limit], neighbor_edges[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Query a saved code graph JSON for nodes and adjacent edges.")
    parser.add_argument("--graph", required=True, help="Path to code_graph_<instance>.json")
    parser.add_argument("--file", help="Relative file path to inspect")
    parser.add_argument("--symbol", help="Exact node label to inspect")
    parser.add_argument("--node-id", help="Exact node id to inspect")
    parser.add_argument("--hops", type=int, default=1, help="How many graph hops to expand")
    parser.add_argument(
        "--direction",
        choices=["in", "out", "both"],
        default="both",
        help="Which edge directions to follow from the starting nodes",
    )
    parser.add_argument("--limit", type=int, default=8, help="Maximum number of neighbor nodes to return")
    args = parser.parse_args()

    provided = [bool(args.file), bool(args.symbol), bool(args.node_id)]
    if sum(provided) != 1:
        parser.error("exactly one of --file, --symbol, or --node-id must be provided")

    try:
        code_graph = _load_code_graph(args.graph)
        node_by_id, adjacency = _build_indexes(code_graph)
        start_nodes = _resolve_start_nodes(
            node_by_id,
            file_path=args.file or "",
            symbol_label=args.symbol or "",
            node_id=args.node_id or "",
        )
        result = {
            "query": {
                "graph": args.graph,
                "file": args.file,
                "symbol": args.symbol,
                "node_id": args.node_id,
                "hops": args.hops,
                "direction": args.direction,
                "limit": args.limit,
            },
            "start_nodes": [],
            "neighbor_nodes": [],
            "neighbor_edges": [],
        }
        if args.file:
            start_nodes = start_nodes[: min(4, max(1, args.limit))]
        else:
            start_nodes = start_nodes[:1]
        result["start_nodes"] = [_node_summary(node) for node in start_nodes]
        if start_nodes:
            neighbor_nodes, neighbor_edges = _walk_neighbors(
                start_nodes,
                adjacency,
                node_by_id,
                hops=max(0, args.hops),
                direction=args.direction,
                limit=max(1, args.limit),
            )
            summarized_nodes, summarized_edges = _summarize_neighbors(
                start_nodes,
                neighbor_nodes,
                neighbor_edges,
                max(1, args.limit),
            )
            result["neighbor_nodes"] = summarized_nodes
            result["neighbor_edges"] = summarized_edges

        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        json.dump({"error": str(exc)}, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
