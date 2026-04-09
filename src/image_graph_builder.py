from typing import Any, Dict


def build_graph_from_normalized_ir(normalized_ir: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_type": str(normalized_ir.get("image_type") or ""),
        "graph_type": str(normalized_ir.get("graph_type") or ""),
        "root_objects": normalized_ir.get("root_objects", []),
        "nodes": normalized_ir.get("nodes", []),
        "edges": normalized_ir.get("edges", []),
    }
