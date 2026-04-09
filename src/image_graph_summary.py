"""Deprecated graph-summary utilities.

Kept for research comparison only. Main image-graph pipeline no longer calls
this module and now outputs structured graph JSON directly.
"""

from collections import Counter
from typing import Any, Dict, List


def _join_or_none(items: List[str]) -> str:
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    if not cleaned:
        return ""
    return ", ".join(cleaned[:3])


def generate_graph_summary(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
    issue_focus = graph.get("issue_focus", {}) if isinstance(graph.get("issue_focus"), dict) else {}

    node_counter = Counter()
    anomaly_texts: List[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "").strip().lower()
        if node_type:
            node_counter[node_type] += 1
        if node_type == "anomaly":
            desc = str(node.get("semantic_desc") or node.get("text") or "").strip()
            if desc:
                anomaly_texts.append(desc)

    top_types = [f"{k}({v})" for k, v in node_counter.most_common(3)]
    structure_part = "key structures: " + (", ".join(top_types) if top_types else "none")

    symptoms = issue_focus.get("symptoms", []) if isinstance(issue_focus.get("symptoms"), list) else []
    symptom_part = _join_or_none(symptoms)
    if symptom_part:
        symptom_part = f"; issue symptoms: {symptom_part}"
    else:
        symptom_part = ""

    if anomaly_texts:
        anomaly_part = f"; anomaly: {anomaly_texts[0]}"
    elif node_counter.get("anomaly", 0) > 0:
        anomaly_part = "; anomaly: detected but not explicitly described"
    else:
        anomaly_part = "; anomaly: none explicitly detected"

    return f"{structure_part}{symptom_part}{anomaly_part}"


def generate_instance_summaries(image_graphs: List[Dict[str, Any]]) -> List[str]:
    summaries: List[str] = []
    for graph in image_graphs:
        summary = str(graph.get("graph_summary") or "").strip()
        if summary:
            summaries.append(summary)
    return summaries
