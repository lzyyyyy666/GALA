"""Prompt definitions for root-centered visual graph extraction."""

from prompt.image_graph_prompt import IMAGE_TYPES


_ROOTED_EXTRACTION_GUIDANCE = {
    "ui_page": (
        "Focus on the UI components that are most relevant to the issue, then expand outward through "
        "the smallest connected set of related controls, containers, labels, and stateful elements."
    ),
    "chart_plot": (
        "Focus on the chart elements most relevant to the issue, then expand through the smallest "
        "connected set of related axes, labels, series, marks, and legend items."
    ),
    "code_screenshot": (
        "Focus on the code or UI snippet most relevant to the issue, then expand through the smallest "
        "connected set of related lines, tokens, highlights, and nearby structural context."
    ),
    "document_layout": (
        "Focus on the document blocks most relevant to the issue, then expand through the smallest "
        "connected set of related headings, tables, rows, cells, and layout context."
    ),
    "generic_diagram": (
        "Focus on the diagram objects most relevant to the issue, then expand through the smallest "
        "connected set of linked objects, connectors, and labels."
    ),
}

_ROOTED_ATTRIBUTE_GUIDANCE = {
    "ui_page": (
        "Recommended keys: state, role, color, visibility, interaction_state, size_hint, alignment_hint."
    ),
    "chart_plot": (
        "Recommended keys: series_role, color, marker_style, axis_role, value_hint, legend_role."
    ),
    "code_screenshot": (
        "Recommended keys: token_role, syntax_kind, highlight_state, line_hint, emphasis."
    ),
    "document_layout": (
        "Recommended keys: block_role, heading_level, alignment, font_weight, list_role, table_role."
    ),
    "generic_diagram": (
        "Recommended keys: shape, connector_style, arrow_direction, group_role, line_style."
    ),
}


ROOTED_EXTRACTION_SYSTEM_PROMPT = (
    "You extract a rooted visual graph for issue analysis. "
    "First select root objects, then other relevant nodes, then edges among the selected nodes. "
    "Return JSON only."
)


def build_rooted_extraction_prompt(
    image_type: str,
    issue_text: str,
) -> str:
    normalized_type = image_type if image_type in IMAGE_TYPES else "generic_diagram"
    guidance = _ROOTED_EXTRACTION_GUIDANCE[normalized_type]
    return (
        "Task: build a compact rooted visual graph for the issue.\n\n"
        "Procedure:\n"
        "1. Select root_objects that are the most directly relevant visible entities for the issue.\n"
        "   Root objects should be the elements most directly affected by the bug, most directly responsible for the bug, "
        "or explicitly mentioned by the issue as the key problematic elements or relations.\n"
        "   Do not choose generic containers or background context as root objects unless they are themselves part of the issue.\n"
        "   For each root object, the reason should explain why it is a root issue-relevant object, "
        "using both the issue text and visible image evidence.\n"
        "2. Build nodes around the issue text and the selected root_objects.\n"
        "   Include the root objects themselves, then add other relevant visible nodes that help explain the root objects, "
        "the bug mechanism, the affected context, or an issue-relevant relation.\n"
        "   Do not add generic surrounding context, containers, or nearby elements unless they are necessary to explain the issue.\n"
        "   For each node, the reason should explain why this node is needed to understand the issue and what visible cue supports it.\n"
        "3. Add edges only among selected nodes, and only when the relation is visually grounded and useful for explaining the issue.\n\n"
        "   For each edge, the reason should explain why this relation is visually supported and why it matters for the issue.\n"
        "   Avoid vague reasons such as \"relevant to the issue\" without concrete support.\n\n"
        f"Predicted image_type: {normalized_type}\n\n"
        "Type-aware guidance:\n"
        f"{guidance}\n\n"
        "For each node.type, use a short, concrete, free-form visible object label.\n"
        "node.type must name the visible object itself, not a diagnosis, interpretation, or bug summary.\n"
        "Do not force node.type into a fixed ontology.\n\n"
        "Issue text:\n"
        f"{issue_text or ''}\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "image_type": "<same as predicted image_type>",\n'
        '  "root_objects": [\n'
        '    {"id":"...", "reason":"..."}\n'
        "  ],\n"
        '  "nodes": [\n'
        '    {"id":"...", "type":"...", "text":"...", "role":"root|supporting", "reason":"..."}\n'
        "  ],\n"
        '  "edges": [\n'
        '    {"source":"node_id","target":"node_id","relation":"...", "reason":"..."}\n'
        "  ]\n"
        "}\n\n"
        "Requirements:\n"
        "- root_objects must be grounded in both the issue text and the image\n"
        "- every root_objects.id must appear exactly in nodes.id\n"
        "- every node with role=\"root\" must appear in root_objects\n"
        "- every node must include a reason for why it was selected\n"
        "- every node.type should be a short, concrete, free-form visible object label for the object itself, not a diagnosis or conclusion\n"
        "- keep the graph small; do not add nodes or edges that do not improve issue understanding\n"
        "- edges may only connect nodes already present in nodes\n"
        "- every edge must include a reason\n"
        "- if an edge is uncertain, omit it\n"
        "- return JSON only"
    )
