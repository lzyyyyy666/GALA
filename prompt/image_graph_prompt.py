"""Prompt definitions for the Type-Aware Image Graph pipeline.

This module only defines prompt-layer logic for:
1) Stage 1 image type classification
2) Stage 2 type-aware rooted graph extraction

Stage 2 emits `root_objects`/`nodes`/`edges` directly.
"""

from typing import Dict


IMAGE_TYPES = [
    "ui_page",
    "chart_plot",
    "code_screenshot",
    "document_layout",
    "generic_diagram",
]


_EXTRACTION_GUIDANCE = {
    "ui_page": (
        "Focus on UI layout regions, components, labels, icons, and visible visual defects "
        "(e.g., overlap, clipping, missing component, wrong spacing/alignment)."
    ),
    "chart_plot": (
        "Focus on chart area, axes, legend, series, marks, labels, and chart-specific defects "
        "(e.g., overlapping labels, wrong scale impression, missing legend item)."
    ),
    "code_screenshot": (
        "Focus on code block regions, lines, tokens, highlights, and rendering defects "
        "(e.g., broken wrapping, missing token, unreadable contrast)."
    ),
    "document_layout": (
        "Focus on document blocks, headings, tables, rows/cells, equations, and layout defects "
        "(e.g., broken table borders, shifted cell alignment, missing section)."
    ),
    "generic_diagram": (
        "Focus on diagram regions, objects, connectors, text labels, and inconsistencies "
        "(e.g., missing connector, crossed arrows, ambiguous object linkage)."
    ),
}

_ATTRIBUTE_GUIDANCE = {
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


CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a strict multimodal classifier for software issue images. "
    "Inspect BOTH the image and issue text, then classify the image into exactly one category."
)


EXTRACTION_SYSTEM_PROMPT = (
    "You are a structured visual extraction engine. "
    "Return structured rooted visual graph JSON only."
)


def classify_image_type_prompt(issue_text: str) -> str:
    """Build Stage 1 classifier prompt.

    Expected model output (strict JSON only):
    {
      "image_type": "...",
      "confidence": 0.0,
      "reason": "..."
    }
    """
    category_text = "\n".join(f"- {name}" for name in IMAGE_TYPES)
    return (
        "Task: classify the image into exactly one image_type for a type-aware visual pipeline.\n\n"
        "You MUST inspect both:\n"
        "1) issue text\n"
        "2) image content\n\n"
        f"Allowed image_type values:\n{category_text}\n\n"
        f"Issue text:\n{issue_text or ''}\n\n"
        "Output requirements:\n"
        "- Return JSON only (no markdown, no explanation outside JSON)\n"
        "- Choose exactly one image_type from the allowed list\n"
        "- confidence must be a float in [0, 1]\n"
        '- JSON schema: {"image_type":"...", "confidence":0.0, "reason":"..."}'
    )


def build_visual_extraction_prompt(image_type: str, issue_text: str) -> str:
    """Build Stage 2 type-aware rooted graph extraction prompt.

    The model must output rooted visual graph extraction JSON with:
    - image_type
    - root_objects
    - nodes
    - edges
    """
    normalized_type = image_type if image_type in IMAGE_TYPES else "generic_diagram"
    guidance = _EXTRACTION_GUIDANCE[normalized_type]
    attribute_guidance = _ATTRIBUTE_GUIDANCE[normalized_type]
    return (
        "Task: perform structured visual extraction for downstream issue analysis.\n"
        "Return only a compact rooted visual graph.\n\n"
        f"Predicted image_type: {normalized_type}\n"
        f"Type-aware guidance: {guidance}\n"
        f"Type-aware attributes guidance: {attribute_guidance}\n"
        "Use open-ended keys when necessary, but keep keys concise and snake_case.\n"
        "Only include nodes and edges that are directly visible/readable from the image.\n"
        "Do not guess invisible properties or uncertain relations.\n\n"
        f"Issue text:\n{issue_text or ''}\n\n"
        "Output JSON schema (strict, JSON only):\n"
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
        "Field definitions:\n"
        "- root_objects: the most directly issue-relevant visible entities\n"
        "- nodes: visual entities (UI components, chart marks, code blocks, labels, cells, etc.)\n"
        '- edges: [{"source":"node_id","target":"node_id","relation":"...","reason":"..."}]\n\n'
        "Constraints:\n"
        "- JSON only\n"
        "- no markdown code fences\n"
        "- every root_objects.id must appear in nodes.id\n"
        "- every node with role=\"root\" must appear in root_objects\n"
        "- every node must include a reason\n"
        "- every edge must include a reason\n"
        "- edges may only reference nodes already present in nodes\n"
        "- keep the graph compact and issue-focused"
    )


IMAGE_TYPE_CLASSIFICATION_PROMPT: Dict[str, str] = {
    "system_prompt": CLASSIFICATION_SYSTEM_PROMPT,
    "user_prompt_template": (
        "Task: classify the image into exactly one image_type for a type-aware visual pipeline.\n\n"
        "You MUST inspect both:\n"
        "1) issue text\n"
        "2) image content\n\n"
        "Allowed image_type values:\n"
        "- ui_page\n"
        "- chart_plot\n"
        "- code_screenshot\n"
        "- document_layout\n"
        "- generic_diagram\n\n"
        "Issue text:\n"
        "{issue_text}\n\n"
        "Output requirements:\n"
        "- Return JSON only (no markdown, no explanation outside JSON)\n"
        "- Choose exactly one image_type from the allowed list\n"
        "- confidence must be a float in [0, 1]\n"
        '- JSON schema: {{"image_type":"...", "confidence":0.0, "reason":"..."}}'
    ),
}
def build_extraction_prompt(image_type: str, issue_text: str) -> str:
    """Alias for rooted graph extraction prompt construction."""
    return build_visual_extraction_prompt(image_type=image_type, issue_text=issue_text)
