import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.llm_client import get_usage_totals, send_chat_completion


GRAPH_SOURCE_EXTENSIONS = (".js", ".jsx", ".ts", ".tsx")
DEFAULT_TOP_N_CANDIDATE_FILES = 10
DEFAULT_TOP_N_SEED_FILES = 5
CANDIDATE_FILE_EXTENSIONS = (
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".scss",
    ".css",
    ".json",
    ".html",
    ".md",
    ".mdx",
    ".svg",
    ".frag",
    ".vert",
    ".lock",
)
SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".next",
    ".nuxt",
    ".cache",
}
MAX_TRAVERSAL_FILES = 20
GENERIC_UI_COMPONENT_DIR_NAMES = {
    "button",
    "card",
    "icon",
    "input",
    "textarea",
    "select",
    "checkbox",
    "radio",
    "label",
    "link",
    "badge",
    "tag",
    "tooltip",
    "popover",
    "spinner",
    "loader",
    "modal",
    "dialog",
}


def _read_int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, str(default))
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


BUG_REPORT_TEMPLATE = """
The bug report is as follows:
```
### GitHub Problem Description ###
{problem_statement}

###

### Candidate Files ###
{structure}

###
```
""".strip()

FILE_SYSTEM_PROMPT_WITHOUT_TOOL = """
You will be presented with a bug report, a focused visual graph summary, and repository structure for the source code of the system under test (SUT).
Your task is to identify source files that may need to be edited to fix the reported issue.
""".strip()

FILE_GUIDENCE_PRMPT_WITHOUT_TOOL = """
Let's locate the faulty file step by step using reasoning.
In order to locate accurately, you can first identify a set of likely relevant files, then refine them based on the repository structure and bug report.
""".strip()

FILE_SUMMARY = """
Based on the available information, reconfirm and provide the complete names of the likely relevant files for the bug.
Return exactly 10 files when at least 10 relevant files are available. If fewer than 10 files are available, return all available relevant files.
Since your answer will be processed automatically, please give your answer in the format as follows.
The returned files should be separated by new lines and wrapped with ```.
```
src/file1.js
styles/file2.scss
package.json
docs/file4.mdx
src/file5.ts
src/file6.tsx
styles/file7.css
src/file8.js
config/file9.json
docs/file10.mdx
```
Replace the example paths with the actual file paths.
For example,
```
src/plugins/plugin.legend.js
package.json
```
""".strip()

FILE_VISUAL_SEED_PROMPT = """
Please look through the following GitHub problem description, focused visual graph summary, and Repository structure and provide a list of files that may need to be edited to fix the problem.

### GitHub Problem Description ###
{issue}

###

### Focused Visual Graph Summary ###
{visual_summary}

###

### Repository Structure ###
{repo_tree}

###

Please only provide the full path.
Please only include source files that may need to be edited, especially JavaScript/TypeScript, JSX/TSX, style, state, selector, action, and config files when relevant.
Do not rank or sort the files by importance.
Return exactly 10 files when at least 10 relevant files are available. If fewer than 10 files are available, return all available relevant files.
The returned files should be separated by new lines and wrapped with ```
For example:
```
path/to/file1.js
path/to/file2.jsx
path/to/file3.ts
path/to/file4.tsx
path/to/file5.css
```
""".strip()

CANDIDATE_SEED_RERANK_PROMPT = """
Issue:
{problem_statement}

Focused Visual Graph Summary:
{image_summary}

Candidate Files:
{candidate_file_summary}

Candidate File Graph:
{candidate_file_graph_summary}

Task:
Based on the inputs above, choose the 5 files that are most likely to need modification to fix the issue.

Requirements:
- Only use files from the Candidate Files list.
- Prefer files that are more directly connected to the issue semantics and the focused visual structure.
- Use the Candidate File Graph only as supporting context.
- Do not invent new files.
- Return exactly {top_n_seed_files} files when at least {top_n_seed_files} suitable files are available. Otherwise return all suitable files.

Return strict JSON with this schema:
{{
  "seed_files": ["path/to/file1.js", "path/to/file2.js"]
}}
""".strip()

FORMAT_CORRECT_PROMPT = """
Here is a localization result, but it seems not in the correct format. Please correct it.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
This is an example of expected output:
```
src/plugins/plugin.legend.js
package.json
```

Please help me corrct the following result.
{res}
""".strip()

ALIGNMENT_GRAPH_MATCH_SYSTEM_PROMPT = """
You are a graph-grounded alignment agent.
You directly align issue-relevant image subgraphs to the provided code graph.
Return only valid JSON.
""".strip()

ALIGNMENT_GRAPH_MATCH_PROMPT = """
Issue:
{problem_statement}

Image Subgraphs:
{image_summary}

Seed Files:
{seed_files}

Code Graph:
{code_graph_json}

Task:
For each issue-relevant image subgraph, directly align it to the provided code graph.

Requirements:
- You are responsible for the semantic alignment decisions. Choose which image nodes best correspond to which code nodes.
- Downstream validation only checks whether the nodes and edges you cite actually exist in the provided graphs.
- Use the provided code graph nodes and edges directly. Do not invent code subgraphs.
- Do NOT score files, nodes, or edges.
- A valid alignment must include:
  1. at least two `node_correspondence` entries that map image nodes to real code graph nodes
  2. at least one `supporting_image_edges` entry copied from the image subgraph between aligned image nodes
  3. at least one `supporting_code_edges` entry copied exactly from the code graph between the corresponding aligned code nodes
- `code_anchor.node_id` must match a real code graph node id.
- Every returned `supporting_code_edges` item must exactly match a real code graph edge.
- Prefer the smallest supported alignment that fully explains the bug-relevant visual structure.
- Only keep an alignment when the image-side relation and code-side relation are both present between the aligned node pairs.
- `matched_files` should be the union of files touched by the code nodes that participate in the validated aligned relations.

Return strict JSON with this schema:
{{
  "subgraph_alignments": [
    {{
      "image_subgraph_id": "img_sg_1",
      "alignment_type": "semantic_graph_anchor",
      "node_correspondence": [
        {{
          "image_node_id": "object_id",
          "code_anchor": {{
            "node_id": "function:src/foo.ts:renderChart",
            "file": "src/foo.ts",
            "node": "renderChart"
          }}
        }}
      ],
      "supporting_image_edges": [
        {{"source": "a", "target": "b", "type": "contains"}}
      ],
      "supporting_code_edges": [
        {{"source": "function:src/foo.ts:renderChart", "target": "function:src/layout.ts:computeLayout", "type": "calls"}}
      ],
      "shared_structure": ["..."],
      "why_it_matches": "...",
      "matched_files": ["src/foo.ts"]
    }}
  ],
  "matched_files": ["src/foo.ts"]
}}
""".strip()

RERANK_SYSTEM_PROMPT = """
You select the most useful seed files from a candidate file list for downstream code localization.
Return only valid JSON.
""".strip()

RERANK_SELECTED_FILES_PROMPT = """
Issue:
{problem_statement}

Focused Visual Graph Summary:
{image_summary}

Selected Files:
{selected_file_summary}

Task:
Rerank the selected files so the most issue-relevant files appear first.

Requirements:
- Only use files from the Selected Files list.
- Prefer files that are more directly connected to the issue semantics and the focused visual structure.
- Keep the output concise and ordered from most relevant to least relevant.
- Do not invent new files.
- Do not use scores.

Return strict JSON with this schema:
{{
  "reranked_top15_files": ["path/to/file1.js", "path/to/file2.js"]
}}
""".strip()

FUNCTION_ALIGNMENT_STAGE_A_SYSTEM_PROMPT = """
You align issue-relevant image graph anchors and relations to a UI-focused function graph.
Use only the provided graph evidence.
Return only valid JSON.
""".strip()

FUNCTION_ALIGNMENT_STAGE_A_PROMPT = """
Issue:
{problem_statement}

Image Graph Summary:
{image_graph_summary}

UI Function Graph Summary:
{ui_function_graph_summary}

Task:
1. Read the image graph summary directly and identify the most important root or issue-relevant image nodes.
2. Match those image nodes to concrete code function nodes using the provided code node `file` and `name` fields.
3. Judge whether the important image relations are reflected by local code graph relations.

Requirements:
- Only use node ids and edge tuples that exist in the provided code graph.
- Do not invent functions.
- Treat each code node as identified by `file + name` and use `code_node_id` exactly as provided in Code Nodes.
- Base your decision on the image node text/type/reason and the code node file/name, together with the provided code edges.
- Prefer the smallest supported set of anchor matches.
- If evidence is weak, keep the match but lower confidence.
- Return valid JSON only. Do not add commentary.

Return strict JSON:
{{
  "anchor_matches": [
    {{
      "image_node_id": "anchor_id",
      "code_node_id": "function:path/file.tsx:Component.renderThing",
      "confidence": "high | medium | low",
      "reason": "graph-grounded reason"
    }}
  ],
  "relation_alignment": [
    {{
      "image_edge": {{"source": "a", "target": "b", "type": "contains"}},
      "code_edge": {{"source": "function:...", "target": "function:...", "type": "renders"}},
      "aligned": true,
      "reason": "why the relation is or is not aligned"
    }}
  ]
}}
""".strip()

FUNCTION_ALIGNMENT_STAGE_B_SYSTEM_PROMPT = """
You identify which functions should be edited to explain the UI anomaly.
Use only the provided anomaly, graph evidence, anchor matches, and relation alignment.
Return only valid JSON.
""".strip()

FUNCTION_ALIGNMENT_STAGE_B_PROMPT = """
Issue:
{problem_statement}

Image Graph Summary:
{image_graph_summary}

Anchor Matches:
{anchor_matches_json}

Relation Alignment:
{relation_alignment_json}

Code Subgraph Summary:
{code_subgraph_summary}

Task:
1. Find the responsibility nodes that explain the UI anomaly.
2. Distinguish:
   - primary: likely requires direct modification
   - secondary: close implementation support, may need modification
   - context: relevant for reasoning, usually not modified

Constraints:
- primary <= 2
- total edit targets <= 5
- every target must exist in the provided code subgraph
- Each target must be selected from the provided code subgraph using the exact `file` and `name` values shown there.
- Use image evidence, anchor matches, relation alignment, and subgraph structure together.
- Return valid JSON only. Do not add commentary.

Return strict JSON:
{{
  "edit_targets": [
    {{
      "function": "functionName",
      "file": "path/to/file.tsx",
      "role": "primary | secondary | context",
      "confidence": "high | medium | low",
      "reason": "graph-grounded reason"
    }}
  ]
}}
""".strip()


def _normalize_file_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _is_graph_source_file(file_name: str) -> bool:
    return file_name.endswith(GRAPH_SOURCE_EXTENSIONS)


def _is_candidate_file(file_name: str) -> bool:
    if _is_graph_source_file(file_name):
        return True
    if file_name.endswith(CANDIDATE_FILE_EXTENSIONS):
        return True
    return "." not in os.path.basename(file_name)


def _is_test_like_path(path: str) -> bool:
    normalized = _normalize_file_path(path).lower()
    parts = [part for part in normalized.split("/") if part]
    return any(
        part in {"test", "tests", "__tests__", "__mocks__", "spec", "specs", "fixtures"}
        for part in parts
    )


def _find_block_end(lines: List[str], start_idx: int) -> int:
    depth = 0
    opened = False
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        opens = line.count("{")
        closes = line.count("}")
        if opens > 0:
            opened = True
        depth += opens
        depth -= closes
        if opened and depth <= 0:
            return idx + 1
    return len(lines)


def _read_text_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as infile:
        return infile.read().splitlines()


def read_file_safe(file_path: str) -> str:
    if not os.path.isfile(file_path):
        return ""
    try:
        return "\n".join(_read_text_file(file_path))
    except Exception:
        return ""


def extract_file_tokens(file_path: str) -> Set[str]:
    name = os.path.basename(file_path)
    stem = os.path.splitext(name)[0]
    tokens = re.split(r"[^a-zA-Z0-9]", stem.lower())
    return {token for token in tokens if len(token) > 2}


def extract_config_keys(content: str) -> Set[str]:
    text = str(content or "")
    pattern = r"\b([a-zA-Z0-9_]+)\s*:"
    return set(re.findall(pattern, text))


def extract_dom_classes(content: str) -> Set[str]:
    text = str(content or "")
    classes: Set[str] = set()

    class_name_values = re.findall(r"className\s*=\s*['\"]([^'\"]+)['\"]", text)
    for value in class_name_values:
        for token in re.split(r"\s+", value.strip()):
            normalized = token.strip()
            if normalized:
                classes.add(normalized)

    selector_values = re.findall(r"querySelector\(\s*['\"]\.([^'\"]+)['\"]\s*\)", text)
    for value in selector_values:
        normalized = value.strip()
        if normalized:
            classes.add(normalized)
    return classes


def extract_css_classes(content: str) -> Set[str]:
    text = str(content or "")
    return set(re.findall(r"\.([a-zA-Z0-9_-]+)", text))


def find_semantic_related_files(
    file_path: str,
    candidate_files: List[str],
    file_tokens_map: Dict[str, Set[str]],
) -> Set[str]:
    neighbors: Set[str] = set()
    tokens = file_tokens_map.get(file_path, set())
    for other in candidate_files:
        if other == file_path:
            continue
        if tokens & file_tokens_map.get(other, set()):
            neighbors.add(other)
    return neighbors


def generate_aliases(name: str) -> List[str]:
    raw = str(name or "").strip()
    if not raw:
        return []

    # Split camelCase/PascalCase boundaries first.
    expanded = re.sub(r"([a-z0-9$])([A-Z])", r"\1 \2", raw)
    # Normalize common separators and paths.
    expanded = re.sub(r"[\\/_\-\.\s]+", " ", expanded)
    # Keep JS/TS-friendly identifier chars.
    expanded = re.sub(r"[^A-Za-z0-9$\s]+", " ", expanded)
    expanded = expanded.lower()

    aliases: List[str] = []
    seen: Set[str] = set()
    for token in expanded.split():
        if not token or token in seen:
            continue
        seen.add(token)
        aliases.append(token)
    return aliases


def parse_jsts_lines(lines: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    class_info: List[Dict[str, Any]] = []
    function_names: List[Dict[str, Any]] = []
    seen_functions: Set[Tuple[str, int]] = set()

    class_pattern = re.compile(
        r"^\s*(?:export\s+default\s+|export\s+|abstract\s+)?class\s+([A-Za-z_$][\w$]*)(?:\s+extends\s+([A-Za-z_$][\w$]*))?(?:\s+implements\s+([A-Za-z_$][\w$]*(?:\s*,\s*[A-Za-z_$][\w$]*)*))?"
    )
    function_pattern = re.compile(
        r"^\s*(?:export\s+default\s+|export\s+|async\s+)*function\s+([A-Za-z_$][\w$]*)\s*\("
    )
    variable_function_pattern = re.compile(
        r"^\s*(?:export\s+default\s+|export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
    )
    class_method_pattern = re.compile(
        r"^\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*\{"
    )

    for idx, line in enumerate(lines):
        class_match = class_pattern.match(line)
        if class_match:
            end_line = _find_block_end(lines, idx)
            parent_class_name = class_match.group(2) or ""
            implements_clause = class_match.group(3) or ""
            implements_names = [item.strip() for item in implements_clause.split(",") if item.strip()]
            methods: List[Dict[str, Any]] = []
            for method_idx in range(idx + 1, min(end_line, len(lines))):
                method_line = lines[method_idx]
                stripped = method_line.strip()
                if not stripped or stripped.startswith(("if ", "for ", "while ", "switch ", "catch ", "return ")):
                    continue
                method_match = class_method_pattern.match(method_line)
                if not method_match:
                    continue
                method_name = method_match.group(1)
                if method_name in {"if", "for", "while", "switch", "catch"}:
                    continue
                method_end = _find_block_end(lines, method_idx)
                methods.append(
                    {
                        "name": method_name,
                        "start_line": method_idx + 1,
                        "end_line": method_end,
                        "text": lines[method_idx:method_end],
                    }
                )
            class_info.append(
                {
                    "name": class_match.group(1),
                    "extends": parent_class_name,
                    "implements": implements_names,
                    "start_line": idx + 1,
                    "end_line": end_line,
                    "text": lines[idx:end_line],
                    "methods": methods,
                }
            )
            continue

        function_match = function_pattern.match(line)
        if function_match:
            function_name = function_match.group(1)
            end_line = _find_block_end(lines, idx)
            key = (function_name, idx + 1)
            if key not in seen_functions:
                seen_functions.add(key)
                function_names.append(
                    {
                        "name": function_name,
                        "start_line": idx + 1,
                        "end_line": end_line,
                        "text": lines[idx:end_line],
                    }
                )
            continue

        variable_match = variable_function_pattern.match(line)
        if variable_match:
            function_name = variable_match.group(1)
            end_line = _find_block_end(lines, idx)
            key = (function_name, idx + 1)
            if key not in seen_functions:
                seen_functions.add(key)
                function_names.append(
                    {
                        "name": function_name,
                        "start_line": idx + 1,
                        "end_line": end_line,
                        "text": lines[idx:end_line],
                    }
                )

    return class_info, function_names


CALL_EXCLUDE_NAMES = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "typeof",
    "instanceof",
    "super",
    "require",
    "import",
}


def _extract_call_targets(text: str) -> Set[str]:
    targets: Set[str] = set()
    pattern = re.compile(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s*\(")
    for match in pattern.finditer(str(text or "")):
        raw = match.group(1).strip()
        if not raw:
            continue
        head = raw.split(".", 1)[0]
        if head in CALL_EXCLUDE_NAMES:
            continue
        targets.add(raw)
    return targets


def _extract_new_targets(text: str) -> Set[str]:
    return set(re.findall(r"\bnew\s+([A-Za-z_$][\w$]*)\s*\(", str(text or "")))


def _extract_jsx_component_targets(text: str) -> Set[str]:
    targets: Set[str] = set()
    for match in re.finditer(r"<([A-Z][A-Za-z0-9_$]*)\b", str(text or "")):
        targets.add(match.group(1))
    return targets


def _extract_exported_names(lines: List[str]) -> Set[str]:
    exported: Set[str] = set()
    patterns = [
        re.compile(r"^\s*export\s+(?:default\s+)?class\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*export\s+(?:default\s+)?function\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)"),
    ]
    for line in lines:
        stripped = line.strip()
        for pattern in patterns:
            match = pattern.match(stripped)
            if match:
                exported.add(match.group(1))
        if stripped.startswith("export {"):
            block_match = re.search(r"export\s*\{([^}]+)\}", stripped)
            if not block_match:
                continue
            for item in block_match.group(1).split(","):
                token = item.strip()
                if not token:
                    continue
                if " as " in token:
                    token = token.split(" as ", 1)[0].strip()
                if re.match(r"^[A-Za-z_$][\w$]*$", token):
                    exported.add(token)
    return exported


def _extract_type_import_modules(lines: List[str]) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("import"):
            continue
        modules = CodeGraphBuilder._extract_import_targets(stripped)
        if not modules:
            continue
        if re.search(r"^\s*import\s+type\b", stripped) or re.search(r"\{\s*type\s+[A-Za-z_$]", stripped):
            for module_name in modules:
                results.append((module_name, stripped))
    return results


def _extract_type_uses(text: str) -> Set[str]:
    targets: Set[str] = set()
    patterns = [
        r":\s*([A-Z][A-Za-z0-9_$]*)\b",
        r"\bas\s+([A-Z][A-Za-z0-9_$]*)\b",
        r"<([A-Z][A-Za-z0-9_$]*)>",
        r"\bimplements\s+([A-Z][A-Za-z0-9_$]*)\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, str(text or "")):
            targets.add(match)
    return targets


def _extract_hook_calls(text: str) -> Set[str]:
    return set(re.findall(r"\b(use[A-Z][A-Za-z0-9_$]*)\s*\(", str(text or "")))


def _extract_state_reads(text: str) -> Set[str]:
    targets: Set[str] = set()
    patterns = [
        r"\b(useSelector|getState|mapStateToProps|select[A-Z][A-Za-z0-9_$]*)\s*\(",
        r"\b([A-Za-z_$][\w$]*)\.getState\s*\(",
        r"\b([A-Za-z_$][\w$]*)\.state\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, str(text or "")):
            if isinstance(match, tuple):
                for item in match:
                    if item:
                        targets.add(item)
            elif match:
                targets.add(match)
    return targets


def _extract_state_writes(text: str) -> Set[str]:
    targets: Set[str] = set()
    patterns = [
        r"\b(this\.setState|setState|set[A-Z][A-Za-z0-9_$]*)\s*\(",
        r"\b([A-Za-z_$][\w$]*)\.setState\s*\(",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, str(text or "")):
            if isinstance(match, tuple):
                for item in match:
                    if item:
                        targets.add(item)
            elif match:
                targets.add(match)
    return targets


def _extract_subscriptions(text: str) -> Set[str]:
    targets: Set[str] = set()
    patterns = [
        r"\b([A-Za-z_$][\w$]*)\.(?:subscribe|addEventListener|on|listen)\s*\(",
        r"\b(useSyncExternalStore)\s*\(",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, str(text or "")):
            if isinstance(match, tuple):
                for item in match:
                    if item:
                        targets.add(item)
            elif match:
                targets.add(match)
    return targets


def _extract_dispatches(text: str) -> Set[str]:
    targets: Set[str] = set()
    patterns = [
        r"\b(dispatch|emit|publish)\s*\(",
        r"\b([A-Za-z_$][\w$]*)\.(?:dispatch|emit|publish)\s*\(",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, str(text or "")):
            if isinstance(match, tuple):
                for item in match:
                    if item:
                        targets.add(item)
            elif match:
                targets.add(match)
    return targets


def _extract_context_providers(text: str) -> Set[str]:
    return set(re.findall(r"<([A-Z][A-Za-z0-9_$]*)\.Provider\b", str(text or "")))


def _extract_context_consumers(text: str) -> Set[str]:
    targets = set(re.findall(r"\buseContext\s*\(\s*([A-Z][A-Za-z0-9_$]*)\s*\)", str(text or "")))
    targets.update(re.findall(r"<([A-Z][A-Za-z0-9_$]*)\.Consumer\b", str(text or "")))
    return targets


def _extract_route_targets(text: str) -> Set[str]:
    targets: Set[str] = set()
    patterns = [
        r"\bcomponent\s*=\s*\{([A-Z][A-Za-z0-9_$]*)\}",
        r"\belement\s*=\s*\{<([A-Z][A-Za-z0-9_$]*)\b",
        r"\brender\s*=\s*\{[^}]*\b([A-Z][A-Za-z0-9_$]*)\b",
        r"\broute(?:s)?\s*:\s*\[",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, str(text or "")):
            if isinstance(match, tuple):
                for item in match:
                    if item:
                        targets.add(item)
            elif match and not match.startswith("route"):
                targets.add(match)
    return targets


def _extract_style_targets(
    source_file: str,
    lines: List[str],
    available_files: Set[str],
) -> Set[str]:
    targets: Set[str] = set()
    for line in lines:
        stripped = line.strip()
        if "import" not in stripped and "require(" not in stripped:
            continue
        for module_name in CodeGraphBuilder._extract_import_targets(stripped):
            target_file = CodeGraphBuilder._resolve_import_target(source_file, module_name, available_files)
            if target_file and os.path.splitext(target_file)[1] in {".css", ".scss"}:
                targets.add(target_file)
    return targets


def build_available_file_list(directory_path: str) -> List[str]:
    available_files: List[str] = []
    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [dir_name for dir_name in dirs if dir_name not in SKIP_DIR_NAMES]
        for file_name in files:
            if not _is_candidate_file(file_name):
                continue
            file_path = os.path.join(root, file_name)
            if not os.path.isfile(file_path):
                continue
            relative_path = _normalize_file_path(os.path.relpath(file_path, directory_path))
            if _is_test_like_path(relative_path):
                continue
            available_files.append(relative_path)
    return sorted(set(available_files))


def build_prompt_tree(available_files: List[str], repo_name: str) -> Dict[str, Any]:
    tree: Dict[str, Any] = {repo_name: {}}
    for file_path in available_files:
        curr = tree[repo_name]
        parts = [part for part in _normalize_file_path(file_path).split("/") if part]
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = None
    return tree


def show_project_structure(structure: Dict[str, Any], spacing: int = 0) -> str:
    output = ""
    for key, value in structure.items():
        is_file = value is None
        if is_file:
            output += " " * spacing + str(key) + "\n"
        else:
            output += " " * spacing + str(key) + "/" + "\n"
            if isinstance(value, dict):
                output += show_project_structure(value, spacing + 4)
    return output


def _coerce_image_graph(image_graph: Any) -> Dict[str, Any]:
    """Normalize single/multi-image graph payloads to a unified dict view.

    - If input is a dict, return a sanitized dict with root_objects/nodes/edges.
    - If input is a list of dicts, merge all graphs into one graph.
      Node ids are namespaced when more than one graph is present to avoid collisions.
    """
    if isinstance(image_graph, dict):
        raw_root_objects = image_graph.get("root_objects")
        raw_nodes = image_graph.get("nodes")
        raw_edges = image_graph.get("edges")

        root_objects = raw_root_objects if isinstance(raw_root_objects, list) else []
        nodes = raw_nodes if isinstance(raw_nodes, list) else []
        edges = raw_edges if isinstance(raw_edges, list) else []

        sanitized_nodes: List[Dict[str, Any]] = []
        known_node_ids: Set[str] = set()
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip() or f"node_{idx + 1}"
            known_node_ids.add(node_id)
            sanitized_nodes.append(dict(node, id=node_id))

        sanitized_root_objects: List[Dict[str, Any]] = []
        for root in root_objects:
            if not isinstance(root, dict):
                continue
            root_id = str(root.get("id") or "").strip()
            if not root_id:
                continue
            sanitized_root_objects.append({"id": root_id, "reason": str(root.get("reason") or "").strip()})

        sanitized_edges: List[Dict[str, Any]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            if source and target and source in known_node_ids and target in known_node_ids:
                sanitized_edges.append(dict(edge, source=source, target=target))

        return {
            "image_type": str(image_graph.get("image_type") or "").strip(),
            "root_objects": sanitized_root_objects,
            "nodes": sanitized_nodes,
            "edges": sanitized_edges,
        }

    if isinstance(image_graph, list):
        valid_graphs = [item for item in image_graph if isinstance(item, dict) and item]
        if not valid_graphs:
            return {}

        merged_root_objects: List[Dict[str, Any]] = []
        merged_nodes: List[Dict[str, Any]] = []
        merged_edges: List[Dict[str, Any]] = []
        multiple_graphs = len(valid_graphs) > 1

        for graph_idx, graph in enumerate(valid_graphs, start=1):
            normalized = _coerce_image_graph(graph)
            if not normalized:
                continue

            prefix = f"img_{graph_idx}::" if multiple_graphs else ""
            id_map: Dict[str, str] = {}

            for node_idx, node in enumerate(normalized.get("nodes", []), start=1):
                if not isinstance(node, dict):
                    continue
                old_id = str(node.get("id") or "").strip() or f"node_{node_idx}"
                new_id = f"{prefix}{old_id}"
                id_map[old_id] = new_id
                merged_nodes.append(dict(node, id=new_id, image_index=graph_idx))

            for root in normalized.get("root_objects", []):
                if not isinstance(root, dict):
                    continue
                old_root_id = str(root.get("id") or "").strip()
                new_root_id = id_map.get(old_root_id)
                if not new_root_id:
                    continue
                merged_root_objects.append(
                    {
                        "id": new_root_id,
                        "reason": str(root.get("reason") or "").strip(),
                        "image_index": graph_idx,
                    }
                )

            for edge in normalized.get("edges", []):
                if not isinstance(edge, dict):
                    continue
                source = id_map.get(str(edge.get("source") or "").strip())
                target = id_map.get(str(edge.get("target") or "").strip())
                if not source or not target:
                    continue
                merged_edges.append(dict(edge, source=source, target=target, image_index=graph_idx))

        if not merged_nodes:
            return {}
        return {
            "image_type": "multi_image" if multiple_graphs else str(valid_graphs[0].get("image_type") or "").strip(),
            "root_objects": merged_root_objects,
            "nodes": merged_nodes,
            "edges": merged_edges,
        }

    return {}


def build_visual_summary(image_graph: Any) -> str:
    normalized_graph = _coerce_image_graph(image_graph)
    if not normalized_graph:
        return ""

    nodes = normalized_graph.get("nodes")
    edges = normalized_graph.get("edges")
    root_objects = normalized_graph.get("root_objects")
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    if not isinstance(root_objects, list):
        root_objects = []

    root_reason_by_id: Dict[str, str] = {}
    root_ids: Set[str] = set()
    for item in root_objects:
        if not isinstance(item, dict):
            continue
        root_id = str(item.get("id") or "").strip()
        if not root_id:
            continue
        root_ids.add(root_id)
        root_reason_by_id[root_id] = str(item.get("reason") or "").strip()

    def _normalize_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _node_label(node: Dict[str, Any]) -> str:
        node_type = _normalize_text(node.get("type"))
        text_value = _normalize_text(node.get("text"))
        node_id = _normalize_text(node.get("id"))
        if node_type and text_value:
            return f"{node_type} [{text_value}]"
        if node_type:
            return node_type
        if text_value:
            return text_value
        return node_id

    node_by_id: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id:
            node_by_id[node_id] = node

    lines: List[str] = ["Image Graph:"]

    lines.append("Root objects:")
    root_lines = 0
    for root_object in root_objects:
        if root_lines >= 5:
            break
        if not isinstance(root_object, dict):
            continue
        root_id = str(root_object.get("id") or "").strip()
        if not root_id:
            continue
        node = node_by_id.get(root_id, {"id": root_id})
        label = _node_label(node)
        lines.append(f"- {root_id}: {label}")
        reason = root_reason_by_id.get(root_id) or _image_node_reason(node)
        if reason:
            lines.append(f"  reason: {reason}")
        root_lines += 1
    if root_lines == 0:
        lines.append("- none")

    lines.append("Nodes:")
    node_lines = 0
    for node in nodes[:12]:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        role = str(node.get("role") or "").strip().lower() or "supporting"
        if node_id in root_ids:
            role = "root"
        label = _node_label(node)
        lines.append(f"- {node_id}: {label} ({role})")
        reason = _image_node_reason(node)
        if reason:
            lines.append(f"  reason: {reason}")
        node_lines += 1
    if node_lines == 0:
        lines.append("- none")

    lines.append("Edges:")
    edge_lines = 0
    for edge in edges[:12]:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("source") or "").strip()
        target_id = str(edge.get("target") or "").strip()
        edge_type = _normalize_text(edge.get("type") or edge.get("relation"))
        if not source_id or not target_id or not edge_type:
            continue
        lines.append(f"- {source_id} --{edge_type}--> {target_id}")
        reason = _normalize_text(edge.get("reason"))
        if reason:
            lines.append(f"  reason: {reason}")
        edge_lines += 1
    if edge_lines == 0:
        lines.append("- none")

    return "\n".join(lines)


def build_candidate_file_graph(
    candidate_files: List[str],
    forward_neighbors: Dict[str, List[str]],
) -> Dict[str, Any]:
    normalized_files = _dedupe_keep_order([
        _normalize_file_path(file_path) for file_path in candidate_files if str(file_path).strip()
    ])
    candidate_set = set(normalized_files)
    nodes = [
        {
            "id": f"file:{file_path}",
            "type": "file",
            "file": file_path,
            "name": os.path.basename(file_path),
        }
        for file_path in normalized_files
    ]
    edges: List[Dict[str, str]] = []
    seen_edges: Set[Tuple[str, str, str]] = set()
    for source_file in normalized_files:
        for target_file in forward_neighbors.get(source_file, []):
            normalized_target = _normalize_file_path(target_file)
            if normalized_target not in candidate_set:
                continue
            edge = (source_file, normalized_target, "imports")
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            edges.append(
                {
                    "source": f"file:{source_file}",
                    "target": f"file:{normalized_target}",
                    "type": "imports",
                }
            )
    return {
        "files": normalized_files,
        "nodes": nodes,
        "edges": edges,
    }


def render_candidate_file_graph_summary(candidate_file_graph: Dict[str, Any]) -> str:
    if not isinstance(candidate_file_graph, dict):
        return "none"
    files = candidate_file_graph.get("files", [])
    edges = candidate_file_graph.get("edges", [])
    if not isinstance(files, list):
        files = []
    if not isinstance(edges, list):
        edges = []
    lines: List[str] = []
    if files:
        lines.append("Files:")
        for file_path in files[:20]:
            lines.append(f"- {file_path}")
    if edges:
        lines.append("Edges:")
        for edge in edges[:30]:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").replace("file:", "", 1)
            target = str(edge.get("target") or "").replace("file:", "", 1)
            edge_type = str(edge.get("type") or edge.get("relation") or "").strip()
            if source and target and edge_type:
                lines.append(f"- {source} --{edge_type}--> {target}")
    return "\n".join(lines).strip() or "none"


def _safe_json_loads(content: str) -> Any:
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    if text.count("```") >= 2:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        for block in blocks:
            parsed = _safe_json_loads(block)
            if parsed is not None:
                return parsed
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
    else:
        candidate = text
    try:
        import json

        return json.loads(candidate)
    except Exception:
        return None


def _node_text_tokens(node: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for key in ("raw_type", "type", "text", "reason", "root_reason"):
        value = str(node.get(key) or "").strip()
        if value and value not in tokens:
            tokens.append(value)
    return tokens


def _image_node_reason(node: Dict[str, Any]) -> str:
    return _normalize_text(node.get("reason") or node.get("root_reason"))


def _format_function_display_name(file_path: str, function_name: str) -> str:
    base_name = os.path.basename(_normalize_file_path(file_path))
    return f"{base_name}/{function_name}" if base_name else function_name


def build_image_subgraphs(image_graph: Any) -> List[Dict[str, Any]]:
    normalized_graph = _coerce_image_graph(image_graph)
    if not normalized_graph:
        return []

    raw_nodes = normalized_graph.get("nodes")
    raw_edges = normalized_graph.get("edges")
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []

    node_map: Dict[str, Dict[str, Any]] = {}
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id:
            node_map[node_id] = node

    root_ids: List[str] = []
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        if (node.get("is_root") or str(node.get("role") or "").strip().lower() == "root") and node_id not in root_ids:
            root_ids.append(node_id)

    if not root_ids:
        root_ids = list(node_map.keys())[:3]

    edge_records: List[Dict[str, str]] = []
    adjacency: Dict[str, List[Dict[str, str]]] = {}
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        edge_type = str(edge.get("type") or edge.get("relation") or "").strip()
        if not source or not target or not edge_type:
            continue
        record = {"source": source, "target": target, "type": edge_type}
        edge_records.append(record)
        adjacency.setdefault(source, []).append(record)
        adjacency.setdefault(target, []).append(record)

    subgraphs: List[Dict[str, Any]] = []
    seen_signatures: Set[Tuple[str, ...]] = set()
    for idx, center_id in enumerate(root_ids, start=1):
        if center_id not in node_map:
            continue
        local_edges = adjacency.get(center_id, [])[:3]
        node_ids: List[str] = [center_id]
        supporting_edges: List[Dict[str, str]] = []
        for edge in local_edges:
            supporting_edges.append(edge)
            for node_id in (edge["source"], edge["target"]):
                if node_id in node_map and node_id not in node_ids:
                    node_ids.append(node_id)
            if len(node_ids) >= 4:
                break
        signature = tuple(sorted(node_ids))
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        nodes = [node_map[node_id] for node_id in node_ids if node_id in node_map]
        summary_parts: List[str] = []
        for node in nodes:
            label_parts = _node_text_tokens(node)
            label = " | ".join(label_parts[:3]) if label_parts else str(node.get("id") or "")
            summary_parts.append(label)
        subgraphs.append(
            {
                "subgraph_id": f"img_sg_{idx}",
                "center_object_id": center_id,
                "node_ids": node_ids,
                "nodes": nodes,
                "edges": supporting_edges,
                "summary": " ; ".join(summary_parts[:4]),
            }
        )
    return subgraphs


def render_image_subgraphs_summary(image_subgraphs: List[Dict[str, Any]]) -> str:
    if not image_subgraphs:
        return "none"
    lines: List[str] = []
    for subgraph in image_subgraphs[:8]:
        lines.append(f"subgraph_id: {subgraph.get('subgraph_id', '')}")
        lines.append(f"  center: {subgraph.get('center_object_id', '')}")
        node_lines: List[str] = []
        for node in subgraph.get("nodes", [])[:4]:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            label = " | ".join(_node_text_tokens(node)[:3]) or node_id
            node_lines.append(f"{node_id} [{label}]")
        if node_lines:
            lines.append(f"  nodes: {', '.join(node_lines)}")
        for edge in subgraph.get("edges", [])[:3]:
            if not isinstance(edge, dict):
                continue
            lines.append(
                f"  edge: {edge.get('source', '')} --{edge.get('type', '')}--> {edge.get('target', '')}"
            )
    return "\n".join(lines).strip() or "none"


def render_ui_function_graph_summary(ui_function_graph: Dict[str, Any]) -> str:
    if not isinstance(ui_function_graph, dict):
        return "none"
    nodes = ui_function_graph.get("nodes", [])
    edges = ui_function_graph.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    lines: List[str] = []
    lines.append(f"node_count: {len(nodes)}")
    lines.append(f"edge_count: {len(edges)}")

    if nodes:
        lines.append("Nodes:")
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            function_name = str(node.get("name") or "").strip()
            lines.append(f"- {node_id} => {function_name}")

    if edges:
        lines.append("Edges:")
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            edge_type = str(edge.get("type") or edge.get("relation") or "").strip()
            if source and target and edge_type:
                lines.append(f"- {source} --{edge_type}--> {target}")

    return "\n".join(lines).strip() or "none"


def _dedupe_keep_order(items: List[str]) -> List[str]:
    deduped: List[str] = []
    for item in items:
        normalized = _normalize_file_path(str(item or "").strip())
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _dedupe_dicts_by_key(items: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: Set[Any] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _truncate_text(value: Any, limit: int = 220) -> str:
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_attribute_roles(node: Dict[str, Any]) -> List[str]:
    attributes = node.get("attributes")
    if not isinstance(attributes, dict):
        return []
    roles: List[str] = []
    for key in ("role", "state", "position", "layout"):
        value = _normalize_text(attributes.get(key))
        if value and value not in roles:
            roles.append(value)
    return roles


def _collect_signal_tokens(*values: Any) -> Set[str]:
    tokens: Set[str] = set()
    for value in values:
        text = _normalize_text(value).lower()
        if not text:
            continue
        for token in re.split(r"[^a-z0-9_]+", text):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


def _extract_props_targets(text: str) -> Set[str]:
    targets: Set[str] = set()
    for match in re.finditer(r"<([A-Z][A-Za-z0-9_$]*)\b([^>]*)>", str(text or ""), re.DOTALL):
        component = match.group(1)
        attrs = match.group(2) or ""
        if re.search(r"\b[A-Za-z_$][\w$]*\s*=\s*\{", attrs):
            targets.add(component)
    return targets


def _extract_inline_style_markers(text: str) -> Set[str]:
    markers: Set[str] = set()
    content = str(text or "")
    patterns = [
        r"\b(className|style|styled|css|sx)\b",
        r"\b(position|margin|padding|width|height|top|left|right|bottom|flex|grid)\b",
    ]
    for pattern in patterns:
        markers.update(re.findall(pattern, content))
    return markers


def _extract_event_handlers(text: str) -> Set[str]:
    handlers: Set[str] = set()
    content = str(text or "")
    handlers.update(re.findall(r"\b(on[A-Z][A-Za-z0-9_$]*)\s*=", content))
    handlers.update(re.findall(r"\b(handle[A-Z][A-Za-z0-9_$]*)\s*\(", content))
    handlers.update(re.findall(r"\b([A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>", content))
    return handlers


def _is_probably_utility_function(name: str, body: str, file_path: str) -> bool:
    lowered_name = str(name or "").lower()
    lowered_file = _normalize_file_path(file_path).lower()
    utility_keywords = {
        "format",
        "parse",
        "string",
        "math",
        "date",
        "time",
        "logger",
        "debug",
        "assert",
        "serialize",
        "deserialize",
        "tokenize",
        "memoize",
    }
    if any(keyword in lowered_name for keyword in utility_keywords):
        return True
    if any(part in lowered_file for part in ("/utils/", "/helpers/", "/lib/", "/logger", "/debug")):
        return True
    body_text = str(body or "").lower()
    if (
        "console." in body_text
        and "<" not in body_text
        and "usestate" not in body_text
        and "onclick" not in body_text
        and "classname" not in body_text
    ):
        return True
    return False


def _function_summary_from_body(name: str, file_path: str, body_text: str, signals: Dict[str, bool]) -> str:
    tags: List[str] = []
    if signals.get("is_component"):
        tags.append("component")
    if signals.get("uses_jsx"):
        tags.append("jsx")
    if signals.get("is_hook"):
        tags.append("hook")
    if signals.get("handles_event"):
        tags.append("event")
    if signals.get("reads_state") or signals.get("writes_state"):
        tags.append("state")
    if signals.get("style_related"):
        tags.append("style/layout")
    snippet = _truncate_text(body_text, 140)
    tag_text = ", ".join(tags) if tags else "ui-related"
    return f"{name} in {file_path} [{tag_text}] {snippet}".strip()


def _infer_responsibility_hints(text: str, relation_types: List[str]) -> List[str]:
    lowered = f"{text} {' '.join(relation_types)}".lower()
    hints: List[str] = []
    hint_keywords = {
        "layout": ("layout", "contains", "is_above", "is_below", "grid", "flex", "container"),
        "style": ("style", "color", "theme", "css", "visible", "hidden"),
        "state": ("state", "notice", "message", "visible", "disabled", "enabled", "status"),
        "event": ("click", "button", "submit", "action", "on_", "event"),
        "position": ("position", "above", "below", "left", "right", "top", "bottom"),
    }
    for hint, keywords in hint_keywords.items():
        if any(keyword in lowered for keyword in keywords):
            hints.append(hint)
    return hints or ["state"]


def _sort_edit_targets(edit_targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    role_order = {"primary": 0, "secondary": 1, "context": 2}
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        edit_targets,
        key=lambda item: (
            role_order.get(str(item.get("role") or "context"), 9),
            confidence_order.get(str(item.get("confidence") or "low"), 9),
            str(item.get("file") or ""),
            str(item.get("function") or ""),
        ),
    )


def _limit_edit_targets(edit_targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    primary = 0
    limited: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for item in _sort_edit_targets(edit_targets):
        function_name = str(item.get("function") or "").strip()
        file_path = _normalize_file_path(str(item.get("file") or "").strip())
        role = str(item.get("role") or "context").strip().lower()
        if not function_name or not file_path:
            continue
        key = (file_path, function_name)
        if key in seen:
            continue
        seen.add(key)
        if role == "primary":
            if primary >= 2:
                item = dict(item)
                item["role"] = "secondary"
                role = "secondary"
            primary += 1
        if len(limited) >= 5:
            break
        limited.append(
            {
                "function": function_name,
                "file": file_path,
                "role": role if role in {"primary", "secondary", "context"} else "context",
                "confidence": str(item.get("confidence") or "low").strip().lower() or "low",
                "reason": _truncate_text(item.get("reason") or "", 220),
            }
        )
    return limited


def _score_image_node_for_alignment(node: Dict[str, Any], issue_tokens: Set[str], root_ids: Set[str]) -> int:
    if not isinstance(node, dict):
        return 0
    node_id = str(node.get("id") or "").strip()
    role = _normalize_text(node.get("role"))
    text = _normalize_text(node.get("text"))
    node_type = _normalize_text(node.get("raw_type") or node.get("type"))
    reason = _image_node_reason(node)
    attribute_roles = " ".join(_extract_attribute_roles(node))
    node_tokens = _collect_signal_tokens(text, node_type, reason, attribute_roles)
    score = len(node_tokens & issue_tokens)
    if node_id in root_ids or role == "root" or bool(node.get("is_root")):
        score += 4
    if text:
        score += 1
    if reason:
        score += 1
    if attribute_roles:
        score += 1
    if any(token in node_tokens for token in {"button", "notice", "banner", "modal", "dialog", "form", "input", "label"}):
        score += 1
    return score


def _is_generic_ui_anchor_file(file_path: str) -> bool:
    normalized = _normalize_file_path(file_path)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 3:
        return False
    if parts[0] != "client" or parts[1] != "components":
        return False
    component_dir = parts[2].lower()
    if component_dir not in GENERIC_UI_COMPONENT_DIR_NAMES:
        return False
    basename = os.path.basename(normalized).lower()
    if basename.startswith("index."):
        return True
    return len(parts) == 3


class FunctionAlignmentAgent:
    def __init__(
        self,
        query_text_model,
        top_n_seed_files: int,
    ) -> None:
        self._query_text_model = query_text_model
        self.top_n_seed_files = top_n_seed_files

    @staticmethod
    def _safe_json_or_empty(raw_output: str) -> Dict[str, Any]:
        parsed = _safe_json_loads(raw_output)
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _json_dumps(payload: Any) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _confidence_from_signal(score: int) -> str:
        if score >= 5:
            return "high"
        if score >= 3:
            return "medium"
        return "low"

    @staticmethod
    def _iter_image_nodes(image_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_nodes = image_graph.get("nodes", [])
        if not isinstance(raw_nodes, list):
            return []
        return [node for node in raw_nodes if isinstance(node, dict)]

    @staticmethod
    def _iter_image_edges(image_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_edges = image_graph.get("edges", [])
        if not isinstance(raw_edges, list):
            return []
        edges: List[Dict[str, Any]] = []
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            edge_type = str(edge.get("type") or edge.get("relation") or "").strip()
            if source and target and edge_type:
                edges.append({"source": source, "target": target, "type": edge_type})
        return edges

    @staticmethod
    def _image_anchor_ids(image_graph: Dict[str, Any]) -> Set[str]:
        root_objects = image_graph.get("root_objects", [])
        anchor_ids = {
            str(item.get("id") or "").strip()
            for item in root_objects
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        if anchor_ids:
            return anchor_ids
        return {
            str(node.get("id") or "").strip()
            for node in FunctionAlignmentAgent._iter_image_nodes(image_graph)
            if str(node.get("id") or "").strip()
            and (
                str(node.get("role") or "").strip().lower() == "root"
                or bool(node.get("is_root"))
            )
        }

    @staticmethod
    def _image_reason_map(image_graph: Dict[str, Any]) -> Dict[str, str]:
        reason_map: Dict[str, str] = {}
        root_objects = image_graph.get("root_objects", [])
        if isinstance(root_objects, list):
            for item in root_objects:
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("id") or "").strip()
                reason = _normalize_text(item.get("reason"))
                if node_id and reason:
                    reason_map[node_id] = reason
        for node in FunctionAlignmentAgent._iter_image_nodes(image_graph):
            node_id = str(node.get("id") or "").strip()
            reason = _image_node_reason(node)
            if node_id and reason and node_id not in reason_map:
                reason_map[node_id] = reason
        return reason_map

    @staticmethod
    def _simple_image_hints(image_graph: Dict[str, Any], problem_statement: str) -> List[str]:
        text_parts = [_normalize_text(problem_statement)]
        for node in FunctionAlignmentAgent._iter_image_nodes(image_graph):
            text_parts.append(_normalize_text(node.get("text")))
            text_parts.append(_image_node_reason(node))
            attributes = node.get("attributes")
            if isinstance(attributes, dict):
                for value in attributes.values():
                    text_parts.append(_normalize_text(value))
        relation_types = [str(edge.get("type") or "").strip() for edge in FunctionAlignmentAgent._iter_image_edges(image_graph)]
        text_parts.extend(relation_types)
        return _infer_responsibility_hints(" ".join(text_parts), relation_types)

    @staticmethod
    def _validate_anchor_matches(
        raw_matches: Any,
        image_graph: Dict[str, Any],
        ui_function_graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        anchors = FunctionAlignmentAgent._image_anchor_ids(image_graph)
        node_by_id = {
            str(node.get("id") or "").strip(): node
            for node in ui_function_graph.get("nodes", [])
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        }
        validated: List[Dict[str, Any]] = []
        seen_images: Set[str] = set()
        for item in raw_matches if isinstance(raw_matches, list) else []:
            if not isinstance(item, dict):
                continue
            image_node_id = str(item.get("image_node_id") or "").strip()
            code_node_id = str(item.get("code_node_id") or "").strip()
            if not image_node_id or image_node_id not in anchors or not code_node_id or code_node_id not in node_by_id:
                continue
            if image_node_id in seen_images:
                continue
            seen_images.add(image_node_id)
            code_node = node_by_id[code_node_id]
            validated.append(
                {
                    "image_node_id": image_node_id,
                    "code_node_id": code_node_id,
                    "file": _normalize_file_path(str(code_node.get("file") or "").strip()),
                    "function": str(code_node.get("name") or "").strip(),
                    "confidence": str(item.get("confidence") or "low").strip().lower() or "low",
                    "reason": _truncate_text(item.get("reason") or "", 220),
                }
            )
        return validated

    @staticmethod
    def _validate_relation_alignment(
        raw_relations: Any,
        image_graph: Dict[str, Any],
        ui_function_graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        image_edge_keys = {
            (
                str(edge.get("source") or "").strip(),
                str(edge.get("target") or "").strip(),
                str(edge.get("type") or edge.get("relation") or "").strip(),
            )
            for edge in FunctionAlignmentAgent._iter_image_edges(image_graph)
        }
        code_edge_keys = {
            (
                str(edge.get("source") or "").strip(),
                str(edge.get("target") or "").strip(),
                str(edge.get("type") or edge.get("relation") or "").strip(),
            )
            for edge in ui_function_graph.get("edges", [])
            if isinstance(edge, dict)
        }
        validated: List[Dict[str, Any]] = []
        for item in raw_relations if isinstance(raw_relations, list) else []:
            if not isinstance(item, dict):
                continue
            image_edge = item.get("image_edge")
            code_edge = item.get("code_edge")
            if not isinstance(image_edge, dict) or not isinstance(code_edge, dict):
                continue
            normalized_image_edge = {
                "source": str(image_edge.get("source") or "").strip(),
                "target": str(image_edge.get("target") or "").strip(),
                "type": str(image_edge.get("type") or image_edge.get("relation") or "").strip(),
            }
            normalized_code_edge = {
                "source": str(code_edge.get("source") or "").strip(),
                "target": str(code_edge.get("target") or "").strip(),
                "type": str(code_edge.get("type") or code_edge.get("relation") or "").strip(),
            }
            if tuple(normalized_image_edge.values()) not in image_edge_keys:
                continue
            if tuple(normalized_code_edge.values()) not in code_edge_keys:
                continue
            validated.append(
                {
                    "image_edge": normalized_image_edge,
                    "code_edge": normalized_code_edge,
                    "aligned": bool(item.get("aligned")),
                    "reason": _truncate_text(item.get("reason") or "", 220),
                }
            )
        return validated

    @staticmethod
    def _heuristic_anchor_matches(
        image_graph: Dict[str, Any],
        ui_function_graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        graph_nodes = [
            node for node in ui_function_graph.get("nodes", [])
            if isinstance(node, dict)
        ]
        reason_map = FunctionAlignmentAgent._image_reason_map(image_graph)
        anchor_ids = FunctionAlignmentAgent._image_anchor_ids(image_graph)
        matches: List[Dict[str, Any]] = []
        for anchor in FunctionAlignmentAgent._iter_image_nodes(image_graph):
            anchor_id = str(anchor.get("id") or "").strip()
            if not anchor_id or anchor_id not in anchor_ids:
                continue
            anchor_tokens = _collect_signal_tokens(
                anchor.get("raw_type"),
                anchor.get("type"),
                anchor.get("text"),
                reason_map.get(anchor_id, ""),
                " ".join(_extract_attribute_roles(anchor)),
            )
            best_node: Optional[Dict[str, Any]] = None
            best_score = -1
            for node in graph_nodes:
                node_tokens = _collect_signal_tokens(
                    node.get("name"),
                    node.get("file"),
                )
                score = len(anchor_tokens & node_tokens)
                file_stem = os.path.splitext(os.path.basename(str(node.get("file") or "")))[0]
                if file_stem:
                    file_stem_tokens = _collect_signal_tokens(file_stem)
                    if anchor_tokens & file_stem_tokens:
                        score += 1
                if anchor_id in anchor_ids and str(node.get("name") or "").strip():
                    score += 1
                if score > best_score:
                    best_score = score
                    best_node = node
            if best_node is None:
                continue
            matches.append(
                {
                    "image_node_id": anchor_id,
                    "code_node_id": str(best_node.get("id") or "").strip(),
                    "file": _normalize_file_path(str(best_node.get("file") or "").strip()),
                    "function": str(best_node.get("name") or "").strip(),
                    "confidence": FunctionAlignmentAgent._confidence_from_signal(best_score),
                    "reason": _truncate_text(
                        f"heuristic anchor overlap between image anchor '{anchor.get('text') or anchor.get('type') or anchor_id}' "
                        f"and code node '{best_node.get('file')}::{best_node.get('name')}'.",
                        220,
                    ),
                }
            )
        return matches

    @staticmethod
    def _heuristic_relation_alignment(
        image_graph: Dict[str, Any],
        ui_function_graph: Dict[str, Any],
        anchor_matches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        image_to_code = {
            str(item.get("image_node_id") or "").strip(): str(item.get("code_node_id") or "").strip()
            for item in anchor_matches
            if isinstance(item, dict)
        }
        code_edge_keys = {
            (
                str(edge.get("source") or "").strip(),
                str(edge.get("target") or "").strip(),
                str(edge.get("type") or "").strip(),
            )
            for edge in ui_function_graph.get("edges", [])
            if isinstance(edge, dict)
        }
        preferred_map = {
            "contains": {"renders", "passes_props"},
            "is_above": {"applies_style", "renders"},
            "is_below": {"applies_style", "renders"},
            "is_left_of": {"applies_style", "renders"},
            "is_right_of": {"applies_style", "renders"},
            "triggers": {"calls", "writes_state"},
            "related_to": {"calls", "reads_state", "writes_state"},
        }
        results: List[Dict[str, Any]] = []
        for edge in FunctionAlignmentAgent._iter_image_edges(image_graph):
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            edge_type = str(edge.get("type") or edge.get("relation") or "").strip()
            source_code = image_to_code.get(source, "")
            target_code = image_to_code.get(target, "")
            if not source_code or not target_code:
                continue
            code_edge = None
            for candidate_type in preferred_map.get(edge_type, {edge_type, "calls", "renders"}):
                if (source_code, target_code, candidate_type) in code_edge_keys:
                    code_edge = {"source": source_code, "target": target_code, "type": candidate_type}
                    break
            if code_edge is None:
                continue
            results.append(
                {
                    "image_edge": {"source": source, "target": target, "type": edge_type},
                    "code_edge": code_edge,
                    "aligned": True,
                    "reason": _truncate_text(
                        f"heuristic alignment: image relation {edge_type} is supported by code edge {code_edge['type']}.",
                        220,
                    ),
                }
            )
        return results

    @staticmethod
    def _build_code_subgraph(
        ui_function_graph: Dict[str, Any],
        anchor_matches: List[Dict[str, Any]],
        relation_alignment: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        relevant_node_ids: List[str] = []
        for item in anchor_matches:
            node_id = str(item.get("code_node_id") or "").strip()
            if node_id and node_id not in relevant_node_ids:
                relevant_node_ids.append(node_id)
        for item in relation_alignment:
            if not isinstance(item, dict):
                continue
            code_edge = item.get("code_edge")
            if not isinstance(code_edge, dict):
                continue
            for key in ("source", "target"):
                node_id = str(code_edge.get(key) or "").strip()
                if node_id and node_id not in relevant_node_ids:
                    relevant_node_ids.append(node_id)

        edge_list = [
            edge for edge in ui_function_graph.get("edges", [])
            if isinstance(edge, dict)
            and str(edge.get("source") or "").strip() in relevant_node_ids
            and str(edge.get("target") or "").strip() in relevant_node_ids
        ]
        node_list = [
            node for node in ui_function_graph.get("nodes", [])
            if isinstance(node, dict) and str(node.get("id") or "").strip() in relevant_node_ids
        ]
        return {"nodes": node_list, "edges": edge_list}

    @staticmethod
    def _validate_edit_targets(
        raw_targets: Any,
        ui_function_graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        node_by_key = {
            (
                _normalize_file_path(str(node.get("file") or "").strip()),
                str(node.get("name") or "").strip(),
            ): node
            for node in ui_function_graph.get("nodes", [])
            if isinstance(node, dict)
        }
        validated: List[Dict[str, Any]] = []
        for item in raw_targets if isinstance(raw_targets, list) else []:
            if not isinstance(item, dict):
                continue
            function_name = str(item.get("function") or "").strip()
            file_path = _normalize_file_path(str(item.get("file") or "").strip())
            node = node_by_key.get((file_path, function_name))
            if not node:
                continue
            validated.append(
                {
                    "function": function_name,
                    "file": file_path,
                    "role": str(item.get("role") or "context").strip().lower() or "context",
                    "confidence": str(item.get("confidence") or "low").strip().lower() or "low",
                    "reason": _truncate_text(item.get("reason") or "", 220),
                    "code_node_id": str(node.get("id") or "").strip(),
                }
            )
        return _limit_edit_targets(validated)

    @staticmethod
    def _heuristic_edit_targets(
        image_graph: Dict[str, Any],
        problem_statement: str,
        ui_function_graph: Dict[str, Any],
        anchor_matches: List[Dict[str, Any]],
        relation_alignment: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        node_by_id = {
            str(node.get("id") or "").strip(): node
            for node in ui_function_graph.get("nodes", [])
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        }
        relation_node_ids: List[str] = []
        for item in relation_alignment:
            code_edge = item.get("code_edge")
            if not isinstance(code_edge, dict):
                continue
            for key in ("source", "target"):
                node_id = str(code_edge.get(key) or "").strip()
                if node_id and node_id not in relation_node_ids:
                    relation_node_ids.append(node_id)
        hints = set(FunctionAlignmentAgent._simple_image_hints(image_graph, problem_statement))
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for item in anchor_matches:
            node = node_by_id.get(str(item.get("code_node_id") or "").strip())
            if not node:
                continue
            score = 1
            if str(item.get("code_node_id") or "").strip() in relation_node_ids:
                score += 3
            node_tokens = _collect_signal_tokens(node.get("file"), node.get("name"))
            overlap = hints & node_tokens
            if overlap:
                score += 2
            role = "primary" if score >= 7 else ("secondary" if score >= 4 else "context")
            scored.append(
                (
                    score,
                    {
                        "function": str(node.get("name") or "").strip(),
                        "file": _normalize_file_path(str(node.get("file") or "").strip()),
                        "role": role,
                        "confidence": FunctionAlignmentAgent._confidence_from_signal(score),
                        "reason": _truncate_text(
                            f"heuristic selection using file/function overlap {sorted(overlap)}, and aligned graph context.",
                            220,
                        ),
                        "code_node_id": str(node.get("id") or "").strip(),
                    },
                ),
            )
        scored.sort(key=lambda item: (-item[0], item[1]["file"], item[1]["function"]))
        return _limit_edit_targets([item for _, item in scored])

    def run(
        self,
        problem_statement: str,
        image_graph: Dict[str, Any],
        ui_function_graph: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        trace: List[Dict[str, Any]] = []
        image_graph_summary = build_visual_summary(image_graph) or "none"
        ui_function_graph_summary = render_ui_function_graph_summary(ui_function_graph)
        prompt_a = FUNCTION_ALIGNMENT_STAGE_A_PROMPT.format(
            problem_statement=problem_statement,
            image_graph_summary=image_graph_summary,
            ui_function_graph_summary=ui_function_graph_summary,
        )
        raw_stage_a = self._query_text_model(
            system_prompt=FUNCTION_ALIGNMENT_STAGE_A_SYSTEM_PROMPT,
            user_prompt=prompt_a,
        )
        parsed_stage_a = self._safe_json_or_empty(raw_stage_a)
        anchor_matches = parsed_stage_a.get("anchor_matches", [])
        relation_alignment = parsed_stage_a.get("relation_alignment", [])
        if not anchor_matches:
            anchor_matches = self._heuristic_anchor_matches(image_graph, ui_function_graph)
            trace.append({"step": "prompt_a", "decision": "fallback_anchor_matches"})
        if not relation_alignment:
            relation_alignment = self._heuristic_relation_alignment(
                image_graph,
                ui_function_graph,
                anchor_matches,
            )
            trace.append({"step": "prompt_a", "decision": "fallback_relation_alignment"})

        code_subgraph = self._build_code_subgraph(
            ui_function_graph=ui_function_graph,
            anchor_matches=anchor_matches,
            relation_alignment=relation_alignment,
        )
        code_subgraph_summary = render_ui_function_graph_summary(code_subgraph)
        prompt_b = FUNCTION_ALIGNMENT_STAGE_B_PROMPT.format(
            problem_statement=problem_statement,
            image_graph_summary=image_graph_summary,
            anchor_matches_json=self._json_dumps(anchor_matches),
            relation_alignment_json=self._json_dumps(relation_alignment),
            code_subgraph_summary=code_subgraph_summary,
        )
        raw_stage_b = self._query_text_model(
            system_prompt=FUNCTION_ALIGNMENT_STAGE_B_SYSTEM_PROMPT,
            user_prompt=prompt_b,
        )
        parsed_stage_b = self._safe_json_or_empty(raw_stage_b)
        edit_targets = parsed_stage_b.get("edit_targets", [])
        if not edit_targets:
            edit_targets = self._heuristic_edit_targets(
                image_graph=image_graph,
                problem_statement=problem_statement,
                ui_function_graph=code_subgraph,
                anchor_matches=anchor_matches,
                relation_alignment=relation_alignment,
            )
            trace.append({"step": "prompt_b", "decision": "fallback_edit_targets"})

        trace.append(
            {
                "step": "commit",
                "anchor_match_count": len(anchor_matches),
                "relation_alignment_count": len(relation_alignment),
                "edit_target_count": len(edit_targets),
            }
        )
        return {
            "anchor_matches": anchor_matches,
            "relation_alignment": relation_alignment,
            "code_subgraph": code_subgraph,
            "edit_targets": edit_targets,
        }, trace, ""


class CodeGraphBuilder:
    def __init__(
        self,
        model_name: str = "",
        base_url: str = "",
        api_key: str = "",
        top_n_candidate_files: int = DEFAULT_TOP_N_CANDIDATE_FILES,
        top_n_seed_files: int = DEFAULT_TOP_N_SEED_FILES,
    ) -> None:
        self._cache_lock = threading.Lock()
        self._repo_cache: Dict[str, Dict[str, Any]] = {}
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.top_n_candidate_files = top_n_candidate_files
        self.top_n_seed_files = top_n_seed_files
        self.last_seed_diagnostics: Dict[str, Any] = {}

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0.0, float(seconds))
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = total_seconds - hours * 3600 - minutes * 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    @staticmethod
    def _usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
        return {
            key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
            for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens")
        }

    def _run_tracked_step(
        self,
        stage_name: str,
        step_name: str,
        func,
        *args,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        started_at = time.time()
        usage_before = get_usage_totals()
        result = func(*args, **kwargs)
        usage_after = get_usage_totals()
        metrics = {
            "step": step_name,
            "duration_seconds": time.time() - started_at,
            "usage": self._usage_delta(usage_before, usage_after),
        }
        usage = metrics["usage"]
        print(
            f"[LOCALIZATION] {stage_name}.{step_name}: "
            f"duration={self._format_duration(metrics['duration_seconds'])}, "
            f"requests={usage['requests']}, "
            f"prompt_tokens={usage['prompt_tokens']}, "
            f"completion_tokens={usage['completion_tokens']}, "
            f"total_tokens={usage['total_tokens']}"
        )
        return result, metrics

    @staticmethod
    def _build_snapshot_payload(
        instance_id: str,
        repo_identifier: str,
        base_commit: str,
        repo_dir_abs: str,
        checkout_base_commit: bool,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "instance_id": instance_id,
            "repo": repo_identifier,
            "base_commit": base_commit,
            "repo_dir": repo_dir_abs,
            "checkout_base_commit": checkout_base_commit,
            "fallback_reason": snapshot.get("fallback_reason", ""),
            "repo_structure": snapshot.get("prompt_tree", {}),
            "code_graph_snapshot": {
                "prompt_tree": snapshot.get("prompt_tree", {}),
            },
        }

    def build_instance_snapshot(
        self,
        instance_id: str,
        repo_identifier: str,
        base_commit: str,
        repo_dir: str,
        checkout_base_commit: bool = True,
    ) -> Dict[str, Any]:
        repo_dir_abs = os.path.abspath(repo_dir)
        cache_key = f"{repo_dir_abs}@@{base_commit if checkout_base_commit else 'working_tree'}"
        snapshot = self._get_or_build_structure_snapshot(
            cache_key=cache_key,
            repo_dir=repo_dir_abs,
            base_commit=base_commit,
            checkout_base_commit=checkout_base_commit,
        )
        return self._build_snapshot_payload(
            instance_id=instance_id,
            repo_identifier=repo_identifier,
            base_commit=base_commit,
            repo_dir_abs=repo_dir_abs,
            checkout_base_commit=checkout_base_commit,
            snapshot=snapshot,
        )

    @staticmethod
    def _extract_snapshot_from_payload(snapshot_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(snapshot_payload, dict):
            return None
        snapshot = snapshot_payload.get("code_graph_snapshot")
        if not isinstance(snapshot, dict):
            return None
        prompt_tree = snapshot.get("prompt_tree")
        if not isinstance(prompt_tree, dict):
            return None
        return {
            "prompt_tree": prompt_tree,
            "available_file_paths": [],
            "forward_neighbors": {},
            "fallback_reason": str(snapshot_payload.get("fallback_reason") or "").strip(),
        }

    @staticmethod
    def _extract_seed_stage_from_payload(snapshot_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(snapshot_payload, dict):
            return {}
        snapshot = snapshot_payload.get("code_graph_snapshot")
        if not isinstance(snapshot, dict):
            return {}
        extracted: Dict[str, Any] = {}
        for key in ("candidate_files", "candidate_file_graph", "seed_files", "candidate_forward_neighbors", "seed_summary"):
            value = snapshot.get(key)
            if value is not None:
                extracted[key] = value
        return extracted

    def build_instance_seed_stage(
        self,
        instance_id: str,
        repo_identifier: str,
        base_commit: str,
        repo_dir: str,
        problem_statement: str = "",
        image_graph: Any = None,
        checkout_base_commit: bool = True,
    ) -> Dict[str, Any]:
        normalized_image_graph = _coerce_image_graph(image_graph)
        snapshot_payload, snapshot_metrics = self._run_tracked_step(
            "build-code-graph",
            "build_snapshot",
            self.build_instance_snapshot,
            instance_id=instance_id,
            repo_identifier=repo_identifier,
            base_commit=base_commit,
            repo_dir=repo_dir,
            checkout_base_commit=checkout_base_commit,
        )
        repo_dir_abs = os.path.abspath(repo_dir)
        cache_key = f"{repo_dir_abs}@@{base_commit if checkout_base_commit else 'working_tree'}"
        snapshot = self._get_or_build_structure_snapshot(
            cache_key=cache_key,
            repo_dir=repo_dir_abs,
            base_commit=base_commit,
            checkout_base_commit=checkout_base_commit,
        )
        available_files = snapshot.get("available_file_paths", [])
        prompt_tree = snapshot.get("prompt_tree", {})
        candidate_files, candidate_metrics = self._run_tracked_step(
            "build-code-graph",
            "select_candidate_files",
            self._select_candidate_files_with_llm,
            problem_statement=problem_statement,
            structure=prompt_tree,
            all_files=available_files,
            image_graph=normalized_image_graph,
        )
        candidate_forward_neighbors, neighbor_metrics = self._run_tracked_step(
            "build-code-graph",
            "build_candidate_forward_neighbors",
            self._build_forward_neighbors,
            available_files=candidate_files,
            repo_dir=repo_dir_abs,
        )
        candidate_file_graph, graph_metrics = self._run_tracked_step(
            "build-code-graph",
            "build_candidate_file_graph",
            build_candidate_file_graph,
            candidate_files=candidate_files,
            forward_neighbors=candidate_forward_neighbors,
        )
        seed_files, seed_metrics = self._run_tracked_step(
            "build-code-graph",
            "select_seed_files",
            self._select_seed_files_from_candidates_with_llm,
            problem_statement=problem_statement,
            candidate_files=candidate_files,
            candidate_file_graph=candidate_file_graph,
            image_graph=normalized_image_graph,
        )
        snapshot_payload["code_graph_snapshot"].update({
            "candidate_files": candidate_files,
            "candidate_forward_neighbors": candidate_forward_neighbors,
            "candidate_file_graph": candidate_file_graph,
            "seed_files": seed_files,
            "seed_summary": {
                "candidate_file_count": len(candidate_files),
                "seed_file_count": len(seed_files),
            },
            "build_code_graph_metrics": [
                snapshot_metrics,
                candidate_metrics,
                neighbor_metrics,
                graph_metrics,
                seed_metrics,
            ],
        })
        return snapshot_payload

    def build_instance_graph(
        self,
        instance_id: str,
        repo_identifier: str,
        base_commit: str,
        repo_dir: str,
        problem_statement: str = "",
        image_graph: Any = None,
        checkout_base_commit: bool = True,
        snapshot_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_image_graph = _coerce_image_graph(image_graph)
        repo_dir_abs = os.path.abspath(repo_dir)
        snapshot = self._extract_snapshot_from_payload(snapshot_payload)
        cache_key = f"{repo_dir_abs}@@{base_commit if checkout_base_commit else 'working_tree'}"
        computed_snapshot = self._get_or_build_structure_snapshot(
            cache_key=cache_key,
            repo_dir=repo_dir_abs,
            base_commit=base_commit,
            checkout_base_commit=checkout_base_commit,
        )
        if snapshot is None:
            snapshot = computed_snapshot
        else:
            snapshot = {
                "available_file_paths": computed_snapshot.get("available_file_paths", []),
                "prompt_tree": snapshot.get("prompt_tree") or computed_snapshot.get("prompt_tree", {}),
                "fallback_reason": snapshot.get("fallback_reason") or computed_snapshot.get("fallback_reason", ""),
            }

        available_files = snapshot["available_file_paths"]
        prompt_tree = snapshot["prompt_tree"]
        seed_stage_payload = self._extract_seed_stage_from_payload(snapshot_payload)
        candidate_files = seed_stage_payload.get("candidate_files")
        candidate_forward_neighbors = seed_stage_payload.get("candidate_forward_neighbors")
        candidate_file_graph = seed_stage_payload.get("candidate_file_graph")
        seed_files = seed_stage_payload.get("seed_files")
        if not isinstance(candidate_files, list) or not isinstance(seed_files, list):
            candidate_files = self._select_candidate_files_with_llm(
                problem_statement=problem_statement,
                structure=prompt_tree,
                all_files=available_files,
                image_graph=normalized_image_graph,
            )
            candidate_forward_neighbors = self._build_forward_neighbors(
                available_files=candidate_files,
                repo_dir=repo_dir_abs,
            )
            candidate_file_graph = build_candidate_file_graph(
                candidate_files=candidate_files,
                forward_neighbors=candidate_forward_neighbors,
            )
            seed_files = self._select_seed_files_from_candidates_with_llm(
                problem_statement=problem_statement,
                candidate_files=candidate_files,
                candidate_file_graph=candidate_file_graph,
                image_graph=normalized_image_graph,
            )
        if not isinstance(candidate_forward_neighbors, dict):
            candidate_forward_neighbors = self._build_forward_neighbors(
                available_files=candidate_files,
                repo_dir=repo_dir_abs,
            )
        if not isinstance(candidate_file_graph, dict):
            candidate_file_graph = build_candidate_file_graph(
                candidate_files=candidate_files,
                forward_neighbors=candidate_forward_neighbors,
            )
        (
            first_order_result,
            align_metrics,
        ) = self._run_tracked_step(
            "align-code-graph",
            "build_first_order_module_graph",
            self.build_first_order_module_graph,
            normalized_seed_files=seed_files,
            problem_statement=problem_statement,
            repo_dir=repo_dir_abs,
            available_files=available_files,
            forward_neighbors=candidate_forward_neighbors,
            image_graph=normalized_image_graph,
        )
        (
            candidate_function_pool,
            ui_function_graph,
            anchor_alignment,
            edit_targets,
            alignment_fallback_reason,
        ) = first_order_result

        summary = {
            "candidate_file_count": len(candidate_files),
            "seed_file_count": len(seed_files),
            "candidate_function_count": len(candidate_function_pool),
            "ui_function_graph_node_count": len(ui_function_graph.get("nodes", [])),
            "ui_function_graph_edge_count": len(ui_function_graph.get("edges", [])),
            "anchor_match_count": len(anchor_alignment.get("anchor_matches", [])),
            "relation_alignment_count": len(anchor_alignment.get("relation_alignment", [])),
            "edit_target_count": len(edit_targets),
            "alignment_trace_step_count": len(anchor_alignment.get("alignment_trace", [])),
            "alignment_fallback_reason": alignment_fallback_reason,
            "align_code_graph_metrics": [align_metrics],
        }

        return {
            "instance_id": instance_id,
            "repo": repo_identifier,
            "base_commit": base_commit,
            "repo_dir": repo_dir_abs,
            "checkout_base_commit": checkout_base_commit,
            "fallback_reason": snapshot.get("fallback_reason", ""),
            "code_graph": {
                "candidate_file_graph": candidate_file_graph,
                "seed_files": seed_files,
                "ui_function_graph": ui_function_graph,
                "anchor_alignment": anchor_alignment,
                "edit_targets": edit_targets,
                "summary": summary,
            },
        }

    def _get_or_build_structure_snapshot(
        self,
        cache_key: str,
        repo_dir: str,
        base_commit: str,
        checkout_base_commit: bool,
    ) -> Dict[str, Any]:
        with self._cache_lock:
            cached = self._repo_cache.get(cache_key)
            if cached is not None:
                return cached

        parse_dir, tmp_root, fallback_reason = self._prepare_parse_dir(
            repo_dir=repo_dir,
            base_commit=base_commit,
            checkout_base_commit=checkout_base_commit,
        )
        try:
            available_files = build_available_file_list(parse_dir)
            prompt_tree = build_prompt_tree(available_files, os.path.basename(parse_dir.rstrip("\\/")))
        finally:
            if tmp_root:
                shutil.rmtree(tmp_root, ignore_errors=True)

        snapshot = {
            "available_file_paths": available_files,
            "prompt_tree": prompt_tree,
            "fallback_reason": fallback_reason,
        }
        with self._cache_lock:
            self._repo_cache[cache_key] = snapshot
        return snapshot

    def _prepare_parse_dir(
        self,
        repo_dir: str,
        base_commit: str,
        checkout_base_commit: bool,
    ) -> Tuple[str, str, str]:
        fallback_reason = ""
        parse_dir = repo_dir
        tmp_root = ""
        if checkout_base_commit:
            if base_commit:
                try:
                    parse_dir, tmp_root = self._prepare_commit_checkout(repo_dir=repo_dir, base_commit=base_commit)
                except Exception as exc:
                    fallback_reason = f"checkout_failed: {exc}"
                    parse_dir = repo_dir
            else:
                fallback_reason = "checkout_skipped_missing_base_commit"
        return parse_dir, tmp_root, fallback_reason

    @staticmethod
    def _prepare_commit_checkout(repo_dir: str, base_commit: str) -> Tuple[str, str]:
        tmp_root = tempfile.mkdtemp(prefix="gala_code_graph_")
        checkout_dir = os.path.join(tmp_root, "repo")

        subprocess.run(
            ["git", "clone", "--shared", repo_dir, checkout_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "-C", checkout_dir, "checkout", "--detach", base_commit],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return checkout_dir, tmp_root

    @staticmethod
    def _extract_text_content(response_data: Dict[str, Any]) -> str:
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                    chunks.append(chunk["text"])
                elif isinstance(chunk, str):
                    chunks.append(chunk)
            return "\n".join(chunks)
        return ""

    @staticmethod
    def _parse_ranked_files(content: str) -> List[str]:
        if not isinstance(content, str) or not content.strip():
            return []
        if content.count("```") % 2 != 0:
            return []
        extracted_output = re.findall(r"```(?:.*?\n)?(.*?)```", content, re.DOTALL)
        if not extracted_output:
            return []
        lines = "\n".join(extracted_output).strip().split("\n")
        parsed_list = []
        for line in lines:
            candidate = line.strip()
            if _is_candidate_file(candidate):
                parsed_list.append(_normalize_file_path(candidate))
        return parsed_list

    @staticmethod
    def _correct_file_paths(model_found_files: List[str], files: List[str]) -> List[str]:
        found_files: List[str] = []
        file_set = set(files)
        basename_to_paths: Dict[str, List[str]] = {}
        for path in files:
            basename_to_paths.setdefault(os.path.basename(path), []).append(path)

        for model_file in model_found_files:
            normalized = _normalize_file_path(model_file)
            if normalized in file_set:
                if normalized not in found_files:
                    found_files.append(normalized)
                continue

            basename = os.path.basename(normalized)
            basename_matches = basename_to_paths.get(basename, [])
            if len(basename_matches) == 1:
                candidate = basename_matches[0]
                if candidate not in found_files:
                    found_files.append(candidate)
                continue

            close = difflib.get_close_matches(normalized, files, n=1, cutoff=0.7)
            if close:
                candidate = close[0]
                if candidate not in found_files:
                    found_files.append(candidate)
        return found_files

    def _query_text_model(self, system_prompt: str, user_prompt: str) -> str:
        response = send_chat_completion(
            api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return self._extract_text_content(response)

    def _select_candidate_files_with_llm(
        self,
        problem_statement: str,
        structure: Dict[str, Any],
        all_files: List[str],
        image_graph: Any = None,
    ) -> List[str]:
        def finalize_candidates(primary_files: List[str]) -> List[str]:
            deduped: List[str] = []
            for file_path in primary_files:
                normalized = _normalize_file_path(file_path)
                if normalized in all_files and normalized not in deduped:
                    deduped.append(normalized)
            return deduped[: self.top_n_candidate_files]

        diagnostics: Dict[str, Any] = {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "all_files_count": len(all_files),
            "reason": "",
        }
        if not self.model_name or not self.base_url or not self.api_key:
            diagnostics["reason"] = "missing_llm_config"
            diagnostics["initial_candidate_count"] = 0
            diagnostics["final_files"] = finalize_candidates([])
            diagnostics["final_candidate_count"] = len(diagnostics["final_files"])
            self.last_seed_diagnostics = diagnostics
            print(
                "candidate file stats: "
                "initial_candidates=0, "
                f"final_candidates={len(diagnostics['final_files'])}"
            )
            return diagnostics["final_files"]
        if not all_files:
            diagnostics["reason"] = "no_candidate_files_indexed"
            self.last_seed_diagnostics = diagnostics
            return []

        def get_best_match(file_path: str, candidates: List[str], cutoff: float = 0.8) -> str:
            if file_path in candidates:
                return file_path
            matches = difflib.get_close_matches(file_path, candidates, n=1, cutoff=cutoff)
            return matches[0] if matches else file_path

        repo_tree = show_project_structure(structure).strip()
        visual_summary = build_visual_summary(image_graph)
        if visual_summary:
            user_msg = FILE_VISUAL_SEED_PROMPT.format(
                issue=problem_statement,
                visual_summary=visual_summary,
                repo_tree=repo_tree,
            )
            diagnostics["visual_summary_preview"] = visual_summary[:500]
        else:
            bug_report = BUG_REPORT_TEMPLATE.format(
                problem_statement=problem_statement,
                structure=repo_tree,
            )
            user_msg = f"""
{bug_report}
{FILE_GUIDENCE_PRMPT_WITHOUT_TOOL}
{FILE_SUMMARY}
""".strip()

        try:
            raw_output = self._query_text_model(
                system_prompt=FILE_SYSTEM_PROMPT_WITHOUT_TOOL,
                user_prompt=user_msg,
            )
            diagnostics["raw_output_preview"] = raw_output[:500]
            model_found_files = self._parse_ranked_files(raw_output)
            diagnostics["model_found_files"] = model_found_files
            found_files = [f for f in model_found_files if f in all_files]
            diagnostics["found_files_after_first_pass"] = found_files

            if len(found_files) == 0:
                corrected_tpl = FORMAT_CORRECT_PROMPT.format(res=raw_output)
                formatted_res = self._query_text_model(
                    system_prompt="You are a helpful assistant.",
                    user_prompt=corrected_tpl,
                )
                diagnostics["formatted_output_preview"] = formatted_res[:500]
                model_found_files = self._parse_ranked_files(formatted_res)
                diagnostics["model_found_files_after_format_fix"] = model_found_files
                found_files = [f for f in model_found_files if f in all_files]
                diagnostics["found_files_after_format_fix"] = found_files
                if len(found_files) == 0:
                    found_files = [get_best_match(f, all_files) for f in model_found_files]
                    found_files = [f for f in found_files if f in all_files]
                    diagnostics["found_files_after_best_match"] = found_files

            result = finalize_candidates(found_files)
            diagnostics["initial_candidate_count"] = len(model_found_files)
            diagnostics["final_candidate_count"] = len(result)
            print(
                "candidate file stats: "
                f"initial_candidates={len(model_found_files)}, "
                f"final_candidates={len(result)}"
            )
            if result:
                diagnostics["reason"] = "ok"
            elif diagnostics.get("model_found_files") or diagnostics.get("model_found_files_after_format_fix"):
                diagnostics["reason"] = "llm_returned_non_matching_paths"
            else:
                diagnostics["reason"] = "llm_output_unparseable_or_empty"
            diagnostics["final_files"] = result
            self.last_seed_diagnostics = diagnostics
            return result
        except Exception as exc:
            print(f"CoSIL-style file localization failed: {exc}")
            diagnostics["reason"] = "llm_request_failed"
            diagnostics["exception"] = str(exc)
            self.last_seed_diagnostics = diagnostics
            return []

    def _select_seed_files_from_candidates_with_llm(
        self,
        problem_statement: str,
        candidate_files: List[str],
        candidate_file_graph: Dict[str, Any],
        image_graph: Any = None,
    ) -> List[str]:
        normalized_candidates = _dedupe_keep_order([
            _normalize_file_path(file_path) for file_path in candidate_files if str(file_path).strip()
        ])
        if not normalized_candidates:
            return []
        if not self.model_name or not self.base_url or not self.api_key:
            return normalized_candidates[: self.top_n_seed_files]

        candidate_file_summary = "\n".join(
            f"- {file_path}" for file_path in normalized_candidates[: self.top_n_candidate_files]
        ) or "none"
        image_summary = build_visual_summary(image_graph) or "none"
        user_prompt = CANDIDATE_SEED_RERANK_PROMPT.format(
            problem_statement=problem_statement,
            image_summary=image_summary,
            candidate_file_summary=candidate_file_summary,
            candidate_file_graph_summary=render_candidate_file_graph_summary(candidate_file_graph),
            top_n_seed_files=self.top_n_seed_files,
        )
        try:
            raw_output = self._query_text_model(
                system_prompt=RERANK_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception:
            return normalized_candidates[: self.top_n_seed_files]

        parsed = _safe_json_loads(raw_output)
        if not isinstance(parsed, dict):
            return normalized_candidates[: self.top_n_seed_files]

        seed_files = [
            _normalize_file_path(file_path)
            for file_path in parsed.get("seed_files", [])
            if _normalize_file_path(file_path) in normalized_candidates
        ]
        final_files = _dedupe_keep_order(seed_files + normalized_candidates)
        return final_files[: self.top_n_seed_files]

    def _render_selected_file_summary(
        self,
        selected_files: List[str],
        repo_dir: str,
    ) -> str:
        if not selected_files:
            return "none"
        lines: List[str] = []
        for file_path in selected_files[:MAX_TRAVERSAL_FILES]:
            lines.append(f"file: {file_path}")
            if _is_graph_source_file(file_path):
                abs_path = os.path.join(repo_dir, file_path)
                try:
                    file_lines = _read_text_file(abs_path)
                except Exception:
                    file_lines = []
                classes, functions = parse_jsts_lines(file_lines)
                symbols: List[str] = []
                for clazz in classes[:4]:
                    class_name = str(clazz.get("name") or "").strip()
                    if class_name:
                        symbols.append(f"class: {class_name}")
                for function in functions[:4]:
                    function_name = str(function.get("name") or "").strip()
                    if function_name:
                        symbols.append(f"function: {function_name}")
                if symbols:
                    lines.append(f"  symbols: {', '.join(symbols[:8])}")
        return "\n".join(lines).strip() or "none"

    def _rerank_selected_files_with_llm(
        self,
        problem_statement: str,
        selected_files: List[str],
        repo_dir: str,
        image_graph: Any = None,
    ) -> List[str]:
        normalized_selected = _dedupe_keep_order([
            _normalize_file_path(file_path) for file_path in selected_files if str(file_path).strip()
        ])
        if not normalized_selected:
            return []
        if not self.model_name or not self.base_url or not self.api_key:
            return normalized_selected[:15]

        user_prompt = RERANK_SELECTED_FILES_PROMPT.format(
            problem_statement=problem_statement,
            image_summary=build_visual_summary(image_graph) or "none",
            selected_file_summary=self._render_selected_file_summary(normalized_selected, repo_dir),
        )
        try:
            raw_output = self._query_text_model(
                system_prompt=RERANK_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception:
            return normalized_selected[:15]

        parsed = _safe_json_loads(raw_output)
        if not isinstance(parsed, dict):
            return normalized_selected[:15]

        reranked = [
            _normalize_file_path(file_path)
            for file_path in parsed.get("reranked_top15_files", [])
            if _normalize_file_path(file_path) in normalized_selected
        ]
        final_files = _dedupe_keep_order(reranked + normalized_selected)
        return final_files[:15]

    def _query_alignment_plan_with_llm(
        self,
        problem_statement: str,
        seed_files: List[str],
        alignment_code_graph: Dict[str, Any],
        image_graph: Any = None,
    ) -> Tuple[Dict[str, Any], str]:
        _ = problem_statement
        _ = seed_files
        _ = alignment_code_graph
        _ = image_graph
        return {
            "alignments": [],
            "subgraph_alignments": [],
            "expansion_plan": [],
            "matched_files": [],
            "image_subgraphs": [],
            "graph_node_count": 0,
            "graph_edge_count": 0,
            "alignment_trace": [],
        }, "deprecated_function_alignment_path"

    @staticmethod
    def _make_function_node_id(file_path: str, function_name: str) -> str:
        return f"function:{_normalize_file_path(file_path)}:{function_name}"

    def _iter_seed_file_functions(
        self,
        seed_files: List[str],
        repo_dir: str,
    ) -> List[Dict[str, Any]]:
        function_items: List[Dict[str, Any]] = []
        for file_path in seed_files:
            if not _is_graph_source_file(file_path):
                continue
            abs_path = os.path.join(repo_dir, file_path)
            if not os.path.isfile(abs_path):
                continue
            try:
                lines = _read_text_file(abs_path)
            except Exception:
                continue
            classes, functions = parse_jsts_lines(lines)
            for function in functions:
                body_text = "\n".join(function.get("text", []))
                function_items.append(
                    {
                        "file": file_path,
                        "function": function.get("name", ""),
                        "start_line": function.get("start_line"),
                        "end_line": function.get("end_line"),
                        "body_text": body_text,
                        "source_kind": "function",
                    }
                )
            for clazz in classes:
                class_body_text = "\n".join(clazz.get("class_content", []))
                function_items.append(
                    {
                        "file": file_path,
                        "function": clazz.get("name", ""),
                        "start_line": clazz.get("start_line"),
                        "end_line": clazz.get("end_line"),
                        "body_text": class_body_text,
                        "source_kind": "class_component",
                    }
                )
                for method in clazz.get("methods", []):
                    function_items.append(
                        {
                            "file": file_path,
                            "function": f"{clazz.get('name', '')}.{method.get('name', '')}",
                            "start_line": method.get("start_line"),
                            "end_line": method.get("end_line"),
                            "body_text": "\n".join(method.get("method_content", [])),
                            "source_kind": "method",
                        }
                    )
        return function_items

    def _build_function_pool_from_seed_files(
        self,
        seed_files: List[str],
        repo_dir: str,
    ) -> List[Dict[str, Any]]:
        pool: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for item in self._iter_seed_file_functions(seed_files, repo_dir):
            file_path = _normalize_file_path(str(item.get("file") or "").strip())
            function_name = str(item.get("function") or "").strip()
            body_text = str(item.get("body_text") or "")
            if not file_path or not function_name or not body_text.strip():
                continue
            if _is_probably_utility_function(function_name, body_text, file_path):
                continue

            node_id = self._make_function_node_id(file_path, function_name)
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            pool.append(
                {
                    "id": node_id,
                    "type": "function",
                    "name": _format_function_display_name(file_path, function_name),
                    "file": file_path,
                    "raw_name": function_name,
                    "body_text": body_text,
                }
            )
        pool.sort(key=lambda item: (str(item.get("file") or ""), str(item.get("name") or "")))
        return pool

    def _build_ui_function_graph(
        self,
        candidate_function_pool: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        node_by_id = {
            str(node.get("id") or "").strip(): node
            for node in candidate_function_pool
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        }
        function_name_to_ids: Dict[str, List[str]] = {}
        short_name_to_ids: Dict[str, List[str]] = {}
        for node_id, node in node_by_id.items():
            raw_name = str(node.get("raw_name") or node.get("name") or "").strip()
            short_name = raw_name.split(".")[-1]
            function_name_to_ids.setdefault(raw_name, []).append(node_id)
            short_name_to_ids.setdefault(short_name, []).append(node_id)

        def resolve_target(raw_name: str) -> List[str]:
            raw = str(raw_name or "").strip()
            if not raw:
                return []
            return function_name_to_ids.get(raw, []) or short_name_to_ids.get(raw.split(".")[-1], [])

        edges: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()

        def add_edge(source: str, target: str, edge_type: str) -> None:
            key = (source, target, edge_type)
            if source == target or key in seen:
                return
            seen.add(key)
            edges.append({"source": source, "target": target, "type": edge_type})

        for node_id, node in node_by_id.items():
            body_text = str(node.get("body_text") or "")
            for target_name in _extract_jsx_component_targets(body_text):
                for target_id in resolve_target(target_name):
                    add_edge(node_id, target_id, "renders")
            for target_name in _extract_call_targets(body_text):
                for target_id in resolve_target(target_name):
                    add_edge(node_id, target_id, "calls")
            for target_name in _extract_state_reads(body_text):
                for target_id in resolve_target(target_name):
                    add_edge(node_id, target_id, "reads_state")
            for target_name in _extract_state_writes(body_text):
                for target_id in resolve_target(target_name):
                    add_edge(node_id, target_id, "writes_state")
            for target_name in _extract_props_targets(body_text):
                for target_id in resolve_target(target_name):
                    add_edge(node_id, target_id, "passes_props")
            if _extract_inline_style_markers(body_text):
                for target_name in _extract_jsx_component_targets(body_text) | _extract_props_targets(body_text):
                    for target_id in resolve_target(target_name):
                        add_edge(node_id, target_id, "applies_style")

        sanitized_nodes = []
        for node in candidate_function_pool:
            copied = dict(node)
            copied.pop("body_text", None)
            copied.pop("raw_name", None)
            sanitized_nodes.append(copied)
        return {"nodes": sanitized_nodes, "edges": edges}

    def _build_alignment_query(
        self,
        image_graph: Any,
        problem_statement: str = "",
    ) -> Dict[str, Any]:
        return {
            "deprecated": True,
            "reason": "step3_removed_step4_reads_image_graph_directly",
            "issue_focus": _truncate_text(problem_statement, 220),
            "image_graph_present": bool(_coerce_image_graph(image_graph)),
        }

    def build_first_order_module_graph(
        self,
        normalized_seed_files: List[str],
        problem_statement: str,
        repo_dir: str,
        available_files: List[str],
        forward_neighbors: Dict[str, List[str]],
        image_graph: Any = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], str]:
        normalized_image_graph = _coerce_image_graph(image_graph)
        available_file_set = {
            _normalize_file_path(str(path).strip())
            for path in available_files
            if str(path).strip()
        }
        seed_files = [file_path for file_path in normalized_seed_files if file_path in available_file_set]
        _ = forward_neighbors
        candidate_function_pool = self._build_function_pool_from_seed_files(
            seed_files=seed_files,
            repo_dir=repo_dir,
        )
        ui_function_graph, _ = self._run_tracked_step(
            "align-code-graph",
            "build_ui_function_graph",
            self._build_ui_function_graph,
            candidate_function_pool=candidate_function_pool,
        )
        print(
            f"[LOCALIZATION] align-code-graph.build_ui_function_graph.output: "
            f"seed_files={len(seed_files)}, "
            f"nodes={len(ui_function_graph.get('nodes', []))}, "
            f"edges={len(ui_function_graph.get('edges', []))}"
        )
        if not candidate_function_pool:
            return (
                [],
                {"nodes": [], "edges": []},
                {
                    "anchor_matches": [],
                    "relation_alignment": [],
                    "code_subgraph": {"nodes": [], "edges": []},
                    "alignment_trace": [],
                },
                [],
                "empty_candidate_function_pool",
            )

        if not self.model_name or not self.base_url or not self.api_key:
            anchor_alignment = {
                "anchor_matches": FunctionAlignmentAgent._heuristic_anchor_matches(normalized_image_graph or {}, ui_function_graph),
                "relation_alignment": [],
                "code_subgraph": {"nodes": [], "edges": []},
                "alignment_trace": [{"step": "bootstrap", "decision": "missing_llm_config"}],
            }
            anchor_alignment["relation_alignment"] = FunctionAlignmentAgent._heuristic_relation_alignment(
                image_graph=normalized_image_graph or {},
                ui_function_graph=ui_function_graph,
                anchor_matches=anchor_alignment["anchor_matches"],
            )
            anchor_alignment["code_subgraph"] = FunctionAlignmentAgent._build_code_subgraph(
                ui_function_graph=ui_function_graph,
                anchor_matches=anchor_alignment["anchor_matches"],
                relation_alignment=anchor_alignment["relation_alignment"],
            )
            edit_targets = FunctionAlignmentAgent._heuristic_edit_targets(
                image_graph=normalized_image_graph or {},
                problem_statement=problem_statement,
                ui_function_graph=anchor_alignment["code_subgraph"],
                anchor_matches=anchor_alignment["anchor_matches"],
                relation_alignment=anchor_alignment["relation_alignment"],
            )
            return (
                candidate_function_pool,
                ui_function_graph,
                anchor_alignment,
                edit_targets,
                "missing_llm_config",
            )

        alignment_agent = FunctionAlignmentAgent(
            query_text_model=self._query_text_model,
            top_n_seed_files=self.top_n_seed_files,
        )
        try:
            agent_output, _ = self._run_tracked_step(
                "align-code-graph",
                "run_alignment_agent",
                alignment_agent.run,
                problem_statement=problem_statement,
                image_graph=normalized_image_graph or {},
                ui_function_graph=ui_function_graph,
            )
            agent_result, alignment_trace, agent_reason = agent_output
        except Exception as exc:
            anchor_matches = FunctionAlignmentAgent._heuristic_anchor_matches(normalized_image_graph or {}, ui_function_graph)
            relation_alignment = FunctionAlignmentAgent._heuristic_relation_alignment(
                image_graph=normalized_image_graph or {},
                ui_function_graph=ui_function_graph,
                anchor_matches=anchor_matches,
            )
            code_subgraph = FunctionAlignmentAgent._build_code_subgraph(
                ui_function_graph=ui_function_graph,
                anchor_matches=anchor_matches,
                relation_alignment=relation_alignment,
            )
            edit_targets = FunctionAlignmentAgent._heuristic_edit_targets(
                image_graph=normalized_image_graph or {},
                problem_statement=problem_statement,
                ui_function_graph=code_subgraph,
                anchor_matches=anchor_matches,
                relation_alignment=relation_alignment,
            )
            return (
                candidate_function_pool,
                ui_function_graph,
                {
                    "anchor_matches": anchor_matches,
                    "relation_alignment": relation_alignment,
                    "code_subgraph": code_subgraph,
                    "alignment_trace": [{"step": "fallback", "reason": str(exc)}],
                },
                edit_targets,
                "llm_request_failed",
            )

        anchor_alignment = {
            "anchor_matches": agent_result.get("anchor_matches", []),
            "relation_alignment": agent_result.get("relation_alignment", []),
            "code_subgraph": agent_result.get("code_subgraph", {"nodes": [], "edges": []}),
            "alignment_trace": alignment_trace,
        }
        edit_targets = agent_result.get("edit_targets", [])
        return (
            candidate_function_pool,
            ui_function_graph,
            anchor_alignment,
            edit_targets,
            agent_reason or "ok",
        )

    def _parse_selected_source_files(
        self,
        selected_files: List[str],
        repo_dir: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        for file_path in selected_files:
            if not _is_graph_source_file(file_path):
                continue
            abs_path = os.path.join(repo_dir, file_path)
            if not os.path.isfile(abs_path):
                continue
            try:
                lines = _read_text_file(abs_path)
            except Exception:
                continue
            file_classes, file_functions = parse_jsts_lines(lines)
            for clazz in file_classes:
                classes.append(
                    {
                        "file": file_path,
                        "name": clazz["name"],
                        "extends": clazz.get("extends", ""),
                        "start_line": clazz["start_line"],
                        "end_line": clazz["end_line"],
                        "class_content": clazz["text"],
                        "methods": [
                            {
                                "name": method["name"],
                                "start_line": method["start_line"],
                                "end_line": method["end_line"],
                                "method_content": method["text"],
                            }
                            for method in clazz.get("methods", [])
                        ],
                    }
                )
            for function in file_functions:
                copied = dict(function)
                copied["file"] = file_path
                functions.append(copied)
        return classes, functions

    def _get_imports_from_disk(self, file_name: str, repo_dir: str) -> List[str]:
        if not _is_graph_source_file(file_name):
            return []
        abs_path = os.path.join(repo_dir, file_name)
        if not os.path.isfile(abs_path):
            return []
        try:
            lines = _read_text_file(abs_path)
        except Exception:
            return []
        imports: List[str] = []
        for line in lines:
            stripped = line.strip()
            if (
                re.match(r"^(?:import|export)\b.*\bfrom\s+['\"]", stripped)
                or "require(" in stripped
                or re.search(r"\bimport\(\s*['\"]", stripped)
            ):
                imports.append(line)
        return imports

    @staticmethod
    def _extract_import_targets(line: str) -> List[str]:
        stripped = line.strip()
        targets: List[str] = []
        patterns = [
            r"(?:import|export)\b.*?\bfrom\s+['\"]([^'\"]+)['\"]",
            r"require\(\s*['\"]([^'\"]+)['\"]\s*\)",
            r"\bimport\(\s*['\"]([^'\"]+)['\"]\s*\)",
        ]
        for pattern in patterns:
            targets.extend(re.findall(pattern, stripped))
        return targets

    def _resolve_import_targets_for_file(
        self,
        source_file: str,
        imports: List[str],
        available_files: Set[str],
    ) -> List[str]:
        resolved: List[str] = []
        seen: Set[str] = set()
        for line in imports:
            for module_name in self._extract_import_targets(line):
                target_file = self._resolve_import_target(source_file, module_name, available_files)
                if not target_file or target_file in seen:
                    continue
                seen.add(target_file)
                resolved.append(target_file)
        return resolved

    @staticmethod
    def _collect_import_edges_for_files(
        selected_files: List[str],
        forward_neighbors: Dict[str, List[str]],
    ) -> List[Tuple[str, str]]:
        selected_set = {_normalize_file_path(path) for path in selected_files if str(path).strip()}
        import_edges: List[Tuple[str, str]] = []
        for source_file in selected_files:
            normalized_source = _normalize_file_path(source_file)
            if normalized_source not in selected_set:
                continue
            for target_file in forward_neighbors.get(normalized_source, []):
                normalized_target = _normalize_file_path(target_file)
                if normalized_target in selected_set:
                    import_edges.append((normalized_source, normalized_target))
        return import_edges

    def _build_code_graph_for_files(
        self,
        selected_files: List[str],
        repo_dir: str,
        forward_neighbors: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        normalized_files = [
            normalized
            for normalized in _dedupe_keep_order(selected_files)
            if normalized
        ]
        classes, functions = self._parse_selected_source_files(
            selected_files=normalized_files,
            repo_dir=repo_dir,
        )
        import_edges = self._collect_import_edges_for_files(
            selected_files=normalized_files,
            forward_neighbors=forward_neighbors,
        )
        nodes, edges = self._build_nodes_and_edges(
            selected_files=normalized_files,
            classes=classes,
            functions=functions,
            import_edges=import_edges,
            repo_dir=repo_dir,
        )
        return {
            "files": normalized_files,
            "nodes": nodes,
            "edges": edges,
            "import_edges": import_edges,
        }

    def _build_forward_neighbors(
        self,
        available_files: List[str],
        repo_dir: str,
    ) -> Dict[str, List[str]]:
        available_file_set = set(available_files)
        forward_neighbors: Dict[str, List[str]] = {}
        for file_path in available_files:
            if not _is_graph_source_file(file_path):
                continue
            imports = self._get_imports_from_disk(file_path, repo_dir)
            forward_neighbors[file_path] = self._resolve_import_targets_for_file(
                file_path,
                imports,
                available_file_set,
            )
        return forward_neighbors

    @staticmethod
    def _build_reverse_neighbors(forward_neighbors: Dict[str, List[str]]) -> Dict[str, List[str]]:
        reverse_neighbors: Dict[str, List[str]] = {}
        for source_file, targets in forward_neighbors.items():
            for target_file in targets:
                reverse_neighbors.setdefault(target_file, []).append(source_file)
        for target_file, sources in reverse_neighbors.items():
            reverse_neighbors[target_file] = sorted(set(sources))
        return reverse_neighbors

    @staticmethod
    def _build_related_available_neighbors(available_files: List[str]) -> Dict[str, List[str]]:
        by_key: Dict[Tuple[str, str], List[str]] = {}
        for file_path in available_files:
            directory = os.path.dirname(file_path)
            stem = os.path.splitext(os.path.basename(file_path))[0]
            by_key.setdefault((directory, stem), []).append(file_path)

        related: Dict[str, List[str]] = {}
        for _, paths in by_key.items():
            if len(paths) < 2:
                continue
            unique_paths = sorted(set(paths))
            for file_path in unique_paths:
                related[file_path] = [path for path in unique_paths if path != file_path]

        root_files = set(available_files)
        package_related = {
            "package.json": ["package-lock.json"],
            "package-lock.json": ["package.json"],
        }
        for file_path, neighbors in package_related.items():
            if file_path not in root_files:
                continue
            related.setdefault(file_path, [])
            for neighbor in neighbors:
                if neighbor in root_files and neighbor not in related[file_path]:
                    related[file_path].append(neighbor)
        return related

    @staticmethod
    def _resolve_import_target(
        source_file: str,
        module_name: str,
        available_files: Set[str],
    ) -> Optional[str]:
        module_name = str(module_name or "").strip()
        if not module_name or module_name.startswith(("#", "node:")):
            return None

        source_dir = os.path.dirname(source_file)
        candidates: List[str] = []
        if module_name.startswith("."):
            base = _normalize_file_path(os.path.normpath(os.path.join(source_dir, module_name)))
            candidates.extend(
                [
                    base,
                    f"{base}.js",
                    f"{base}.jsx",
                    f"{base}.ts",
                    f"{base}.tsx",
                    f"{base}.scss",
                    f"{base}.css",
                    f"{base}.json",
                    f"{base}.html",
                    f"{base}.md",
                    f"{base}.mdx",
                    f"{base}.svg",
                    f"{base}.frag",
                    f"{base}.vert",
                    f"{base}/index.js",
                    f"{base}/index.jsx",
                    f"{base}/index.ts",
                    f"{base}/index.tsx",
                ]
            )
        else:
            normalized = _normalize_file_path(module_name)
            candidates.extend(
                [
                    normalized,
                    f"{normalized}.js",
                    f"{normalized}.jsx",
                    f"{normalized}.ts",
                    f"{normalized}.tsx",
                    f"{normalized}.scss",
                    f"{normalized}.css",
                    f"{normalized}.json",
                    f"{normalized}.html",
                    f"{normalized}.md",
                    f"{normalized}.mdx",
                    f"{normalized}.svg",
                    f"{normalized}.frag",
                    f"{normalized}.vert",
                    f"{normalized}/index.js",
                    f"{normalized}/index.jsx",
                    f"{normalized}/index.ts",
                    f"{normalized}/index.tsx",
                ]
            )
        for candidate in candidates:
            if candidate in available_files:
                return candidate
        return None

    @staticmethod
    def _build_nodes_and_edges(
        selected_files: List[str],
        classes: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        import_edges: List[Tuple[str, str]],
        repo_dir: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        selected_file_set = set(selected_files)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_node_ids: Set[str] = set()
        seen_edges: Set[Tuple[str, str, str]] = set()
        class_node_id_by_file_and_name: Dict[Tuple[str, str], str] = {}
        class_node_ids_by_name: Dict[str, List[str]] = {}
        function_node_id_by_file_and_name: Dict[Tuple[str, str], str] = {}
        function_node_ids_by_name: Dict[str, List[str]] = {}
        pending_extends_edges: List[Tuple[str, str, str]] = []
        pending_implements_edges: List[Tuple[str, str, str]] = []
        file_lines_map: Dict[str, List[str]] = {}
        symbol_body_by_node_id: Dict[str, str] = {}

        def add_node(node: Dict[str, Any]) -> None:
            node_id = str(node["id"])
            if node_id in seen_node_ids:
                return
            seen_node_ids.add(node_id)
            nodes.append(node)

        def add_edge(source: str, target: str, edge_type: str) -> None:
            key = (source, target, edge_type)
            if key in seen_edges:
                return
            seen_edges.add(key)
            edges.append({"source": source, "target": target, "type": edge_type})

        for file_path in selected_files:
            file_node_id = f"file:{file_path}"
            basename = os.path.basename(file_path)
            file_stem, _ = os.path.splitext(basename)
            add_node(
                {
                    "id": file_node_id,
                    "type": "file",
                    "name": file_path,
                    "aliases": generate_aliases(file_stem),
                    "file": file_path,
                }
            )
            abs_path = os.path.join(repo_dir, file_path)
            try:
                file_lines_map[file_path] = _read_text_file(abs_path)
            except Exception:
                file_lines_map[file_path] = []

        for clazz in classes:
            file_path = _normalize_file_path(clazz["file"])
            if file_path not in selected_file_set:
                continue
            class_node_id = f"class:{file_path}:{clazz['name']}"
            class_name = str(clazz.get("name") or "")
            parent_class_name = str(clazz.get("extends") or "").strip()
            add_node(
                {
                    "id": class_node_id,
                    "type": "class",
                    "name": class_name,
                    "aliases": generate_aliases(class_name),
                    "file": file_path,
                    "start_line": clazz.get("start_line"),
                    "end_line": clazz.get("end_line"),
                }
            )
            if class_name:
                class_node_id_by_file_and_name[(file_path, class_name)] = class_node_id
                class_node_ids_by_name.setdefault(class_name, []).append(class_node_id)
            if parent_class_name:
                pending_extends_edges.append((class_node_id, file_path, parent_class_name))
            for interface_name in clazz.get("implements", []):
                pending_implements_edges.append((class_node_id, file_path, interface_name))
            add_edge(f"file:{file_path}", class_node_id, "defines")
            symbol_body_by_node_id[class_node_id] = "\n".join(clazz.get("class_content", []))

            for method in clazz.get("methods", []):
                method_name = method.get("name", "")
                if not method_name:
                    continue
                method_node_id = f"function:{file_path}:{clazz['name']}.{method_name}"
                add_node(
                    {
                        "id": method_node_id,
                        "type": "function",
                        "name": f"{clazz['name']}.{method_name}",
                        "aliases": generate_aliases(method_name),
                        "file": file_path,
                        "start_line": method.get("start_line"),
                        "end_line": method.get("end_line"),
                    }
                )
                function_node_id_by_file_and_name[(file_path, f"{clazz['name']}.{method_name}")] = method_node_id
                function_node_ids_by_name.setdefault(method_name, []).append(method_node_id)
                add_edge(class_node_id, method_node_id, "defines")
                symbol_body_by_node_id[method_node_id] = "\n".join(method.get("method_content", []))

        # Add class inheritance edges after class nodes are indexed.
        for child_node_id, child_file_path, parent_class_name in pending_extends_edges:
            parent_node_id = class_node_id_by_file_and_name.get((child_file_path, parent_class_name))
            if not parent_node_id:
                candidates = sorted(set(class_node_ids_by_name.get(parent_class_name, [])))
                parent_node_id = candidates[0] if candidates else ""
            if parent_node_id:
                add_edge(child_node_id, parent_node_id, "extends")

        for child_node_id, child_file_path, interface_name in pending_implements_edges:
            target_node_id = class_node_id_by_file_and_name.get((child_file_path, interface_name))
            if not target_node_id:
                candidates = sorted(set(class_node_ids_by_name.get(interface_name, [])))
                target_node_id = candidates[0] if candidates else f"file:{child_file_path}"
            if target_node_id:
                add_edge(child_node_id, target_node_id, "implements")

        for function in functions:
            file_path = _normalize_file_path(function["file"])
            if file_path not in selected_file_set:
                continue
            function_name = function.get("name", "")
            if not function_name:
                continue
            function_node_id = f"function:{file_path}:{function_name}"
            add_node(
                {
                    "id": function_node_id,
                    "type": "function",
                    "name": function_name,
                    "aliases": generate_aliases(function_name),
                    "file": file_path,
                    "start_line": function.get("start_line"),
                    "end_line": function.get("end_line"),
                }
            )
            function_node_id_by_file_and_name[(file_path, function_name)] = function_node_id
            function_node_ids_by_name.setdefault(function_name, []).append(function_node_id)
            add_edge(f"file:{file_path}", function_node_id, "defines")
            symbol_body_by_node_id[function_node_id] = "\n".join(function.get("text", []))

        for source, target in import_edges:
            add_edge(f"file:{source}", f"file:{target}", "imports")

        for file_path in selected_files:
            lines = file_lines_map.get(file_path, [])
            exported_names = _extract_exported_names(lines)
            for export_name in exported_names:
                target_id = class_node_id_by_file_and_name.get((file_path, export_name))
                if not target_id:
                    target_id = function_node_id_by_file_and_name.get((file_path, export_name))
                if target_id:
                    add_edge(f"file:{file_path}", target_id, "exports")

            for line in lines:
                stripped = line.strip()
                if not stripped.startswith("export") or " from " not in stripped:
                    continue
                for module_name in CodeGraphBuilder._extract_import_targets(stripped):
                    target_file = CodeGraphBuilder._resolve_import_target(file_path, module_name, selected_file_set)
                    if target_file:
                        add_edge(f"file:{file_path}", f"file:{target_file}", "re_exports")

        symbol_candidates_by_file: Dict[str, Dict[str, str]] = {}
        for (file_path, class_name), node_id in class_node_id_by_file_and_name.items():
            symbol_candidates_by_file.setdefault(file_path, {})[class_name] = node_id
        for (file_path, function_name), node_id in function_node_id_by_file_and_name.items():
            short_name = function_name.split(".")[-1]
            symbol_candidates_by_file.setdefault(file_path, {}).setdefault(function_name, node_id)
            symbol_candidates_by_file[file_path].setdefault(short_name, node_id)

        global_symbol_index: Dict[str, List[str]] = {}
        for (_, class_name), node_id in class_node_id_by_file_and_name.items():
            global_symbol_index.setdefault(class_name, []).append(node_id)
        for (_, function_name), node_id in function_node_id_by_file_and_name.items():
            global_symbol_index.setdefault(function_name, []).append(node_id)
            global_symbol_index.setdefault(function_name.split(".")[-1], []).append(node_id)

        def resolve_symbol_target(current_file: str, raw_name: str) -> str:
            direct = symbol_candidates_by_file.get(current_file, {}).get(raw_name)
            if direct:
                return direct
            short = raw_name.split(".")[-1]
            direct = symbol_candidates_by_file.get(current_file, {}).get(short)
            if direct:
                return direct
            candidates = global_symbol_index.get(raw_name) or global_symbol_index.get(short) or []
            if len(candidates) == 1:
                return candidates[0]
            for candidate in candidates:
                if str(candidate).startswith(f"function:{current_file}:") or str(candidate).startswith(f"class:{current_file}:"):
                    return candidate
            return candidates[0] if candidates else ""

        def add_relation_to_symbol_or_file(source_node_id: str, source_file: str, target_name: str, edge_type: str) -> None:
            target_node_id = resolve_symbol_target(source_file, target_name)
            if target_node_id and target_node_id != source_node_id:
                add_edge(source_node_id, target_node_id, edge_type)
                return
            file_node_id = f"file:{source_file}"
            if file_node_id != source_node_id:
                add_edge(source_node_id, file_node_id, edge_type)

        for source_node_id, body_text in symbol_body_by_node_id.items():
            source_file = ""
            if source_node_id.startswith("class:") or source_node_id.startswith("function:"):
                parts = source_node_id.split(":", 2)
                if len(parts) >= 3:
                    source_file = parts[1]
            if not source_file:
                continue

            for target_name in _extract_call_targets(body_text):
                target_node_id = resolve_symbol_target(source_file, target_name)
                if target_node_id and target_node_id != source_node_id:
                    add_edge(source_node_id, target_node_id, "calls")

            for target_name in _extract_new_targets(body_text):
                target_node_id = resolve_symbol_target(source_file, target_name)
                if target_node_id and target_node_id != source_node_id:
                    add_edge(source_node_id, target_node_id, "instantiates")

            for target_name in _extract_jsx_component_targets(body_text):
                target_node_id = resolve_symbol_target(source_file, target_name)
                if target_node_id and target_node_id != source_node_id:
                    add_edge(source_node_id, target_node_id, "renders")

            for target_name in _extract_hook_calls(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "uses_hook")

            for target_name in _extract_type_uses(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "type_uses")

            for target_name in _extract_state_reads(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "reads_state")

            for target_name in _extract_state_writes(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "writes_state")

            for target_name in _extract_subscriptions(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "subscribes_to")

            for target_name in _extract_dispatches(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "dispatches")

            for target_name in _extract_context_providers(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "provides_context")

            for target_name in _extract_context_consumers(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "consumes_context")

            for target_name in _extract_route_targets(body_text):
                add_relation_to_symbol_or_file(source_node_id, source_file, target_name, "routes_to")

        for file_path in selected_files:
            lines = file_lines_map.get(file_path, [])
            for target_file in _extract_style_targets(file_path, lines, selected_file_set):
                add_edge(f"file:{file_path}", f"file:{target_file}", "uses_style")

            for module_name, _ in _extract_type_import_modules(lines):
                target_file = CodeGraphBuilder._resolve_import_target(file_path, module_name, selected_file_set)
                if target_file:
                    add_edge(f"file:{file_path}", f"file:{target_file}", "type_imports")

            file_text = "\n".join(lines)
            for target_name in _extract_type_uses(file_text):
                target_node_id = resolve_symbol_target(file_path, target_name)
                if target_node_id:
                    add_edge(f"file:{file_path}", target_node_id, "type_uses")

            for target_name in _extract_route_targets(file_text):
                target_node_id = resolve_symbol_target(file_path, target_name)
                if target_node_id:
                    add_edge(f"file:{file_path}", target_node_id, "routes_to")

        return nodes, edges
