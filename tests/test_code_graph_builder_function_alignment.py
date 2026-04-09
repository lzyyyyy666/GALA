import os

from src.code_graph_builder import CodeGraphBuilder


def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(content)


def test_build_candidate_function_pool_filters_ui_functions(tmp_path):
    repo_dir = tmp_path / "repo"
    _write_file(
        str(repo_dir / "src" / "SignupForm.tsx"),
        """
import React, { useState } from "react";

export function formatCurrency(value) {
    return String(value);
}

export function SignupForm(props) {
    const [email, setEmail] = useState("");
    const handleSubmit = () => setEmail("done");
    return <NoticeBanner onClick={handleSubmit} className="signup-form">{props.children}</NoticeBanner>;
}

export const useSignupNotice = () => {
    const [visible, setVisible] = useState(true);
    return { visible, setVisible };
};
""".strip(),
    )

    builder = CodeGraphBuilder()
    pool = builder._build_function_pool_from_seed_files(
        seed_files=["src/SignupForm.tsx"],
        repo_dir=str(repo_dir),
    )

    raw_names = {item["raw_name"] for item in pool}
    assert "SignupForm" in raw_names
    assert "handleSubmit" in raw_names
    assert "useSignupNotice" in raw_names
    assert "formatCurrency" not in raw_names

    names = {item["name"] for item in pool}
    assert "SignupForm.tsx/SignupForm" in names
    assert "SignupForm.tsx/handleSubmit" in names
    assert "SignupForm.tsx/useSignupNotice" in names


def test_alignment_agent_uses_image_graph_directly():
    image_graph = {
        "root_objects": [
            {"id": "notice_bar", "reason": "Verification notice stays visible while action remains enabled."}
        ],
        "nodes": [
            {
                "id": "notice_bar",
                "type": "component",
                "text": "Please verify your email",
                "role": "root",
                "attributes": {"role": "notification", "state": "visible"},
            },
            {
                "id": "create_store_button",
                "type": "component",
                "text": "Create your store",
                "attributes": {"role": "action", "state": "active"},
            },
        ],
        "edges": [
            {"source": "notice_bar", "target": "create_store_button", "type": "contains"},
        ],
    }
    ui_graph = {
        "nodes": [
            {
                "id": "function:src/NoticeBanner.tsx:NoticeBanner",
                "name": "NoticeBanner",
                "file": "src/NoticeBanner.tsx",
                "ui_score": 6,
                "signals": ["component", "reads_state", "event"],
                "summary": "NoticeBanner shows verification notice and continue button",
            }
        ],
        "edges": [],
    }

    from src.code_graph_builder import FunctionAlignmentAgent

    matches = FunctionAlignmentAgent._heuristic_anchor_matches(image_graph, ui_graph)
    hints = FunctionAlignmentAgent._simple_image_hints(
        image_graph,
        "Verification notice stays visible while continue action remains enabled.",
    )

    assert matches
    assert any(hint in hints for hint in ("state", "event"))


def test_build_first_order_module_graph_outputs_edit_targets_without_llm(tmp_path):
    repo_dir = tmp_path / "repo"
    _write_file(
        str(repo_dir / "src" / "NoticeBanner.tsx"),
        """
import React, { useState } from "react";

export function NoticeBanner() {
    const [visible, setVisible] = useState(true);
    const handleContinue = () => setVisible(false);
    return <button className="notice-banner" onClick={handleContinue}>Continue</button>;
}
""".strip(),
    )

    builder = CodeGraphBuilder()
    candidate_pool, ui_graph, anchor_alignment, edit_targets, fallback_reason = (
        builder.build_first_order_module_graph(
            normalized_seed_files=["src/NoticeBanner.tsx"],
            problem_statement="The verification notice should block continuing until the email is confirmed.",
            repo_dir=str(repo_dir),
            available_files=["src/NoticeBanner.tsx"],
            forward_neighbors={},
            image_graph={
                "root_objects": [
                    {"id": "notice_bar", "reason": "The notice remains visible while the button can still be used."},
                    {"id": "continue_button", "reason": "The action should be blocked until verification."},
                ],
                "nodes": [
                    {"id": "notice_bar", "type": "component", "text": "Verify email", "role": "root"},
                    {"id": "continue_button", "type": "component", "text": "Continue", "role": "root"},
                ],
                "edges": [
                    {"source": "notice_bar", "target": "continue_button", "type": "contains"},
                ],
            },
        )
    )

    assert candidate_pool
    assert ui_graph["nodes"]
    assert anchor_alignment["anchor_matches"]
    assert edit_targets
    assert len([item for item in edit_targets if item["role"] == "primary"]) <= 2
    assert len(edit_targets) <= 5
    assert fallback_reason == "missing_llm_config"
