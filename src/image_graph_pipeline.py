from typing import Any, Dict, List

from prompt.image_graph_prompt import IMAGE_TYPE_CLASSIFICATION_PROMPT
from prompt.rooted_image_graph_prompt import ROOTED_EXTRACTION_SYSTEM_PROMPT, build_rooted_extraction_prompt
from src.image_graph_normalizer import normalize_visual_ir
from src.image_graph_parser import parse_visual_extraction_response, parse_image_type
from src.image_graph_schema import graph_type_from_image_type
from src.utils.llm_client import send_chat_completion


class TypeAwareImageGraphPipeline:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    @staticmethod
    def _extract_model_content(response_data: Dict[str, Any]) -> str:
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                elif isinstance(chunk, str):
                    chunks.append(chunk)
            return "\n".join(chunks)
        return ""

    def _classify_image_type(self, image_url: str, issue_text: str) -> str:
        response = send_chat_completion(
            api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            system_prompt=IMAGE_TYPE_CLASSIFICATION_PROMPT["system_prompt"],
            user_prompt=IMAGE_TYPE_CLASSIFICATION_PROMPT["user_prompt_template"].format(issue_text=issue_text or ""),
            image_url=image_url,
            temperature=0.0,
        )
        raw = self._extract_model_content(response)
        return parse_image_type(raw)

    def _extract_visual_structure(
        self,
        image_url: str,
        issue_text: str,
        image_type: str,
    ) -> Dict[str, Any]:
        prompt = build_rooted_extraction_prompt(
            image_type=image_type,
            issue_text=issue_text or "",
        )
        response = send_chat_completion(
            api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            system_prompt=ROOTED_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            image_url=image_url,
            temperature=0.0,
        )
        raw = self._extract_model_content(response)
        parsed_ir = parse_visual_extraction_response(response_text=raw, image_type=image_type)
        return normalize_visual_ir(parsed_ir)

    def run_single(self, image_url: str, issue_text: str, image_path: str = "") -> Dict[str, Any]:
        image_type = self._classify_image_type(image_url=image_url, issue_text=issue_text)
        normalized_ir = self._extract_visual_structure(
            image_url=image_url,
            issue_text=issue_text,
            image_type=image_type,
        )
        final_image_type = str(normalized_ir.get("image_type") or image_type)

        return {
            "image_path": image_path,
            "image_type": final_image_type,
            "graph_type": graph_type_from_image_type(final_image_type),
            "root_objects": normalized_ir.get("root_objects", []),
            "nodes": normalized_ir.get("nodes", []),
            "edges": normalized_ir.get("edges", []),
        }
