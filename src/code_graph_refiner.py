from typing import List, Tuple


def _normalize_file_path(file_path: str) -> str:
    return str(file_path or "").replace("\\", "/").lstrip("./")


def run_single_stage_refinement(
    refine_files: List[str],
) -> Tuple[str, List[str]]:
    normalized_files: List[str] = []
    for file_path in refine_files:
        normalized = _normalize_file_path(file_path)
        if normalized and normalized not in normalized_files:
            normalized_files.append(normalized)

    if not normalized_files:
        return "empty_refine_file_set", []

    return "", normalized_files
