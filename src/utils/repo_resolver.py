import os


def resolve_repo_dir(repo_path: str, repo_identifier: str) -> str:
    """
    Resolve repository directory from a repo identifier like "owner/name".

    Prefer owner directory first to match local dataset layout, and fall back
    to repo name for compatibility with other layouts.
    """
    parts = repo_identifier.split("/", 1)
    owner = parts[0]
    name = parts[1] if len(parts) > 1 else parts[0]

    owner_dir = os.path.join(repo_path, owner)
    if os.path.isdir(owner_dir):
        return owner_dir

    name_dir = os.path.join(repo_path, name)
    if os.path.isdir(name_dir):
        return name_dir

    return owner_dir
