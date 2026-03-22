from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.backend.utils.io import read_json


OntologyTree = Dict[str, Any]


def default_ontology_root(project_root: Path, use_test_ontology: bool) -> Path:
    mode = "test" if use_test_ontology else "production"
    return project_root / "src" / "backend" / "ontology" / "separate" / mode


def load_ontology_triplet(ontology_root: str | Path) -> Dict[str, OntologyTree]:
    root = Path(ontology_root)
    return {
        "PROFILE": read_json(root / "PROFILE" / "profile.json"),
        "OPINION": read_json(root / "OPINION" / "opinion.json"),
        "ATTACK": read_json(root / "ATTACK" / "attack.json"),
    }


def iter_leaf_paths(tree: OntologyTree, prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    leaves: List[Tuple[str, ...]] = []
    for node, child in tree.items():
        path = prefix + (node,)
        if isinstance(child, dict) and child:
            leaves.extend(iter_leaf_paths(child, path))
        else:
            leaves.append(path)
    return leaves


def flatten_leaf_paths(tree: OntologyTree) -> List[str]:
    return [" > ".join(path) for path in iter_leaf_paths(tree)]


def leaf_to_key(path: str) -> str:
    return path.lower().replace(" ", "").replace(">", "_").replace("-", "_")


def find_primary_node(path: str) -> str:
    parts = [x.strip() for x in path.split(">")]
    if len(parts) >= 2:
        return parts[1]
    return parts[0]
