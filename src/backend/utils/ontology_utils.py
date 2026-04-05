from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.backend.utils.io import read_json


OntologyTree = Dict[str, Any]

# Keys that are metadata annotations, not subtree nodes.
_METADATA_KEYS = frozenset({"adversarial_direction", "description", "notes", "examples"})


def _is_metadata_key(key: str) -> bool:
    """Return True for keys that carry leaf-level metadata, not subtree structure."""
    return key.startswith("_") or key in _METADATA_KEYS or not key[0].isupper()


def _is_leaf_node(child: Any) -> bool:
    """Return True if child represents a leaf (empty dict, non-dict, or metadata-only dict)."""
    if not isinstance(child, dict):
        return True
    if not child:
        return True
    # A dict is a leaf if ALL its keys are metadata keys (no uppercase-starting subtree keys)
    return all(_is_metadata_key(k) for k in child)


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
        if _is_metadata_key(node):
            continue  # skip _metadata blocks and other annotation keys
        path = prefix + (node,)
        if _is_leaf_node(child):
            leaves.append(path)
        else:
            leaves.extend(iter_leaf_paths(child, path))
    return leaves


def flatten_leaf_paths(tree: OntologyTree) -> List[str]:
    return [" > ".join(path) for path in iter_leaf_paths(tree)]


def get_leaf_metadata(tree: OntologyTree, leaf_path: str) -> Dict[str, Any]:
    """Return the metadata dict stored at a given leaf path (e.g. adversarial_direction)."""
    parts = [p.strip() for p in leaf_path.split(">")]
    node: Any = tree
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    if isinstance(node, dict):
        return {k: v for k, v in node.items() if _is_metadata_key(k)}
    return {}


def load_adversarial_directions_from_opinion(
    opinion_tree: OntologyTree,
) -> Tuple[Dict[str, int], str]:
    """Extract adversarial direction mappings from an opinion ontology tree.

    Returns:
        directions: dict mapping leaf_name -> adversarial_direction (1, -1, or 0).
            Only non-zero directions are returned (0 = neutral, excluded from scoring).
        goal: the adversarial_operator_goal string if present, else ''.
    """
    meta = opinion_tree.get("_metadata", {})
    goal: str = meta.get("adversarial_operator_goal", "")

    leaf_paths = flatten_leaf_paths(opinion_tree)
    directions: Dict[str, int] = {}
    for leaf_path in leaf_paths:
        leaf_meta = get_leaf_metadata(opinion_tree, leaf_path)
        direction = leaf_meta.get("adversarial_direction")
        if direction is not None:
            d = int(direction)
            if d != 0:
                leaf_name = leaf_path.split(">")[-1].strip()
                directions[leaf_name] = d

    return directions, goal


def leaf_to_key(path: str) -> str:
    return path.lower().replace(" ", "").replace(">", "_").replace("-", "_")


def find_primary_node(path: str) -> str:
    parts = [x.strip() for x in path.split(">")]
    if len(parts) >= 2:
        return parts[1]
    return parts[0]
