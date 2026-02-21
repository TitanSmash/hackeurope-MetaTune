from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from simulations.io_utils import ensure_dir, slugify_dataset_id, write_json


def normalize_dataset_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, str):
        return {"id": entry}
    if isinstance(entry, dict) and "id" in entry:
        return dict(entry)
    raise ValueError(f"Dataset entry must be a string or dict with 'id': {entry!r}")


def dataset_cache_key(dataset_entry: Dict[str, Any]) -> str:
    dataset_id = dataset_entry["id"]
    config_name = dataset_entry.get("config")
    if config_name:
        return f"{dataset_id}:{config_name}"
    return dataset_id


def _load_hf_dataset(dataset_entry: Dict[str, Any], hf_cache_dir: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError(
            "Missing `datasets` package. Install with `pip install datasets`."
        ) from exc

    dataset_id = dataset_entry["id"]
    config_name = dataset_entry.get("config")
    revision = dataset_entry.get("revision")
    trust_remote_code = bool(dataset_entry.get("trust_remote_code", False))
    kwargs: Dict[str, Any] = {
        "cache_dir": str(hf_cache_dir),
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision

    if config_name:
        return load_dataset(dataset_id, config_name, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def _is_saved_dataset_dir(path: Path) -> bool:
    # `save_to_disk` for DatasetDict writes dataset_dict.json.
    # `save_to_disk` for Dataset writes state.json.
    if not path.exists() or not path.is_dir():
        return False
    return (path / "dataset_dict.json").exists() or (path / "state.json").exists()


def _resolve_raw_destination(raw_root: Path, slug: str, force: bool) -> Path:
    base = raw_root / slug
    if force:
        return base
    if not base.exists() or _is_saved_dataset_dir(base):
        return base

    # Base path exists but is not a `save_to_disk` directory (for example HF cache layout).
    # Avoid mutating that directory and write into a dedicated sibling path.
    index = 1
    while True:
        candidate = raw_root / f"{slug}__saved{index}"
        if not candidate.exists() or _is_saved_dataset_dir(candidate):
            return candidate
        index += 1


def download_one_dataset(
    dataset_entry: Dict[str, Any],
    raw_root: Path,
    hf_cache_dir: Path,
    force: bool = False,
) -> Dict[str, Any]:
    key = dataset_cache_key(dataset_entry)
    slug = slugify_dataset_id(key)
    destination = _resolve_raw_destination(raw_root=raw_root, slug=slug, force=force)
    ensure_dir(raw_root)
    ensure_dir(hf_cache_dir)

    if destination.exists() and not force and _is_saved_dataset_dir(destination):
        metadata = {
            "dataset_id": dataset_entry["id"],
            "dataset_config": dataset_entry.get("config"),
            "dataset_key": key,
            "raw_path": str(destination),
            "downloaded": False,
        }
        return metadata

    dataset = _load_hf_dataset(dataset_entry, hf_cache_dir=hf_cache_dir)
    if destination.exists() and _is_saved_dataset_dir(destination):
        # Keep deterministic behavior if force is true.
        import shutil

        shutil.rmtree(destination)
    dataset.save_to_disk(str(destination))

    split_names = []
    if hasattr(dataset, "keys"):
        split_names = [str(split_name) for split_name in dataset.keys()]
    metadata = {
        "dataset_id": dataset_entry["id"],
        "dataset_config": dataset_entry.get("config"),
        "dataset_key": key,
        "raw_path": str(destination),
        "split_names": split_names,
        "downloaded": True,
    }
    write_json(destination / "download_metadata.json", metadata)
    return metadata


def download_datasets(
    dataset_entries: Iterable[Any],
    raw_root: Path | str,
    hf_cache_dir: Path | str,
    force: bool = False,
) -> List[Dict[str, Any]]:
    raw_root_path = Path(raw_root)
    hf_cache_path = Path(hf_cache_dir)
    normalized_entries = [normalize_dataset_entry(entry) for entry in dataset_entries]
    results = []
    for entry in normalized_entries:
        result = download_one_dataset(
            dataset_entry=entry,
            raw_root=raw_root_path,
            hf_cache_dir=hf_cache_path,
            force=force,
        )
        results.append(result)
    return results
