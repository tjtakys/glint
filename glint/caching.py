"""Fit結果のfingerprint付きcache。

設定と入力データのmetadataからSHA-256のfingerprintを作り、一致すれば保存済みresultを返して再fitを省略する。
staleの場合は何が変わったかを表示する。
設定は同じままコードを変更した時などはmetadataへcache_versionを入れて更新する
"""

from __future__ import annotations
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    # np.float64などをJSON化できる素の値へ。その他はstrへfallbackする。
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def fingerprint(metadata: Mapping) -> str:
    serialized = json.dumps(metadata, sort_keys=True, default=_canonical)
    return hashlib.sha256(serialized.encode()).hexdigest()


def flatten_metadata(metadata, prefix: str = "") -> dict:
    """入れ子dictを'key.subkey'形式へ平坦化する（stale差分表示用）。"""
    flat = {}
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            flat.update(flatten_metadata(value, f"{prefix}{key}."))
    else:
        flat[prefix[:-1]] = metadata
    return flat


def metadata_diff(cached: Mapping, current: Mapping) -> list[str]:
    """cache時と現在のmetadataの差分を'key: old -> new'の行リストで返す。"""
    old = flatten_metadata(cached)
    new = flatten_metadata(current)
    lines = []
    for key in sorted(set(old) | set(new)):
        old_value = old.get(key, "<missing>")
        new_value = new.get(key, "<missing>")
        if str(old_value) != str(new_value):
            lines.append(f"{key}: {old_value!r} -> {new_value!r}")
    return lines


def load_cached_result(path, metadata: Mapping, *, label: Optional[str] = None) -> Any:
    """fingerprintが一致すればcacheのresultを返し、そうでなければNoneを返す。

    staleの場合は差分を表示する（大量になる場合は先頭のみ）。
    """
    path = Path(path)
    name = label if label is not None else path.stem
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        print(f"{name}: unreadable cache ignored")
        return None
    cached_metadata = payload.get("metadata")
    if cached_metadata is not None and fingerprint(cached_metadata) == fingerprint(metadata):
        return payload["result"]
    print(f"{name}: stale cache ignored")
    if cached_metadata is not None:
        for line in metadata_diff(cached_metadata, metadata)[:10]:
            print(f"    {line}")
    return None


def save_cached_result(path, metadata: Mapping, result: Any) -> None:
    """resultをmetadata・fingerprintと共にatomicに保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {
        "fingerprint": fingerprint(metadata),
        "metadata": metadata,
        "result": result,
    }
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
