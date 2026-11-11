"""Download a pinned ModelScope snapshot and verify every downloaded file."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests


MODELS = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}
DATA_ROOT = Path("/host/mnt/nas/urban-visual-gene/streetspatial_v0")


def manifest(model_id, revision, session):
    url = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files"
    response = session.get(url, params={"Revision": revision, "Recursive": "true"}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if data.get("Code") != 200:
        raise RuntimeError(data)
    return [row for row in data["Data"]["Files"] if row["Type"] == "blob"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(model_id, revision, target, session, include_duplicate_bin=False):
    files = manifest(model_id, revision, session)
    if not include_duplicate_bin:
        files = [row for row in files if row["Path"] != "pytorch_model.bin"]
    target.mkdir(parents=True, exist_ok=True)
    metadata = []
    for row in files:
        relative = Path(row["Path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe model path: {relative}")
        destination = target / relative
        expected = row.get("Sha256")
        if destination.exists() and destination.stat().st_size == row["Size"] and (not expected or sha256(destination) == expected):
            print(f"Verified cached: {relative}", flush=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://modelscope.cn/models/{model_id}/resolve/{revision}/{quote(row['Path'])}"
            partial = destination.with_name(destination.name + ".part")
            downloaded = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            print(f"Downloading: {relative} ({row['Size'] / 1e9:.2f} GB)", flush=True)
            with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if downloaded and response.status_code != 206:
                    downloaded = 0
                mode = "ab" if downloaded else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size != row["Size"]:
                raise ValueError(f"Size mismatch for {relative}: {partial.stat().st_size} != {row['Size']}")
            if expected and sha256(partial) != expected:
                raise ValueError(f"SHA256 mismatch for {relative}")
            partial.replace(destination)
        metadata.append({"path": row["Path"], "size": row["Size"], "sha256": expected})
    (target / "snapshot_manifest.json").write_text(
        json.dumps({"model_id": model_id, "revision": revision, "files": metadata}, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    target = args.target or DATA_ROOT / "models" / args.model
    with requests.Session() as session:
        download(MODELS[args.model], args.revision, target, session)
    print(f"Model verified at {target}", flush=True)


if __name__ == "__main__":
    main()
