"""Download and inventory the Columbia video and LR-ASD label archive.

Sources match the official LR-ASD Columbia_test.py implementation. Existing
non-empty outputs are retained, so interrupted setup can be resumed safely.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

VIDEO_URL = "https://www.youtube.com/watch?v=6GzxbrO0DHM"
LABEL_FILE_ID = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
LABEL_URL = (
    "https://drive.usercontent.google.com/download"
    f"?id={LABEL_FILE_ID}&export=download&confirm=t"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive, destination):
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    archive.extractall(destination)


def download_labels(output_dir):
    labels = output_dir / "col_labels"
    expected = labels / "fusion" / "abbas.txt"
    if expected.is_file() and list(labels.glob("tracks_*/*.txt")):
        print(f"Labels already present: {labels}")
        return labels, None
    archive_path = output_dir / "col_labels.tar.gz.part"
    print(f"Downloading LR-ASD Columbia labels to {archive_path}...")
    urllib.request.urlretrieve(LABEL_URL, archive_path)
    if archive_path.stat().st_size == 0:
        raise RuntimeError("Downloaded label archive is empty")
    archive_hash = sha256(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, output_dir)
    archive_path.unlink()
    if not expected.is_file() or not list(labels.glob("tracks_*/*.txt")):
        raise RuntimeError("Extracted label archive is incomplete")
    return labels, archive_hash


def download_video(output_dir):
    video = output_dir / "col.mp4"
    if video.is_file() and video.stat().st_size > 0:
        print(f"Video already present: {video}")
        return video
    if shutil.which("yt-dlp"):
        command = ["yt-dlp"]
    else:
        command = [sys.executable, "-m", "yt_dlp"]
    command += [
        "--no-playlist",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(video),
        VIDEO_URL,
    ]
    print("Downloading the public Columbia video with yt-dlp...")
    subprocess.run(command, check=True)
    if not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError(f"Video download did not create {video}")
    return video


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=root / "columbia" / "colDataPath")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels, downloaded_label_hash = download_labels(args.output_dir)
    video = download_video(args.output_dir)
    fusion_files = sorted(labels.glob("fusion/*.txt"))
    track_files = sorted(labels.glob("tracks_*/*.txt"))
    speakers = sorted(path.stem for path in fusion_files)
    report = {
        "video_source": VIDEO_URL,
        "label_source": "LR-ASD Google Drive label archive",
        "label_google_drive_file_id": LABEL_FILE_ID,
        "video_path": str(video.resolve()),
        "video_bytes": video.stat().st_size,
        "video_sha256": sha256(video),
        "labels_path": str(labels.resolve()),
        "label_archive_sha256_when_downloaded": downloaded_label_hash,
        "fusion_label_files": len(fusion_files),
        "track_label_files": len(track_files),
        "fusion_speakers": speakers,
        "passed": len(fusion_files) == 6 and bool(track_files),
    }
    manifest = args.output_dir / "download_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Columbia download audit failed")


if __name__ == "__main__":
    main()
