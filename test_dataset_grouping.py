import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def extract_video_id(sample_dir: str) -> str:
    sample_dir = sample_dir.replace("\\", "/")
    parts = Path(sample_dir).parts
    basename = parts[-1]

    marker = "_frame_"
    idx = basename.rfind(marker)
    if idx != -1:
        clip_id = basename[:idx]
        prefix = "/".join(parts[:-1])
        return f"{prefix}/{clip_id}" if prefix else clip_id

    if basename.startswith("frame_") and len(parts) > 1:
        return "/".join(parts[:-1])

    return sample_dir


def sample_frames(paths, num_frames):
    n = len(paths)
    if n == 0:
        return []
    if n >= num_frames:
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
    else:
        indices = np.arange(n)
    return [paths[i] for i in indices]


def expected_label_from_dir(sample_dir, flip_labels):
    top = sample_dir.replace("\\", "/").split("/", 1)[0]
    if top == "real":
        return 1 if flip_labels else 0
    if top == "fake":
        return 0 if flip_labels else 1
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="E:/Work/sampled_30k/manifest_onct.csv")
    parser.add_argument("--root_dir", default="E:/Work/sampled_30k")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--flip_labels", action="store_true",
                        help="Use for CDF manifests where real=1 and fake=0 in the CSV.")
    parser.add_argument("--show_videos", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root_dir)
    df = pd.read_csv(args.manifest, sep=None, engine="python")
    required = {"sample_dir", "label"}
    if not required.issubset(df.columns):
        raise AssertionError(f"Manifest must contain {required}. Found {list(df.columns)}")

    df["sample_dir"] = df["sample_dir"].astype(str).str.replace("\\", "/", regex=False)
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["sample_dir"].apply(extract_video_id)
    df["top_dir"] = df["sample_dir"].str.split("/", n=1).str[0]
    df["expected_label"] = df["sample_dir"].apply(
        lambda x: expected_label_from_dir(x, args.flip_labels)
    )

    label_mismatches = df[df["label"] != df["expected_label"]]
    missing = [
        rel for rel in df["sample_dir"]
        if not (root / rel / "image.png").is_file()
    ]

    mixed_label_videos = []
    video_frame_counts = {}
    selected_examples = []

    for video_id, group in df.groupby("video_id", sort=True):
        labels = sorted(group["label"].unique().tolist())
        if len(labels) != 1:
            mixed_label_videos.append((video_id, labels))

        paths = sorted(group["sample_dir"].tolist())
        video_frame_counts[video_id] = len(paths)
        if len(selected_examples) < args.show_videos:
            selected_examples.append((video_id, labels[0], paths, sample_frames(paths, args.num_frames)))

    frame_count_hist = Counter(video_frame_counts.values())

    print(f"manifest: {args.manifest}")
    print(f"root_dir: {root}")
    print(f"rows: {len(df)}")
    print(f"videos: {df['video_id'].nunique()}")
    print(f"top_dir counts: {df['top_dir'].value_counts().to_dict()}")
    print(f"label counts: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"frame-count histogram: {dict(sorted(frame_count_hist.items()))}")
    print(f"missing image.png files: {len(missing)}")
    print(f"label/directory mismatches: {len(label_mismatches)}")
    print(f"mixed-label videos: {len(mixed_label_videos)}")

    for video_id, label, paths, selected in selected_examples:
        print()
        print(f"video_id={video_id} label={label} available_frames={len(paths)} selected={len(selected)}")
        print("first_available:", paths[:5])
        print("selected_frames:", selected)

    if missing:
        print("first missing:", missing[:10])
    if len(label_mismatches):
        print(label_mismatches[["sample_dir", "label", "expected_label"]].head(10).to_string(index=False))
    if mixed_label_videos:
        print("first mixed-label videos:", mixed_label_videos[:10])

    assert not missing, "Some manifest rows do not have image.png on disk."
    assert len(label_mismatches) == 0, "Some labels do not match their top-level directory."
    assert len(mixed_label_videos) == 0, "Some grouped videos contain multiple labels."
    for _, group in df.groupby("video_id"):
        paths = sorted(group["sample_dir"].tolist())
        selected = sample_frames(paths, args.num_frames)
        assert len(selected) == min(len(paths), args.num_frames), "Sampler repeated or dropped frames unexpectedly."
        assert len(selected) == len(set(selected)), "Sampler returned duplicate frames."


if __name__ == "__main__":
    main()
