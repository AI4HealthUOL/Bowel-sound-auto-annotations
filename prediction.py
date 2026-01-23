from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
import librosa
from transformers.modeling_outputs import SequenceClassifierOutput

from pipeline.config import Settings
from pipeline.models import get_device


def pad_or_trim(seg: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim to a fixed number of samples."""
    if len(seg) < target_len:
        out = np.zeros(target_len, dtype=seg.dtype)
        out[: len(seg)] = seg
        return out
    if len(seg) > target_len:
        return seg[:target_len]
    return seg


def predict_segments_for_file(
    wav_path: str,
    segments: List[Tuple[float, float]],
    feature_extractor,
    model: torch.nn.Module,
    settings: Settings,
) -> List[int]:
    """
    Predict a class label for each segment.

    Compatibility:
    - Works with a single AST model or an ensemble.
    - Accepts outputs that behave like SequenceClassifierOutput (has `.logits`),
      or dict-like outputs containing "logits".
    """
    device = get_device()

    y, _ = librosa.load(wav_path, sr=settings.target_sr, mono=True)
    preds: List[int] = []

    seg_len_samples = int(settings.seg_fixed_len_sec * settings.target_sr) if settings.seg_fixed_len_sec else None

    with torch.no_grad():
        for (start, end) in segments:
            s = int(start * settings.target_sr)
            e = int(end * settings.target_sr)
            seg = y[s:e]

            if seg_len_samples is not None:
                seg = pad_or_trim(seg, seg_len_samples)

            inputs = feature_extractor(seg, sampling_rate=settings.target_sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Normalize outputs to a logits tensor
            if isinstance(outputs, SequenceClassifierOutput):
                logits = outputs.logits
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs  # last resort

            pred_id = int(torch.argmax(logits, dim=-1).item())
            preds.append(pred_id)

    return preds


def fill_time_gaps(
    segments,
    labels,
    *,
    gap_label=0,
    include_edges=False,
    total_duration=None,
    decimals=None,
    min_gap=0.0,
):
    """
    Insert explicit "gap" segments between predicted segments.

    Why this is useful:
    - Some plots/metrics assume the whole timeline is covered.
    - Makes it obvious where the model predicted "nothing" (class 0, usually Silent).
    """
    assert len(segments) == len(labels), "segments and labels length mismatch"
    if not segments:
        if include_edges and total_duration and total_duration > 0:
            end = round(float(total_duration), decimals) if decimals is not None else float(total_duration)
            return [(0.0, end)], [gap_label]
        return [], []

    order = sorted(range(len(segments)), key=lambda i: (float(segments[i][0]), float(segments[i][1])))
    segs_sorted = [(float(segments[i][0]), float(segments[i][1])) for i in order]
    labels_sorted = [labels[i] for i in order]

    r = (lambda x: round(x, decimals)) if decimals is not None else (lambda x: x)

    new_segs, new_labs = [], []

    st0, en0 = r(segs_sorted[0][0]), r(segs_sorted[0][1])
    if include_edges and st0 > 0 and (st0 - 0.0) >= min_gap:
        new_segs.append((0.0, st0))
        new_labs.append(gap_label)

    new_segs.append((st0, en0))
    new_labs.append(labels_sorted[0])
    prev_end = en0

    for (st, en), lb in zip(segs_sorted[1:], labels_sorted[1:]):
        st_r, en_r = r(st), r(en)
        if st_r > prev_end and (st_r - prev_end) >= min_gap:
            new_segs.append((prev_end, st_r))
            new_labs.append(gap_label)
        new_segs.append((st_r, en_r))
        new_labs.append(lb)
        prev_end = max(prev_end, en_r)

    if include_edges and total_duration is not None:
        tail_end = r(float(total_duration))
        if tail_end > prev_end and (tail_end - prev_end) >= min_gap:
            new_segs.append((prev_end, tail_end))
            new_labs.append(gap_label)

    return new_segs, new_labs


def merge_same_label_segments(
    segments: List[Tuple[float, float]],
    labels: List[int],
    max_gap: float,
    min_dur: float,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Merge adjacent segments if they have the same label and the gap is small.

    This reduces “label jitter” and produces cleaner timelines.
    """
    if not segments:
        return [], []

    merged = []
    merged_labels = []

    cur_start, cur_end = segments[0]
    cur_label = labels[0]

    for (st, en), lb in zip(segments[1:], labels[1:]):
        if lb == cur_label and st - cur_end <= max_gap + 1e-9:
            cur_end = max(cur_end, en)
        else:
            if (cur_end - cur_start) >= min_dur:
                merged.append((cur_start, cur_end))
                merged_labels.append(cur_label)
            cur_start, cur_end, cur_label = st, en, lb

    if (cur_end - cur_start) >= min_dur:
        merged.append((cur_start, cur_end))
        merged_labels.append(cur_label)

    return merged, merged_labels
