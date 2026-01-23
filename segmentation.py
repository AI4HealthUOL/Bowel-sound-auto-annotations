from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import librosa
from scipy.ndimage import gaussian_filter1d
from transformers import AutoProcessor, AutoModelForAudioClassification

from pipeline.config import Settings
from pipeline.filtering import apply_pre_filter
from pipeline.models import get_device


# Lazily loaded models to avoid paying startup cost unless needed
_sed_processor = None
_sed_model = None

_atst_processor = None
_atst_model = None

_yamnet_model = None
_silero = None


def _activity_to_segments(
    activity: np.ndarray,
    total_duration_sec: float,
    threshold: float,
    min_seg_dur: float,
) -> List[Tuple[float, float]]:
    """
    Convert a 1D activity curve (per-frame probability) to (start, end) segments.
    Frames are assumed uniformly spaced over [0, total_duration_sec].
    """
    if activity.ndim != 1 or len(activity) == 0:
        return []

    mask = activity > threshold
    n_frames = len(activity)
    frame_dur = total_duration_sec / float(n_frames)

    segments: List[Tuple[float, float]] = []
    in_seg = False
    start_idx = 0

    for i, active in enumerate(mask):
        if active and not in_seg:
            in_seg = True
            start_idx = i
        elif (not active) and in_seg:
            in_seg = False
            end_idx = i
            start_t = start_idx * frame_dur
            end_t = end_idx * frame_dur
            if end_t - start_t >= min_seg_dur:
                segments.append((start_t, end_t))

    if in_seg:
        end_idx = n_frames
        start_t = start_idx * frame_dur
        end_t = end_idx * frame_dur
        if end_t - start_t >= min_seg_dur:
            segments.append((start_t, end_t))

    return segments


def detect_energy_segments(y: np.ndarray, sr: int, settings: Settings) -> List[Tuple[float, float]]:
    """
    Simple energy-based segmentation:
      - STFT magnitude -> dB
      - Average over frequency bins -> 1D energy curve
      - Smooth with a small Gaussian
      - Threshold at median
      - Convert above-threshold runs into segments
    """
    S = np.abs(librosa.stft(y, n_fft=settings.n_fft, hop_length=settings.hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    energy_profile = np.mean(S_db, axis=0)
    smoothed = gaussian_filter1d(energy_profile, sigma=2)

    threshold = np.percentile(smoothed, 50)
    activity_mask = smoothed > threshold

    segments: List[Tuple[float, float]] = []
    in_segment = False
    start_frame = 0

    for i, active in enumerate(activity_mask):
        if active and not in_segment:
            start_frame = i
            in_segment = True
        elif (not active) and in_segment:
            end_frame = i
            in_segment = False

            start_sec = start_frame * settings.hop_length / sr
            end_sec = end_frame * settings.hop_length / sr
            if end_sec - start_sec >= settings.min_seg_duration_sec:
                segments.append((start_sec, end_sec))

    if in_segment:
        end_frame = len(activity_mask)
        start_sec = start_frame * settings.hop_length / sr
        end_sec = end_frame * settings.hop_length / sr
        if end_sec - start_sec >= settings.min_seg_duration_sec:
            segments.append((start_sec, end_sec))

    return segments


def detect_energy_mod_segments(y: np.ndarray, sr: int, settings: Settings) -> List[Tuple[float, float]]:
    """
    energy_mod segmentation:
    - Derives an energy curve from STFT
    - Uses dE_frame (frame-to-frame change) and dE_base (above-baseline energy)
    - Detects segments via a hangover-based state machine
    """
    S = np.abs(librosa.stft(y, n_fft=settings.n_fft, hop_length=settings.hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    energy_profile = np.mean(S_db, axis=0)
    smoothed = gaussian_filter1d(energy_profile, sigma=2.0)

    dE_frame = np.diff(smoothed, prepend=smoothed[0])
    baseline = np.percentile(smoothed, 30.0)
    dE_base = smoothed - baseline

    thr_df = np.percentile(np.abs(dE_frame), 80.0)
    thr_db = np.percentile(dE_base, 40.0)

    hop_time = settings.hop_length / float(sr)
    hangover_sec = 0.1
    hang_frames = max(1, int(hangover_sec / hop_time))

    state = 0
    hang = 0
    start_idx = None
    segments_idx: List[Tuple[int, int]] = []

    for i in range(len(smoothed)):
        high_df = abs(dE_frame[i]) >= thr_df
        high_db = dE_base[i] >= thr_db

        if state == 0:
            if high_df and high_db:
                state = 1
                start_idx = i
                hang = hang_frames
        else:
            if high_df or high_db:
                hang = hang_frames
            else:
                hang -= 1
                if hang <= 0:
                    end_idx = i
                    dur_sec = (end_idx - start_idx) * hop_time
                    if dur_sec >= settings.min_seg_duration_sec:
                        segments_idx.append((start_idx, end_idx))
                    state = 0
                    start_idx = None

    if state == 1 and start_idx is not None:
        end_idx = len(smoothed) - 1
        dur_sec = (end_idx - start_idx) * hop_time
        if dur_sec >= settings.min_seg_duration_sec:
            segments_idx.append((start_idx, end_idx))

    segments: List[Tuple[float, float]] = []
    for s_idx, e_idx in segments_idx:
        start_sec = s_idx * settings.hop_length / float(sr)
        end_sec = e_idx * settings.hop_length / float(sr)
        if end_sec - start_sec >= settings.min_seg_duration_sec:
            segments.append((start_sec, end_sec))

    return segments


def detect_rms_mod_segments(y: np.ndarray, sr: int, settings: Settings) -> List[Tuple[float, float]]:
    """
    RMS-based segmentation:
    - Computes frame-wise RMS
    - Smooths RMS
    - Uses dR_frame and dR_base thresholds
    - Detects segments via hangover state machine
    """
    rms = librosa.feature.rms(
        y=y,
        frame_length=settings.n_fft,
        hop_length=settings.hop_length,
        center=True,
    )[0]
    smoothed = gaussian_filter1d(rms, sigma=2.0)

    dR_frame = np.diff(smoothed, prepend=smoothed[0])
    baseline = np.percentile(smoothed, 30.0)
    dR_base = smoothed - baseline

    thr_df = np.percentile(np.abs(dR_frame), 80.0)
    thr_db = np.percentile(dR_base, 40.0)

    hop_time = settings.hop_length / float(sr)
    hangover_sec = 0.1
    hang_frames = max(1, int(hangover_sec / hop_time))

    state = 0
    hang = 0
    start_idx = None
    segments_idx: List[Tuple[int, int]] = []

    for i in range(len(smoothed)):
        high_df = abs(dR_frame[i]) >= thr_df
        high_db = dR_base[i] >= thr_db

        if state == 0:
            if high_df and high_db:
                state = 1
                start_idx = i
                hang = hang_frames
        else:
            if high_df or high_db:
                hang = hang_frames
            else:
                hang -= 1
                if hang <= 0:
                    end_idx = i
                    dur_sec = (end_idx - start_idx) * hop_time
                    if dur_sec >= settings.min_seg_duration_sec:
                        segments_idx.append((start_idx, end_idx))
                    state = 0
                    start_idx = None

    if state == 1 and start_idx is not None:
        end_idx = len(smoothed) - 1
        dur_sec = (end_idx - start_idx) * hop_time
        if dur_sec >= settings.min_seg_duration_sec:
            segments_idx.append((start_idx, end_idx))

    segments: List[Tuple[float, float]] = []
    for s_idx, e_idx in segments_idx:
        start_sec = s_idx * settings.hop_length / float(sr)
        end_sec = e_idx * settings.hop_length / float(sr)
        if end_sec - start_sec >= settings.min_seg_duration_sec:
            segments.append((start_sec, end_sec))

    return segments


def _load_sed_model(settings: Settings):
    """Lazy-load pretrained SED model for segmentation."""
    global _sed_processor, _sed_model

    if _sed_model is None:
        print(f"[SED] Loading pretrained model: {settings.sed_model_id}")
        _sed_processor = AutoProcessor.from_pretrained(settings.sed_model_id)
        _sed_model = AutoModelForAudioClassification.from_pretrained(settings.sed_model_id)
        _sed_model.to(get_device()).eval()

    return _sed_processor, _sed_model


def _load_atst_model(settings: Settings):
    """Lazy-load ATST-SED model for segmentation."""
    global _atst_processor, _atst_model

    if _atst_model is None:
        print(f"[ATST] Loading pretrained model: {settings.atst_model_id}")
        _atst_processor = AutoProcessor.from_pretrained(settings.atst_model_id)
        _atst_model = AutoModelForAudioClassification.from_pretrained(settings.atst_model_id)
        _atst_model.to(get_device()).eval()

    return _atst_processor, _atst_model


def detect_sed_segments(y: np.ndarray, sr: int, settings: Settings) -> List[Tuple[float, float]]:
    """Segmentation using a pretrained SED model (frame-wise if available)."""
    device = get_device()

    if sr != settings.target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=settings.target_sr)
        sr = settings.target_sr

    processor, model = _load_sed_model(settings)

    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits
    probs = torch.sigmoid(logits)

    if probs.ndim == 3:
        frame_probs, _ = probs.max(dim=-1)  # (B, T)
        activity = frame_probs[0].detach().cpu().numpy()
    elif probs.ndim == 2:
        p_any, _ = probs.max(dim=-1)
        p_val = float(p_any[0].item())
        dur = len(y) / float(sr)
        if p_val > settings.sed_activity_threshold and dur >= settings.min_seg_duration_sec:
            return [(0.0, dur)]
        return []
    else:
        raise RuntimeError(f"[SED] Unexpected logits shape: {tuple(logits.shape)}")

    total_dur = len(y) / float(sr)
    return _activity_to_segments(
        activity=activity,
        total_duration_sec=total_dur,
        threshold=settings.sed_activity_threshold,
        min_seg_dur=settings.min_seg_duration_sec,
    )


def detect_atst_segments(y: np.ndarray, sr: int, settings: Settings) -> List[Tuple[float, float]]:
    """Segmentation using ATST-SED / ATST-Frame."""
    device = get_device()

    if sr != settings.target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=settings.target_sr)
        sr = settings.target_sr

    processor, model = _load_atst_model(settings)

    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits
    probs = torch.sigmoid(logits)

    if probs.ndim == 3:
        frame_probs, _ = probs.max(dim=-1)
        activity = frame_probs[0].detach().cpu().numpy()
    elif probs.ndim == 2:
        p_any, _ = probs.max(dim=-1)
        p_val = float(p_any[0].item())
        dur = len(y) / float(sr)
        if p_val > settings.atst_activity_threshold and dur >= settings.min_seg_duration_sec:
            return [(0.0, dur)]
        return []
    else:
        raise RuntimeError(f"[ATST] Unexpected logits shape: {tuple(logits.shape)}")

    total_dur = len(y) / float(sr)
    return _activity_to_segments(
        activity=activity,
        total_duration_sec=total_dur,
        threshold=settings.atst_activity_threshold,
        min_seg_dur=settings.min_seg_duration_sec,
    )


def detect_yamnet_segments(y: np.ndarray, sr: int, settings: Settings) -> List[Tuple[float, float]]:
    """
    YAMNet embedding-change segmentation.
    Dependency: tensorflow_hub (lazy-loaded).
    """
    global _yamnet_model

    if sr != settings.target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=settings.target_sr)
        sr = settings.target_sr

    if _yamnet_model is None:
        import tensorflow_hub as hub
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    waveform = y.astype(np.float32)
    _, embeddings, _ = _yamnet_model(waveform)
    embeddings = embeddings.numpy()

    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    diffs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs) + 1e-12)
    activity = diffs > settings.yamnet_threshold

    segments = []
    start = None
    for i, active in enumerate(activity):
        if active and start is None:
            start = i * settings.yamnet_frame_hop
        elif not active and start is not None:
            end = (i + 1) * settings.yamnet_frame_hop
            if end - start >= settings.min_seg_duration_sec:
                segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, (len(activity) + 1) * settings.yamnet_frame_hop))

    return segments


def detect_vad_segments(wav_path: str, settings: Settings) -> List[Tuple[float, float]]:
    """
    Voice activity detection via Silero VAD (lazy-loaded via torch.hub).
    """
    global _silero
    if _silero is None:
        silero_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _silero = (silero_model, utils)

    silero_model, utils = _silero
    get_speech_timestamps = utils[0]
    read_audio = utils[2]

    audio = read_audio(wav_path, sampling_rate=settings.target_sr)
    speech_timestamps = get_speech_timestamps(audio, silero_model, sampling_rate=settings.target_sr)

    segments = [
        (t["start"] / settings.target_sr, t["end"] / settings.target_sr)
        for t in speech_timestamps
        if (t["end"] - t["start"]) / settings.target_sr >= settings.min_seg_duration_sec
    ]
    return segments


def segment_audio_for_file(wav_path: str, settings: Settings) -> List[Tuple[float, float]]:
    """
    Unified segmentation entrypoint.
    - Loads audio
    - Applies optional pre-filter
    - Runs the selected segmentation method
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # Pre-filtering can stabilize energy/RMS-based segmentation.
    y = apply_pre_filter(y, sr, mode=settings.prefilter_mode)

    if settings.segment_method == "energy":
        return detect_energy_segments(y, sr, settings)
    if settings.segment_method == "energy_mod":
        return detect_energy_mod_segments(y, sr, settings)
    if settings.segment_method == "rms_mod":
        return detect_rms_mod_segments(y, sr, settings)
    if settings.segment_method == "yamnet":
        return detect_yamnet_segments(y, sr, settings)
    if settings.segment_method == "vad":
        return detect_vad_segments(wav_path, settings)
    if settings.segment_method == "sed":
        return detect_sed_segments(y, sr, settings)
    if settings.segment_method == "atst":
        return detect_atst_segments(y, sr, settings)

    raise ValueError(f"Unknown segment_method: {settings.segment_method}")
