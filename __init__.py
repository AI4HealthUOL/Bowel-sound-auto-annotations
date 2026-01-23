"""
Segmentation method dispatcher.

"""

from .energy import detect_energy_segments
from .rms import detect_rms_segments
from .sed import detect_sed_segments
from .yamnet import detect_yamnet_segments
from .vad import detect_vad_segments
import librosa

def segment_audio(wav_path, cfg):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    method = cfg["segmentation"]["method"]

    if method == "energy":
        return detect_energy_segments(y, sr, cfg)
    if method == "rms_mod":
        return detect_rms_segments(y, sr, cfg)
    if method == "sed":
        return detect_sed_segments(y, sr, cfg)
    if method == "yamnet":
        return detect_yamnet_segments(y, sr, cfg)
    if method == "vad":
        return detect_vad_segments(wav_path, cfg)

    raise ValueError(f"Unknown segmentation method: {method}")
