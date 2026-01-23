import torch
from config_loader import load_config
from io_utils import collect_inputs
from models.loader import load_model
from segmentation import segment_audio
from prediction import predict_segments
from merging import merge_segments
from writers import write_outputs
from evaluation import run_evaluation

def resolve_device(cfg):
    if cfg["model"]["device"] == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["model"]["device"]

def main():
    cfg = load_config("config/config.yml")
    device = resolve_device(cfg)

    wav_pairs = collect_inputs(cfg)
    feature_extractor, model = load_model(cfg, device)

    all_preds, all_merged = [], []

    for wav_path, patient in wav_pairs:
        segments = segment_audio(wav_path, cfg)
        preds = predict_segments(wav_path, segments, cfg, feature_extractor, model)
        merged_seg, merged_lab = merge_segments(segments, preds, cfg)

        write_outputs(
            wav_path, patient,
            segments, preds,
            merged_seg, merged_lab,
            cfg,
            all_preds,
            all_merged
        )

    if cfg["evaluation"]["enabled"]:
        run_evaluation(cfg)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
