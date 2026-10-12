import { registerTrainingUI } from "./dinki_training_ui.js";

registerTrainingUI({
    endpoint: "/dinki/minimax_h3", title: "MiniMax H3",
    extensionName: "DINKI.MinimaxH3Training", nodeClass: "DINKI_Minimax_H3_LoRA_Trainer",
    settingId: "DKST.Training.MinimaxH3.Models", jobProperty: "dkst_minimax_h3_job_id", statusType: "DKST_MINIMAX_TRAIN_STATUS",
    optionalModels: ["ref_dit", "training_adapter", "turbo_lora"],
    modelLabels: { python: "Training Python (.venv/Scripts/python.exe)", dit: "MiniMax H3 fl2va DiT",
        ref_dit: "MiniMax H3 ref2va DiT (optional)", vae: "MiniMax H3 Video VAE",
        text_encoder: "Qwen3-VL-32B text encoder (H3 NVFP4 AWQ / BF16)",
        training_adapter: "H3 training adapter (Circlestone recommended, optional)",
        turbo_lora: "H3 Turbo LoRA (optional fl2va previews)" },
});
