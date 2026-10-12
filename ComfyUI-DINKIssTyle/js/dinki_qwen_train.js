import { registerTrainingUI } from "./dinki_training_ui.js";

registerTrainingUI({
    endpoint: "/dinki/qwen21", title: "Qwen 2.1",
    extensionName: "DINKI.Qwen21Training", nodeClass: "DINKI_Qwen21_LoRA_Trainer",
    settingId: "DKST.Training.Qwen21.Models", jobProperty: "dkst_qwen21_job_id", statusType: "DKST_QWEN_TRAIN_STATUS",
    optionalModels: ["turbo_lora"],
    modelLabels: { python: "Training Python (.venv/Scripts/python.exe)", dit: "Qwen Image 2.1 base DiT",
        vae: "Qwen Image 2.1 VAE (ComfyUI / Diffusers)", text_encoder: "Qwen3-VL-8B text encoder",
        training_adapter: "Fizgig Qwen 2.1 training adapter", turbo_lora: "Qwen 2.1 Turbo LoRA (optional previews)" },
});
