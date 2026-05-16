# `artifacts/gemma-3-4b-it-qlora-finetune/`

Fine-tuned LoRA adapter weights (`.safetensors`) and model checkpoints. **Not versioned in git** because the files exceed GitHub's recommended size limits and are fully reproducible from the training pipeline.

## How to regenerate

From the repository root, ensuring the raw data has already been processed:

```bash
python -m src.pipelines.training_pipeline
```
