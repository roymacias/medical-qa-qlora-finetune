# `data/raw/`

Raw datasets downloaded from HuggingFace Hub. **Not versioned in git** because
the files exceed GitHub's recommended size limits and are fully reproducible
from the extraction script.

## Sources

| Dataset | HuggingFace ID | Reference | License |
|---|---|---|---|
| MedMCQA | `openlifescienceai/medmcqa` | Pal et al., CHIL 2022 | MIT |
| MedQA-USMLE-4-options | `GBaker/MedQA-USMLE-4-options` | Jin et al., 2020 | MIT |

## How to regenerate

From the repository root:

```bash
python -m src.data.extract
```

The script is idempotent — re-running with the data already present is a no-op
unless `--force` is passed. See `src/data/extract.py` for details.
