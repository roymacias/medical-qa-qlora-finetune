# Qualitative analysis — gemma-3-4b-it-qlora-finetune / test_ood

Case identifiers are configured in `src/eval/qualitative.py` (`CASE_IDS`).

| Case | id | `gemma-3-4b-it` | `gemma-3-4b-it-qlora-finetune` | `medgemma-4b-it` |
|---|---|---|---|---|
| 3 | `test_ood-00207` | `A` ✓ | `B` ✗ | `A` ✓ |

---

## Case 3

- **id**: `test_ood-00207`
- **gold**: `A`
- **predicted**: `B` (parse_success=True)

**Model generation:**

```text
Reasoning: Management of CIN depends upon the stage of disease. For CIN I, treatment options include close observation with repeat Pap smears every 3-6 months for 2 years or LEEP. Treatment should be considered if there is persistent dysplasia after two negative Pap tests. In addition, patients who have been treated for CIN II or III may require additional therapy. Patients with CIN IV or V should undergo hysterectomy.

Answer: B
```
