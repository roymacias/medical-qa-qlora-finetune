# Qualitative analysis — gemma-3-4b-it-qlora-finetune / test_id

Case identifiers are configured in `src/eval/qualitative.py` (`CASE_IDS`).

| Case | id | `gemma-3-4b-it` | `gemma-3-4b-it-qlora-finetune` | `medgemma-4b-it` |
|---|---|---|---|---|
| 1 | `de09d388-bd4e-42a9-ac6b-ee2d95f822e2` | `B` ✓ | `B` ✓ | `B` ✓ |
| 2 | `d2398cd6-b205-4fb3-a4c4-9e575662b0bf` | `A` ✓ | `—` ✗ | `A` ✓ |

---

## Case 1

- **id**: `de09d388-bd4e-42a9-ac6b-ee2d95f822e2`
- **gold**: `B`
- **predicted**: `B` (parse_success=True)

**Model generation:**

```text
Reasoning: In pulmonary atresia there is absence of pulmonary aerial blood flow to lungs leading to hypoxemia and cyanosis. There is no cardiac enlargement because right ventricle is not working. So it will be smaller than normal size.

Answer: B
```

## Case 2

- **id**: `d2398cd6-b205-4fb3-a4c4-9e575662b0bf`
- **gold**: `A`
- **predicted**: `None` (parse_success=False)

**Model generation:**

```text
Reasoning: Ans. (a) Transthyretin(Ref: Harper 31/e page 254; Robbins 9th/pg 670)* Transthyretin is involved in thyroid hormone transport and copper homeostasis.* Ceruloplasmin is an acute phase protein that binds to Cu+ ions and transports them throughout the body.* Hepcidin regulates iron absorption from the gut and release of Fe from macrophages into circulation.* Ferritin is the major storage form of Fe in the body. It can store up to 4500 mg of Fe per molecule.FerritinIron StorageProtein* Ferritin* Hemosiderin* Siderophilin* Transferrin* Apoferritin* Hemeoxygenase* Ferroportin* Iron binding proteins* Ceruloplasmin* Transferrin receptor* FPN* Hephaestin* Hepcidin* Iron regulatory P-protein* Iron regulatory factor* Iron response element* Transcription factors* RNA polymerase II* mRNA transcriptions* Ribosomal RNA* tRNA* rRNA* UBIAD1* UBIAD2* UBIAD3* UBIAD4* UBIAD5* UBIAD6* UBIAD7* UBIAD8* UBIAD9* UBIAD10* UBIAD11* UBIAD12* UBIAD13* UBIAD14* UBIAD15* UBIAD16* UBIAD17* UBIAD18* UBIAD19* UBIAD20* UBIAD21* UBIAD22* UBIAD23* UBIAD24* UBIAD25* UBIAD26* UBIAD27* UBIAD28* UBIAD29* UBIAD30* UBIAD31* UBIAD32* UBIAD33* UBIAD34* UBIAD35* UBIAD36* UBIAD37* UBIAD38* UBIAD39* UBIAD40* UBIAD41* UBIAD42* UBIAD43* UBIAD44* UBIAD45* UBIAD46* UBIAD47* UBIAD48* UBIAD49* UBIAD50* UBIAD51* UBIAD52* UBIAD53* UBIAD54* UBIAD55* UBIAD56* UBIAD57* UBIAD58* UBIAD59* UBIAD60* UBIAD61* UBIAD62* UBIAD63* UBIAD64* UBIAD65* UBIAD66* UBIAD67* UBIAD68* UBIAD69* UBIAD70* UBIAD71* UBIAD72* UBIAD73* UBIAD74* UBIAD75* UBIAD76* UBIAD77* UBIAD78* UBIAD79* UBIAD80* UBIAD81* UBIAD82* UBIAD83* UBIAD84* UBIAD85* UBIAD86* UBIAD87* UBIAD88* UBIAD89* UBIULP* ULPD1* ULPD2* ULPD3* ULPD4* ULPD5* ULPD6* ULPD7* ULPD8* ULPD9* ULPD10* ULPD11* ULPD12* ULPD13* ULPD14* ULPD15* ULPD16* ULPD17* ULPD18* ULPD19* ULPD20* ULPD21* ULPD22* ULPD23* ULPD24* ULPD25* ULPD26* ULPD27* ULPD28* ULPD29* ULPD30* ULPD31* ULPD32* ULPD33* ULPD34* ULPD35* ULPD36* ULPD37* ULPD38* ULPD39* ULPD40* ULPD41* ULPD42* ULPD43* ULPD44* ULPD45* ULPD46* ULPD47* ULPD48* ULPD49* ULPD50* ULPD51* ULPD52* ULPD53* ULPD54* ULPD55* ULPD56* ULPD57* ULPD58
```
