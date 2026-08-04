# phenoage-transport

**Transporting the PhenoAge clinical aging clock to a high-altitude, mixed-ancestry cohort — and showing why density-ratio adaptation isn't enough.**

`Python · scikit-learn · ONNX · Vercel`

---

Aging clocks are fit at sea level on predominantly European-ancestry cohorts (PhenoAge: NHANES III). When you apply one to a Colombian cohort in Bogotá (~2,600 m), chronic hypobaric hypoxia inflates the erythroid biomarkers — RDW and MCV — and the clock reads that adaptation as accelerated biological aging. This repo builds the full pipeline to quantify that failure and correct it, and serves the corrected model as an explainable inference endpoint.

The central result is a negative one about a popular fix: **importance-weighting (density-ratio domain adaptation) barely moves the calibration bias, because the bias is mechanistic, not distributional.** Altitude is a first-order confounder on the measurement, so you have to correct the biomarker — you can't reweight your way out of it.

## Result (synthetic demonstration cohort)

| Transport strategy | Spearman ρ (discrimination) | Calibration slope | Mean bias (yr) |
|---|---|---|---|
| Naïve (no adaptation) | 0.990 | 1.034 | **+5.51** |
| uLSIF density-ratio reweighting | 0.991 | 1.037 | **+5.48** |
| **Altitude-corrected (mechanistic)** | 0.997 | 1.013 | **+0.12** |

Discrimination was never the problem — the ranking is near-perfect throughout. The clock is *miscalibrated*, and only the mechanistic correction fixes it. Separating these two axes matters especially here: with a small target cohort and no local mortality outcomes, calibration cannot be taken for granted from good discrimination.

> Numbers above are from a synthetic cohort with a mechanistically-injected altitude effect, so the pipeline runs end-to-end without restricted clinical data. The methods transfer directly to the real cohort.

## Method

- **PhenoAge reference** (`src/phenoage.py`) — published Levine et al. (2018) coefficients and the Gompertz mortality-score → phenotypic-age mapping. The erythroid marker is **RDW, not RBC** (a frequent transcription error).
- **Covariate shift by construction** (`src/data_gen.py`) — a sea-level source cohort and an altitude target cohort whose erythroid markers are shifted mechanistically per 1,000 m, while ground-truth aging is held at the sea-level-equivalent value. This encodes *altitude perturbs the measurement, not the person*.
- **Density-ratio domain adaptation** (`src/domain_adapt.py`) — **uLSIF** (closed-form, preferred at n≈150) and **KLIEP**, estimating importance weights w(x)=p_target/p_source directly, with Kish effective-sample-size diagnostics.
- **Biomarker network** (`src/graphical_model.py`) — **EBICglasso** partial-correlation graph. EBICglasso (precision matrix, direct effects, EBIC-selected sparsity) is the right tool at p=10 / n≈150; WGCNA is built for high-dimensional transcriptomics and estimates marginal co-expression, which is the wrong object here.
- **Interpretability** (`src/train.py`) — SHAP on the target cohort; RDW is the top biomarker driver of the bio-age gap after chronological age, which is exactly the altitude signal.

## Serving

The adapted model is exported to **ONNX** and served from a dependency-light **Vercel** Python function (`api/predict.py`, `onnxruntime` + `numpy` only — no torch/sklearn in the bundle). `POST /api/predict` returns phenotypic age, the chronological-age gap, a per-feature contribution breakdown, and an optional altitude-corrected estimate.

```bash
curl -X POST https://<your-deployment>/api/predict \
  -H "Content-Type: application/json" \
  -d '{"albumin":45,"creatinine":80,"glucose":5.2,"crp":1.0,
       "lymphocyte_pct":30,"mcv":92,"rdw":14.5,"alkaline_phosphatase":75,
       "wbc":6.5,"age":52,"altitude_m":2600}'
```

## Reproduce

```bash
pip install -r requirements.txt
python run_pipeline.py      # data -> adaptation -> networks -> model -> ONNX
python -m pytest tests/     # scientific sanity + ONNX parity
python api/predict.py       # smoke-test the endpoint locally
```

Deploy: `vercel deploy` (config in `vercel.json`; the front-end in `public/` calls the function).

## Layout

```
src/phenoage.py         PhenoAge reference implementation (Levine 2018)
src/data_gen.py         source / altitude-target cohort generation
src/domain_adapt.py     uLSIF + KLIEP density-ratio estimators
src/graphical_model.py  EBICglasso biomarker network
src/train.py            training, transportability eval, SHAP, ONNX export
api/predict.py          Vercel serverless inference + explanation
public/index.html       explainable front-end
tests/                  scientific + parity tests
```

## Scope

Research artifact — **not a medical device**, not for clinical or diagnostic use. The cohort here is synthetic; the modeling choices are built to transfer to the real HumanoLab AI cohort under the thesis protocol.

---

*Levine ME et al. "An epigenetic biomarker of aging for lifespan and healthspan." Aging (2018). PhenoAge parameterized on NHANES III (1988–1994).*
