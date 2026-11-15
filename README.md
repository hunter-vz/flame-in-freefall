# 🔥 Flame in Freefall

**NASA's microgravity fire data stops at 22% oxygen. Planned Moon habitats will run at 26.5–34% oxygen.
Our tool predicts how spacecraft cabin materials burn in that gap, tells NASA which fire experiment to run next to close it, and locks its predictions publicly so they can be checked.**

NASA Space Apps Challenge 2026 · Challenge: *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data* · Local event: Pretoria, South Africa

| | |
|---|---|
| 🌐 **Live dashboard** | https://flame-in-freefall-hvz.streamlit.app/ |
| 📓 **Full analysis** | [`Flame_in_Freefall.ipynb`](Flame_in_Freefall.ipynb) (every step, output and finding) |
| 🔒 **Locked predictions** | [`prediction_registry.csv`](prediction_registry.csv) (v1) · [`prediction_registry_v2.csv`](prediction_registry_v2.csv) (v2) |

---

## The pitch

Fire in microgravity behaves differently from fire on Earth: with no buoyancy, a flame only lives if the air moving past it brings in enough oxygen, and too much flow blows it off. NASA has measured this on the ISS (BASS-II), but only up to **22% O₂**. Future lunar habitats are being designed around richer air: **34% O₂ at 8.2 psia** and **26.5% O₂ at 10.2 psia**. Every fire test in that range costs ISS crew time.

Our answer has three parts:

1. **Predict:** a model trained on NASA's own ISS burns gives the chance that a flame survives, for five cabin materials, any oxygen level and any airflow, with an honest 90% range and a label saying how far it is from real data.
2. **Recommend:** a next-experiment method that picks the test that teaches the most. Replayed on NASA's real BASS-II campaign, it learned the flame-out boundary in **27% fewer tests than a careful human strategy**, and the obvious "test near Moon conditions first" strategy did **worse than random**.
3. **Commit:** our predictions for those next tests are frozen with SHA-256 fingerprints and committed here *before* anyone runs them. When NASA runs them, anyone can score us.

---

## Key results

### 1. Fewer experiments to learn the fire limit
We replayed Olson's 26 BASS-II PMMA-rod tests in different orders, from the same 20 random starting sets, and counted how many tests each order needed before the model's error was within 10% of the all-data model's, **and stayed there**.

| Order of testing | Tests needed (mean, range) | Our method vs this order, same start |
|---|---|---|
| **Our method** (most model disagreement inside a trust region) | **11.2** (4–16) | — |
| NASA's real order (one run) | 13 | 14% fewer |
| Spread out (a careful human) | 15.4 (12–18) | **27% fewer**; won 14, tied 3, lost 3 of 20 |
| Random | 17.9 (6–26) | 38% fewer; won 15, tied 2, lost 3 of 20 |
| Highest O₂ first (the "obvious" guess) | 23.1 (13–26) | 52% fewer; won 20 of 20 |

**Why the obvious guess fails:** at high oxygen everything burns, so those tests teach nothing about where flames go out. You have to find the boundary first, then step towards Moon air.

### 2. A materials map of the oxygen limit (microgravity, 14.7 psia)

Yes/no limit = the O₂ below which the model calls a flame-out (with 90% bootstrap range).

| Material | at 5 cm/s | at 20 cm/s | Notes |
|---|---|---|---|
| PMMA plastic: rod, concurrent flow | **15.9%** (13.2–16.4) | 17.3% (16.7–18.7) | lowest limit: survives at the lowest oxygen |
| PMMA plastic: rod, opposed flow | **17.5%** (12.7–18.6) | 18.9% (15.8–21.9) | rests on one test (P8): treat with care |
| PMMA plastic: thin film | **17.6%** (15.4–18.0) | 19.0% (17.1–20.8) | |
| PMMA plastic: thick slab | never went out | — | limit is below 16.8% (lowest burn) |
| SIBAL fabric (cotton–fibreglass) | never went out | — | limit is below 16.4% (lowest burn) |

- Every PMMA form is **extrapolated** to burn at both Moon set-points (26.5% and 34%).
- **SIBAL**, the fabric NASA burned at full scale in SAFFIRE-II, never went out in any BASS-II test, so its microgravity oxygen limit is still unknown in this data. Our *Bracket* test targets exactly that.

### 3. Better than a smart rule, honestly measured
Leave-one-test-out validation (a whole test hidden at a time), all materials, 278 points:

| Method | Balanced accuracy | Brier score (lower is better) |
|---|---|---|
| **Our model** | **0.80** | **0.043** |
| Expert rule "burns if O₂ > threshold" (threshold learned from the data) | 0.76 | 0.049 |
| Base rate ("always burns") | 0.50 | 0.052 |

Oxygen dominates (each extra 1% O₂ multiplies the odds of burning by about 3; doubling the airflow roughly halves them), so a single threshold already gets most of the answer. The model adds airflow, material differences and an uncertainty range on every prediction.

---

## Locked predictions

| Registry | SHA-256 | Frozen (UTC) | Commit |
|---|---|---|---|
| v1: PMMA rods | `fa8bea740f0177e775cd2b87e874a6ee32ae1ae933315fc4225c67a3f8ca0781` | 2026-11-14T15:27:37Z | [`d42ba4c`](https://github.com/hunter-vz/flame-in-freefall/commit/d42ba4c) |
| v2: all materials | `5d6ed7dfb2f3bace6007286fb480a2d72183d1b9ba9dfd3b5fc133eda0f3d8b3` | 2026-11-15T06:46:41Z | [`ee38234`](https://github.com/hunter-vz/flame-in-freefall/commit/ee38234ecc7d3d5524462d6b56cab0210291b526) |

**Registry v2** (all at 14.7 psia, microgravity, ISS SoFIE hardware):

| Track | Material | O₂ | Airflow (cm/s) | Prediction |
|---|---|---|---|---|
| Refine | PMMA rod, concurrent (¼", clear) | 13.5% | 7.75 | 56% chance of burning (90%: 26–92%) → flame-out warning |
| Explore | PMMA rod, concurrent | 25.2% | 20.3 | >99% → burns (extrapolated) |
| Refine | PMMA rod, opposed (3/8", black) | 16.5% | 23.94 | 56% (90%: 14–98%) → toss-up |
| Explore | PMMA rod, opposed | 23.9% | 20.3 | >99% → burns (extrapolated) |
| Refine | PMMA thin film | 17.0% | 30.0 | 64% (90%: 37–93%) → flame-out warning |
| Explore | PMMA thin film | 25.2% | 20.3 | >99% → burns (extrapolated) |
| Bracket | PMMA thick slab | 15.3% | 8 | no prediction: finds the limit |
| Bracket | SIBAL fabric | 14.9% | 10 | no prediction: finds the limit |

Registry v1 (PMMA rods only) also holds the **frontier pressure pair**: 34% O₂ at 5 cm/s at 8.2 psia (no prediction: reduced pressure isn't in the data) and at 14.7 psia, which separates the oxygen effect from the pressure effect. Both are within the planned SoFIE range (0.2–1.5 atm, 10–40% O₂).

**Check us:** download a registry from its commit, then run `certutil -hashfile prediction_registry_v2.csv SHA256` (Windows) or `shasum -a 256 prediction_registry_v2.csv` (Mac/Linux) and compare with the table. The dashboard also checks both fingerprints live.

---

## How it works

### Data (all NASA, nothing invented)
- **BASS-II as-run table** (NASA PSI-25): 129 burn tests run on the ISS in 2014–2016. There is no outcome column: whether a flame went out is written as free-text notes ("ext", "blowoff", "would not stay lit"). We transcribed **123 tests into 430 points** by hand, keeping NASA's raw text next to every point.
- **Fan calibration sheet** (PSI-25 reference data): many airflows were recorded as fan-knob settings. We convert them with that day's and that restrictor's calibration, and never extrapolate. Checked against readings NASA wrote down: knob → fan within 0.55 units, fan → airflow within about 1 cm/s on average.
- **278 usable points, 15 usable flame-outs**, across 5 solid cabin materials. Points that can't be trusted (airflow below the calibrated range, airflow 0, outcome not stated) are kept in the table, flagged with the reason, and left out of the model.
- **SAFFIRE-II** (NASA PSI-99): used only as a sanity check. SIBAL and PMMA both burned in space at about 22% O₂ and 20 cm/s, matching our model's "burns" call there.

### Pipeline (notebook sections)
1. **Transcribe and clean:** every flow setting in time order, O₂ stepped from each test's start to end value, airflow converted from the calibration, a weight per point (1 measured, 0.5 converted).
2. **Model:** logistic regression on O₂, ln(airflow) and a material indicator, fitted with per-point weights and **no class reweighting**, so probabilities stay honest. The yes/no call uses the weighted burn base rate as its threshold.
3. **Validate:** leave-one-**test**-out (points from one test share a sample and a day, so hiding single points would flatter the model), compared with a base-rate baseline and an expert O₂-threshold rule.
4. **Uncertainty:** 100 bootstrap models, resampled by test within each material, give a 90% range for every prediction and a *disagreement* score.
5. **Recommend the next test**, in three tracks:
   - **Refine:** the test inside a *trust region* where the models disagree most, weighted towards Moon-relevant oxygen and the crew-cabin ventilation band (7.6–20.3 cm/s).
   - **Explore:** one controlled step (3 points of O₂) above the highest oxygen tested, towards Moon air. The trust region grows ×1.5 when predictions come true and shrinks ×0.5 when they fail.
   - **Bracket:** for materials that never went out, 1.5 points of O₂ below their lowest burn, at the same airflow.
6. **Replay:** NASA's real campaign is replayed one test at a time. Each prediction is locked before the real result is revealed, then scored.
7. **Freeze:** registries are written once, locked against overwriting, fingerprinted and committed.

### Decisions that matter
- **Never invent data:** unreadable values stay blank and every point traces back to NASA's text.
- **A simple model on purpose:** 15 flame-outs is far too few for deep learning. A small model with honest uncertainty is what safety work needs.
- **Beat a smart human, not just random:** the "spread out" baseline was added before seeing its result, and our method was not retuned afterwards.
- **Label extrapolation everywhere:** every prediction carries a trust badge, and anything above 22.2% O₂ is marked as extrapolated.

---

## Dashboard

Live at **https://flame-in-freefall-hvz.streamlit.app/**. It uses the same model code as the notebook (`fif_model.py`); the notebook checks that both give identical numbers to within 1e-13.

1. **Try it yourself:** pick a material, oxygen and airflow, and get the chance of burning, a 90% range, the call and a trust badge (🟢 inside the data, 🟡 near, 🔴 extrapolation, ⚪ low flow). NASA's own points and the 3 nearest tests are shown alongside.
2. **Locked predictions:** both registries, with their fingerprints verified live and instructions to check them yourself.
3. **Next experiment:** the Refine, Explore and Bracket tests for each material, and the replay results.
4. **Materials map:** each material's oxygen limit against ISS air and the two Moon set-points, at any airflow.
5. **Add new results:** enter a new experiment (or upload a CSV), score it against the locked predictions, retrain the model and see how the limit and the next recommended test move. Session only: nothing is written to this repository.

### Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The notebook runs in Google Colab (it mounts Google Drive) or locally from this folder.

---

## Repository contents

| File | What it is |
|---|---|
| `Flame_in_Freefall.ipynb` | Full analysis: data, model, validation, replay, registries, export |
| `app.py`, `requirements.txt` | Streamlit dashboard |
| `fif_model.py` | Shared model module (fit, predict, next experiment), used by the notebook check and the dashboard |
| `model_export.json` | Fitted coefficients, 100 bootstrap sets, threshold and tested ranges per material |
| `all_test_points.csv` | Every transcribed point (430), with flags and reasons for any exclusion |
| `prediction_registry.csv`, `prediction_registry_v2.csv` | Frozen predictions (never edited) |
| `PSI-25_Experimental table_BASS-II.csv`, `Fan calibration data.xlsx` | NASA source data |

---

## Limitations (stated plainly)
- **Pressure:** every BASS-II test ran at ISS cabin pressure (~14.7 psia). The effect of the lower habitat pressures (8.2 and 10.2 psia) is **not in the data**; that is why the frontier pressure pair exists.
- **Extrapolation:** NASA's data stops at 22.2% O₂. Predictions above it extend the same straight line and the 90% range can't show that uncertainty; they are labelled as extrapolated.
- **Few flame-outs:** 15 usable flame-outs in total. The opposed-rod limit rests on one test (P8). Low-airflow flame-outs were mostly recorded below the calibrated fan range, so the boundary below 2 cm/s is unknown.
- **Replay scope:** the replay covers Olson's PMMA rods only (≤22% O₂). It validates the Refine part of the method, not the Moon-level predictions.

---

## Sources
- NASA Physical Sciences Informatics, **PSI-25: BASS-II** experimental table and fan calibration reference data: https://psi.nasa.gov
- NASA Physical Sciences Informatics, **PSI-99: SAFFIRE-II** experimental table
- NASA NTRS **20240003211**, *Moon to Mars (M2M): Exploration Atmosphere* (8.2 psia, 34% O₂)
- NASA NTRS **20220013669**, *Internal Layout of a Lunar Surface Habitat* (8.2 psia / 34% O₂ or 10.2 psia / 26.5% O₂)
- NASA, *Researcher's Guide to Combustion Science* (SoFIE planned ranges: 0.2–1.5 atm, 10–40% O₂, flows up to 30 cm/s)
- NASA NESC Technical Bulletin 26-05, *Webbings for Use in Elevated Oxygen Environments*

## Use of AI
This project was built with the help of an AI assistant (Anthropic's Claude), used for writing and debugging code, checking transcriptions against NASA's raw text, drafting analysis text and documentation, and reviewing results. All data comes from NASA; nothing was generated or invented. Project direction, design decisions and final review were done by the team.

## Team
Hunter, North-West University (Potchefstroom), South Africa
