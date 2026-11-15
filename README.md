# Flame in Freefall

Predicting material flammability in microgravity at lunar-habitat oxygen levels, and selecting the next fire experiment to run.

NASA Space Apps Challenge 2026, challenge *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*. Local event: Pretoria, South Africa.

- Dashboard: https://flame-in-freefall-hvz.streamlit.app/
- Analysis notebook: [Flame_in_Freefall.ipynb](Flame_in_Freefall.ipynb)
- Prediction registries: [prediction_registry.csv](prediction_registry.csv) (v1), [prediction_registry_v2.csv](prediction_registry_v2.csv) (v2)

## Overview

NASA's microgravity flammability data for solid materials (BASS-II, ISS) covers oxygen concentrations up to 22% by volume. Proposed lunar surface habitat atmospheres are richer in oxygen: 34% O₂ at 8.2 psia and 26.5% O₂ at 10.2 psia. No microgravity burn data exists in that range, and each new test uses limited ISS resources.

This project does three things:

1. Fits a model to NASA's BASS-II results that estimates the probability that a flame survives, for five cabin materials, as a function of oxygen concentration and airflow, with a 90% uncertainty range.
2. Recommends the next experiment to run, choosing the test expected to improve the model most. On a replay of the BASS-II campaign, this approach reached the final model's accuracy in fewer tests than random, chronological or human-designed test orders.
3. Records its predictions for the recommended tests in registries that were frozen and committed to this repository, with SHA-256 checksums, before any of the tests are run.

## Results

### Test-selection replay

Olson's 26 BASS-II PMMA rod tests were replayed in different orders from the same 20 randomly drawn starting sets. For each order we counted the number of tests needed for the model's Brier score to fall within 10% of the all-data value and remain there.

| Test order | Tests needed, mean (range) | Comparison with our method, same start |
|---|---|---|
| Our method | 11.2 (4–16) | |
| NASA's original order (single run) | 13 | 14% fewer tests |
| Evenly spread conditions | 15.4 (12–18) | 27% fewer; 14 wins, 3 ties, 3 losses |
| Random | 17.9 (6–26) | 38% fewer; 15 wins, 2 ties, 3 losses |
| Highest oxygen first | 23.1 (13–26) | 52% fewer; 20 wins of 20 |

Testing the highest oxygen levels first performed worse than a random order. Nearly all high-oxygen tests burn, so they add little information about the extinction boundary.

### Oxygen limit by material

Oxygen concentration below which the model predicts extinction, at 14.7 psia in microgravity, with 90% bootstrap intervals.

| Material | 5 cm/s | 20 cm/s | Notes |
|---|---|---|---|
| PMMA rod, concurrent flow | 15.9% (13.2–16.4) | 17.3% (16.7–18.7) | Lowest limit of the materials tested |
| PMMA rod, opposed flow | 17.5% (12.7–18.6) | 18.9% (15.8–21.9) | Based on a single test (P8) |
| PMMA thin film | 17.6% (15.4–18.0) | 19.0% (17.1–20.8) | |
| PMMA thick slab | not determined | not determined | No extinction observed; lowest burn at 16.8% |
| SIBAL fabric (cotton–fibreglass) | not determined | not determined | No extinction observed; lowest burn at 16.4% |

All three PMMA forms are predicted to burn at both lunar set-points; these predictions are extrapolations beyond the tested range. SIBAL and the thick PMMA slab did not extinguish in any BASS-II test, so their limits remain undetermined.

### Validation

Leave-one-test-out cross-validation over all materials (278 points):

| Method | Balanced accuracy | Brier score |
|---|---|---|
| Logistic model | 0.80 | 0.043 |
| Oxygen threshold rule (threshold fitted to training data) | 0.76 | 0.049 |
| Base rate | 0.50 | 0.052 |

Oxygen concentration is the dominant variable. Each additional percentage point of O₂ multiplies the odds of burning by approximately 2.9, and doubling the airflow multiplies them by approximately 0.47.

## Prediction registries

| Registry | SHA-256 | Frozen (UTC) | Commit |
|---|---|---|---|
| v1, PMMA rods | `fa8bea740f0177e775cd2b87e874a6ee32ae1ae933315fc4225c67a3f8ca0781` | 2026-11-14 15:27:37 | [d42ba4c](https://github.com/hunter-vz/flame-in-freefall/commit/d42ba4c) |
| v2, all materials | `5d6ed7dfb2f3bace6007286fb480a2d72183d1b9ba9dfd3b5fc133eda0f3d8b3` | 2026-11-15 06:46:41 | [ee38234](https://github.com/hunter-vz/flame-in-freefall/commit/ee38234ecc7d3d5524462d6b56cab0210291b526) |

Registry v2 (all tests at 14.7 psia, microgravity, ISS SoFIE hardware):

| Track | Material | O₂ | Airflow (cm/s) | P(burn), 90% range | Call |
|---|---|---|---|---|---|
| Refine | PMMA rod, concurrent (0.25 in, clear) | 13.5% | 7.75 | 0.56 (0.26–0.92) | Flame-out warning |
| Explore | PMMA rod, concurrent | 25.2% | 20.3 | >0.99 | Burns (extrapolated) |
| Refine | PMMA rod, opposed (0.375 in, black) | 16.5% | 23.94 | 0.56 (0.14–0.98) | Toss-up |
| Explore | PMMA rod, opposed | 23.9% | 20.3 | >0.99 | Burns (extrapolated) |
| Refine | PMMA thin film | 17.0% | 30.0 | 0.64 (0.37–0.93) | Flame-out warning |
| Explore | PMMA thin film | 25.2% | 20.3 | >0.99 | Burns (extrapolated) |
| Bracket | PMMA thick slab | 15.3% | 8 | none | Locates the limit |
| Bracket | SIBAL fabric | 14.9% | 10 | none | Locates the limit |

Registry v1 also includes a pressure pair at 34% O₂ and 5 cm/s, at 8.2 psia (no prediction, as reduced pressure is outside the data) and 14.7 psia. Both conditions fall within the planned SoFIE operating range (0.2–1.5 atm, 10–40% O₂).

To verify a registry, download it from its commit and compute its checksum with `certutil -hashfile <file> SHA256` (Windows) or `shasum -a 256 <file>` (macOS/Linux). The dashboard also verifies both checksums.

## Method

### Data

- **BASS-II as-run table (NASA PSI-25).** 129 tests run on the ISS between 2014 and 2016. Outcomes are recorded only as free-text notes (for example "ext", "blowoff", "would not stay lit"), so each test was transcribed manually. 123 tests give 430 data points, each stored with the original NASA text.
- **Fan calibration data (PSI-25 reference data).** Airflow was often logged as a fan setting. These were converted using the calibration for the same day and flow restrictor, without extrapolation. Against airflows recorded during the tests, the fan-setting conversion agrees within 0.55 display units and the airflow conversion within about 1 cm/s on average.
- **Usable data.** 278 points with 15 extinctions across five solid materials. Points with airflow below the calibrated range, zero airflow or no recorded outcome are retained in the dataset with the reason for exclusion and are not used for fitting.
- **SAFFIRE-II (NASA PSI-99).** Used as a consistency check only. SIBAL and PMMA burned at approximately 22% O₂ and 20 cm/s, in agreement with the model.

### Pipeline

1. **Transcription and cleaning.** Flow settings are recorded in time order, O₂ is interpolated between each test's initial and final values, airflow is converted from the calibration, and each point is weighted (1.0 measured, 0.5 converted).
2. **Model.** Weighted logistic regression on O₂, ln(airflow) and material indicators. No class reweighting is applied, so predicted probabilities remain calibrated; the burn/extinction call uses the weighted base rate as its threshold.
3. **Validation.** Leave-one-test-out, since points within a test share a sample and conditions. Compared against a base-rate model and a fitted oxygen-threshold rule.
4. **Uncertainty.** 100 bootstrap models, resampled by test within each material, provide 90% intervals and a disagreement measure.
5. **Experiment selection**, in three tracks:
   - **Refine:** the condition within a trust region where the bootstrap models disagree most, weighted towards lunar-relevant oxygen levels and the crew-cabin ventilation range (7.6–20.3 cm/s).
   - **Explore:** a fixed step of 3 percentage points of O₂ above the highest tested level. The trust region expands by a factor of 1.5 after a correct prediction and contracts by 0.5 after an incorrect one.
   - **Bracket:** for materials with no observed extinction, 1.5 percentage points of O₂ below the lowest burn, at the same airflow.
6. **Replay.** The BASS-II campaign is replayed test by test; each prediction is recorded before the actual outcome is revealed.
7. **Registry.** Predictions are written once, protected against overwriting, checksummed and committed.

### Design choices

- No values are estimated or filled in; every data point can be traced to the NASA record.
- A logistic model was chosen over more complex models because only 15 extinctions are available.
- The evenly spread baseline was defined before its results were seen, and the selection method was not adjusted afterwards.
- Every prediction is labelled according to its distance from the tested data.

## Dashboard

https://flame-in-freefall-hvz.streamlit.app/

The dashboard uses the same model code as the notebook (`fif_model.py`). The notebook confirms that both produce identical results.

1. **Try it yourself.** Select a material, oxygen level and airflow to obtain the probability of burning, its 90% interval, the predicted outcome and an indicator of how far the condition lies from tested data.
2. **Locked predictions.** Both registries, with checksums verified on load.
3. **Next experiment.** Recommended Refine, Explore and Bracket tests for each material, and the replay results.
4. **Materials map.** Oxygen limits by material compared with ISS air and the lunar set-points, at a selectable airflow.
5. **Add new results.** Enter new test results or upload a CSV, compare them with the registered predictions, and retrain the model. Entries are kept for the session only.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The notebook runs in Google Colab (with Google Drive) or locally from the repository folder.

## Repository contents

| File | Description |
|---|---|
| `Flame_in_Freefall.ipynb` | Full analysis: data processing, model, validation, replay, registries and export |
| `app.py`, `requirements.txt` | Streamlit dashboard |
| `fif_model.py` | Model module used by the dashboard and checked against the notebook |
| `model_export.json` | Fitted coefficients, bootstrap coefficient sets, threshold and tested ranges |
| `all_test_points.csv` | All 430 transcribed data points with exclusion flags |
| `prediction_registry.csv`, `prediction_registry_v2.csv` | Frozen prediction registries |
| `PSI-25_Experimental table_BASS-II.csv`, `Fan calibration data.xlsx` | NASA source data |

## Limitations

- All BASS-II tests ran at ISS cabin pressure (approximately 14.7 psia). The effect of reduced habitat pressure is not represented in the data.
- The highest oxygen level tested is 22.2%. Predictions above this are extrapolations, and the bootstrap intervals do not capture that uncertainty.
- Only 15 usable extinctions are available. The opposed-flow rod limit depends on a single test, and the boundary below 2 cm/s is undetermined because low-flow extinctions were recorded below the calibrated fan range.
- The replay covers Olson's PMMA rod tests only (up to 22% O₂) and validates the Refine track, not the predictions at lunar oxygen levels.

## Sources

- NASA Physical Sciences Informatics, PSI-25: BASS-II experimental table and fan calibration data. https://psi.nasa.gov
- NASA Physical Sciences Informatics, PSI-99: SAFFIRE-II experimental table.
- NASA NTRS 20240003211, *Moon to Mars (M2M): Exploration Atmosphere*.
- NASA NTRS 20220013669, *Internal Layout of a Lunar Surface Habitat*.
- NASA, *Researcher's Guide to Combustion Science* (SoFIE operating ranges).
- NASA NESC Technical Bulletin 26-05, *Webbings for Use in Elevated Oxygen Environments*.

## Use of AI tools

An AI assistant (Anthropic's Claude) was used for code development and debugging, checking transcriptions against the NASA records, and drafting documentation. All data comes from NASA sources. Project direction, design decisions and review of all results were done by the team.

## Team

Hunter, North-West University, Potchefstroom, South Africa
