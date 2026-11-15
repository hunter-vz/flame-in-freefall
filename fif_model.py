"""
fif_model.py - Flame in Freefall: the pooled flammability model, as a plain Python module.

The dashboard imports this file. It reproduces the notebook's pooled model (sections 18-19) and its
Refine / Explore logic (sections 13 and 21) exactly, from all_test_points.csv (written by the notebook).

Uses only numpy, pandas and scikit-learn. No Colab-specific code.

Typical use:
    import fif_model
    fif_model.init('all_test_points.csv')          # fits the model and the 100 bootstrap models
    fif_model.predict('PMMA plastic: thin film', o2=21.0, airflow=10.0)
    fif_model.next_experiment('PMMA plastic: thin film')
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# settings (identical to the notebook)
# ------------------------------------------------------------------------------
SEED = 42                                   # random seed for the bootstrap
N_BOOT = 100                                # number of bootstrap models
REGULARISATION_C = 1.0                      # logistic regression regularisation (scikit-learn default)
MIN_OUTS_BOOT = 3                           # fewest flame-outs a bootstrap resample must contain
REFERENCE_MATERIAL = 'PMMA plastic: rod (concurrent)'
MATERIAL_ORDER = ['PMMA plastic: rod (concurrent)', 'PMMA plastic: rod (opposed)', 'PMMA plastic: thin film',
                  'PMMA plastic: thick slab', 'SIBAL fabric (cotton–fibreglass)']
MIN_OUTS_FOR_LIMIT = 2                      # usable flame-outs needed to say a limit was found
NO_DATA_BELOW_CM_S = 2.0                    # below this airflow the boundary is not known
FLOW_MAX_RECOMMEND = 30.0                   # top of SoFIE's planned flow range (cm/s)
O2_MARGIN = 3.0                             # trust region: O2 step above the highest tested (percentage points)
AIRFLOW_FACTOR = 3.0                        # trust region: airflow factor beyond the tested range
RADIUS = 1.0                                # trust radius (starting size)
O2_GRID_STEP = 0.5                          # Refine grid: O2 step
N_AIRFLOWS = 25                             # Refine grid: number of airflows
N_BAND_AIRFLOWS = 25                        # Explore: number of airflows tried in the cabin band
RAMP_TOP_O2 = 34.0                          # relevance ramp reaches 1 here (NASA's main habitat set-point)
RELEVANCE_FLOOR = 0.2                       # lowest relevance
CABIN_FLOW_BAND = (7.6, 20.3)               # cabin ventilation band (cm/s): 15-40 ft/min
FLOW_WIDTH_FACTOR = 2.0                     # how fast airflow relevance fades outside the band
TEST_PRESSURE_PSIA = 14.7                   # every prediction is at ISS cabin pressure
PLATFORM = 'ISS (SoFIE, Combustion Integrated Rack)'
CAVEAT = 'at ISS cabin pressure (~14.7 psia); effect of reduced pressure not in the data'

# filled in by init()
STATE = {}


# ------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------
def load_data(path):
    """Read all_test_points.csv and return the usable points, in file order."""
    df = pd.read_csv(path)
    # usable_for_model is written as True/False text
    usable = df['usable_for_model'].astype(str).str.lower() == 'true'
    return df[usable].reset_index(drop=True)


def indicator_materials():
    """Every material except the reference, in the fixed order."""
    return [m for m in MATERIAL_ORDER if m != REFERENCE_MATERIAL]


def make_xyw(df):
    """Model inputs (O2, ln airflow, one 0/1 column per non-reference material), answers and weights."""
    X = np.column_stack([df['o2_pct'], np.log(df['airflow_cm_s'])]
                        + [(df['material'] == m).astype(float) for m in indicator_materials()])
    return X, df['burned'].to_numpy(dtype=float), df['weight'].to_numpy(dtype=float)


def burn_base_rate(y, w):
    """Weighted share of burns: the yes/no threshold."""
    return float(np.sum(w * y) / np.sum(w))


# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
def _fit_pipeline(X, y, w):
    """Scale the inputs, then a logistic regression with no class weighting."""
    m = Pipeline([('scale', StandardScaler()), ('logreg', LogisticRegression(C=REGULARISATION_C, class_weight=None))])
    m.fit(X, y, logreg__sample_weight=w)
    return m


def _original_units(m):
    """Coefficients in original units: [b0, b_o2, b_flow, b_material...]."""
    mean, spread = m['scale'].mean_, m['scale'].scale_
    c, c0 = m['logreg'].coef_[0], m['logreg'].intercept_[0]
    return np.concatenate([[c0 - np.sum(c * mean / spread)], c / spread])


def fit_model(df):
    """Fit the pooled model on usable points. Returns a dict with the pipeline, coefficients and threshold."""
    X, y, w = make_xyw(df)
    pipe = _fit_pipeline(X, y, w)
    return {'pipeline': pipe, 'coefs': _original_units(pipe), 'threshold': burn_base_rate(y, w)}


def bootstrap_models(df, n=N_BOOT, seed=SEED):
    """n coefficient sets, each from a resample of whole tests drawn separately inside each material."""
    g = np.random.default_rng(seed)
    tests_by_material = {m: df.loc[df['material'] == m, 'test_id'].unique() for m in MATERIAL_ORDER}
    rows_of = {t: np.flatnonzero(df['test_id'].to_numpy() == t) for t in df['test_id'].unique()}
    coefs = []
    while len(coefs) < n:
        picked = np.concatenate([g.choice(tests_by_material[m], size=len(tests_by_material[m]), replace=True)
                                 for m in MATERIAL_ORDER])
        sample = df.iloc[np.concatenate([rows_of[t] for t in picked])]
        Xb, yb, wb = make_xyw(sample)
        if (yb == 0).sum() < MIN_OUTS_BOOT or (yb == 1).sum() == 0:
            continue
        coefs.append(_original_units(_fit_pipeline(Xb, yb, wb)))
    return np.array(coefs)


def material_info(df):
    """Tested ranges, counts and limit status for each material (usable points)."""
    info = {}
    for m in MATERIAL_ORDER:
        u = df[df['material'] == m]
        n_out = int((u['burned'] == 0).sum())
        info[m] = {'o2_min': float(u['o2_pct'].min()), 'o2_max': float(u['o2_pct'].max()),
                   'airflow_min': float(u['airflow_cm_s'].min()), 'airflow_max': float(u['airflow_cm_s'].max()),
                   'tests': int(u['test_id'].nunique()), 'usable_points': int(len(u)), 'flame_outs': n_out,
                   'limit_found': n_out >= MIN_OUTS_FOR_LIMIT}
    return info


def init(path, n_boot=N_BOOT, seed=SEED):
    """Load the data, fit the model and the bootstrap models, and keep them for predict() / next_experiment()."""
    df = load_data(path)
    STATE.update({'df': df, 'model': fit_model(df), 'boot': bootstrap_models(df, n_boot, seed),
                  'info': material_info(df)})
    return STATE


# ------------------------------------------------------------------------------
# predictions
# ------------------------------------------------------------------------------
def _features(material, o2, airflow):
    """Feature rows [1, O2, ln airflow, indicators...] for arrays of O2 and airflow."""
    if material not in MATERIAL_ORDER:
        raise ValueError(f'unknown material: {material}')
    o2, airflow = np.atleast_1d(np.asarray(o2, float)), np.atleast_1d(np.asarray(airflow, float))
    o2, airflow = np.broadcast_arrays(o2, airflow)
    cols = [np.ones_like(o2), o2, np.log(airflow)] + [np.full_like(o2, float(material == m)) for m in indicator_materials()]
    return np.column_stack(cols)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def trust_badge(material, o2, airflow):
    """Where a condition sits relative to this material's tested data."""
    info = STATE['info'][material]
    if airflow < NO_DATA_BELOW_CM_S:
        badge = 'low-flow (<2 cm/s): boundary not known'
    elif info['o2_min'] <= o2 <= info['o2_max'] and info['airflow_min'] <= airflow <= info['airflow_max']:
        badge = 'inside tested range'
    elif (info['o2_min'] - O2_MARGIN <= o2 <= info['o2_max'] + O2_MARGIN
          and info['airflow_min'] / AIRFLOW_FACTOR <= airflow <= info['airflow_max'] * AIRFLOW_FACTOR):
        badge = 'near tested range (within 3% O2 or a factor of 3 in airflow)'
    else:
        badge = 'extrapolation'
    note = '' if info['limit_found'] else 'no limit found: this material never went out in the tested range'
    return badge, note


def call_label(p, lo, hi, o2, material, threshold):
    """Yes/no call: 'toss-up' when the 90% range straddles the threshold; otherwise 'flame-out warning', or 'burns'
    (marked '(extrapolated)' above the material's highest tested O2)."""
    if lo < threshold <= hi:
        return f"toss-up ({'burns' if p >= threshold else 'flame-out'} by base-rate rule)"
    if p < threshold:
        return 'flame-out warning'
    return 'burns' + (' (extrapolated)' if o2 > STATE['info'][material]['o2_max'] else '')


def predict(material, o2, airflow):
    """P(burn) from the pooled model, its 90% range from the bootstrap, the yes/no call and a trust badge."""
    x = _features(material, o2, airflow)[0]
    p = float(_sigmoid(x @ STATE['model']['coefs']))
    pb = _sigmoid(STATE['boot'] @ x)
    lo, hi = float(np.percentile(pb, 5)), float(np.percentile(pb, 95))
    t = STATE['model']['threshold']
    badge, note = trust_badge(material, float(o2), float(airflow))
    return {'material': material, 'o2_pct': float(o2), 'airflow_cm_s': float(airflow), 'p_burn': p,
            'p_low_5': lo, 'p_high_95': hi, 'call': call_label(p, lo, hi, float(o2), material, t),
            'threshold': t, 'trust_badge': badge, 'note': note, 'caveat': CAVEAT}


# ------------------------------------------------------------------------------
# next experiment (same trust-region logic as the notebook)
# ------------------------------------------------------------------------------
def trust_region(material, radius=RADIUS):
    """One-sided in O2 (never below the lowest tested), airflow clipped to 2-30 cm/s."""
    info = STATE['info'][material]
    return {'o2_low': info['o2_min'], 'o2_high': info['o2_max'] + O2_MARGIN * radius,
            'flow_low': max(NO_DATA_BELOW_CM_S, info['airflow_min'] / AIRFLOW_FACTOR ** radius),
            'flow_high': min(FLOW_MAX_RECOMMEND, info['airflow_max'] * AIRFLOW_FACTOR ** radius)}


def score_candidates(material, o2, airflow):
    """Bootstrap P(burn) mean and 90% range, disagreement, relevance and score for candidate conditions."""
    X = _features(material, o2, airflow)
    P = _sigmoid(STATE['boot'] @ X.T)                       # models x candidates
    floor_o2 = STATE['info'][material]['o2_min']
    ramp = np.clip((X[:, 1] - floor_o2) / (RAMP_TOP_O2 - floor_o2), 0, 1)
    o2_rel = RELEVANCE_FLOOR + (1 - RELEVANCE_FLOOR) * ramp
    lo_b, hi_b = CABIN_FLOW_BAND
    u = np.exp(X[:, 2])
    outside = np.maximum(np.log(lo_b / u), 0) + np.maximum(np.log(u / hi_b), 0)
    flow_rel = RELEVANCE_FLOOR + (1 - RELEVANCE_FLOOR) * np.exp(-0.5 * (outside / np.log(FLOW_WIDTH_FACTOR)) ** 2)
    disagreement = P.std(axis=0)
    return pd.DataFrame({'o2_pct': X[:, 1], 'airflow_cm_s': u, 'p_burn': P.mean(axis=0),
                         'p_low_5': np.percentile(P, 5, axis=0), 'p_high_95': np.percentile(P, 95, axis=0),
                         'disagreement': disagreement, 'relevance': o2_rel * flow_rel,
                         'score': disagreement * o2_rel * flow_rel})


def next_experiment(material, radius=RADIUS):
    """Refine (most disagreement x relevance inside the trust region) and Explore (top O2 edge, cabin band)."""
    region = trust_region(material, radius)
    t = STATE['model']['threshold']
    # Refine: grid inside the region
    o2_values = np.arange(np.ceil(region['o2_low'] / O2_GRID_STEP) * O2_GRID_STEP, region['o2_high'] + 1e-9, O2_GRID_STEP)
    airflows = np.geomspace(region['flow_low'], region['flow_high'], N_AIRFLOWS)
    grid_o2, grid_u = np.repeat(o2_values, len(airflows)), np.tile(airflows, len(o2_values))
    scored = score_candidates(material, grid_o2, grid_u).sort_values('score', ascending=False, kind='mergesort')
    best = scored.iloc[0]
    # Explore: top O2 edge, airflow in the cabin band with the most disagreement
    lo, hi = max(CABIN_FLOW_BAND[0], region['flow_low']), min(CABIN_FLOW_BAND[1], region['flow_high'])
    band = np.geomspace(lo, hi, N_BAND_AIRFLOWS)
    top = score_candidates(material, np.full(len(band), region['o2_high']), band)
    ex = top.loc[top['disagreement'].idxmax()]
    out = {}
    for track, r in [('Refine', best), ('Explore', ex)]:
        p, l, h = float(r['p_burn']), float(r['p_low_5']), float(r['p_high_95'])
        out[track] = {'track': track, 'material': material, 'o2_pct': float(r['o2_pct']),
                      'airflow_cm_s': float(r['airflow_cm_s']), 'pressure_psia': TEST_PRESSURE_PSIA,
                      'platform': PLATFORM, 'p_burn': p, 'p_low_5': l, 'p_high_95': h,
                      'call': call_label(p, l, h, float(r['o2_pct']), material, t),
                      'score': float(r['score']), 'disagreement': float(r['disagreement']),
                      'trust_radius': radius, 'caveat': CAVEAT}
    return out
