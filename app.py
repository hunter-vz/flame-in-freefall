"""
Flame in Freefall: Streamlit dashboard.

Uses fif_model.py for all model maths (init, predict, next_experiment).
Reads model_export.json, all_test_points.csv and the two frozen prediction registries (read-only).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fif_model

HERE = Path(__file__).parent
DATA_CSV = HERE / 'all_test_points.csv'
EXPORT_JSON = HERE / 'model_export.json'
REG_V1 = HERE / 'prediction_registry.csv'
REG_V2 = HERE / 'prediction_registry_v2.csv'

# colours (no purple)
C_BURN, C_OUT, C_GREY, C_OURS = 'orange', 'dodgerblue', 'grey', 'white'
BG = '#0e1117'

HEADLINE = ("NASA's microgravity fire data stops at 22% O2. Moon habitat air is 26.5–34% O2. "
            "This tool predicts what happens in between, and tells NASA which test to run next.")
CAPTION = "At ISS cabin pressure (~14.7 psia), microgravity; effect of reduced pressure not in the data."
REF_LINES = [(21.0, 'Earth air 21%'), (26.5, 'Moon habitat 26.5%'), (34.0, 'Moon habitat 34%')]

REGISTRIES = {
    'v1': {'path': REG_V1, 'sha': 'fa8bea740f0177e775cd2b87e874a6ee32ae1ae933315fc4225c67a3f8ca0781',
           'frozen': '2026-11-14T15:27:37Z',
           'commit': 'https://github.com/hunter-vz/flame-in-freefall/commit/d42ba4c',
           'raw': 'https://github.com/hunter-vz/flame-in-freefall/raw/d42ba4c/prediction_registry.csv'},
    'v2': {'path': REG_V2, 'sha': '5d6ed7dfb2f3bace6007286fb480a2d72183d1b9ba9dfd3b5fc133eda0f3d8b3',
           'frozen': '2026-11-15T06:46:41Z',
           'commit': 'https://github.com/hunter-vz/flame-in-freefall/commit/ee38234ecc7d3d5524462d6b56cab0210291b526',
           'raw': 'https://github.com/hunter-vz/flame-in-freefall/raw/ee38234ecc7d3d5524462d6b56cab0210291b526/'
                  'prediction_registry_v2.csv'},
}

st.set_page_config(page_title='Flame in Freefall', page_icon='🔥', layout='wide')


# ------------------------------------------------------------------------------
# model (fitted once per server process through fif_model)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner='Fitting the model and 100 bootstrap models…')
def load_model():
    fif_model.init(str(DATA_CSV))
    return fif_model.STATE


@st.cache_data
def load_export():
    return json.loads(EXPORT_JSON.read_text(encoding='utf-8'))


@st.cache_data
def load_all_points():
    return pd.read_csv(DATA_CSV)


STATE = load_model()
EXPORT = load_export()
MATERIALS = EXPORT['materials']                 # names exactly as in model_export.json


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def pct(p):
    """Probability as plain odds, same style as registry v2."""
    if p > 0.99:
        return '>99%'
    if p < 0.01:
        return '<1%'
    return f'{100 * p:.0f}%'


def plain_odds(r):
    return f"{pct(r['p_burn'])} chance of burning (90% range: {pct(r['p_low_5'])}–{pct(r['p_high_95'])})"


def lowest_burn(material):
    u = STATE['df'][(STATE['df']['material'] == material) & (STATE['df']['burned'] == 1)]
    r = u.loc[u['o2_pct'].idxmin()]
    return float(r['o2_pct']), float(r['airflow_cm_s']), str(r['test_id'])


def boundary_o2(material, airflow, coefs=None):
    """O2 where the model's P(burn) equals the yes/no threshold, at the given airflow(s).
    Uses fif_model's feature rows and coefficients; the logit is linear in O2."""
    coefs = STATE['model']['coefs'] if coefs is None else coefs
    t = STATE['model']['threshold']
    airflow = np.atleast_1d(np.asarray(airflow, float))
    z_without_o2 = fif_model._features(material, 0.0, airflow) @ coefs
    return (np.log(t / (1 - t)) - z_without_o2) / coefs[1]


BADGES = {
    'inside': ('🟢', 'inside the tested range for this material'),
    'near': ('🟡', 'near the tested range: within 3 points of O2 or a factor of 3 in airflow'),
    'extrapolation': ('🔴', 'extrapolation: further from the tested range than that'),
    'low-flow': ('⚪', 'low flow: below 2 cm/s, boundary unknown (no flame-out data)'),
}


def badge_for(trust_text):
    for key, val in BADGES.items():
        if trust_text.startswith(key):
            return val
    return ('🔴', trust_text)


def call_colour(call):
    if call.startswith('burns'):
        return C_BURN
    if call.startswith('flame-out'):
        return C_OUT
    return C_GREY


def add_ref_lines_h(fig):
    for y, label in REF_LINES:
        fig.add_hline(y=y, line_dash='dot', line_color='#bbbbbb', line_width=1,
                      annotation_text=label, annotation_position='top left',
                      annotation_font_color='#bbbbbb', annotation_font_size=11)


def dark_layout(fig, **kw):
    fig.update_layout(template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color='#e6e6e6'), margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0), **kw)
    return fig


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------
# PAGE 1: Try it yourself
# ------------------------------------------------------------------------------
AIRFLOW_STEPS = sorted({float(f'{v:.2g}') for v in np.geomspace(0.5, 40.0, 90)} | {1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0})


def page_try():
    st.header('Try it yourself')
    c1, c2, c3 = st.columns([2, 2, 2])
    material = c1.selectbox('Material', MATERIALS, index=MATERIALS.index('PMMA plastic: thin film'))
    o2 = c2.slider('Oxygen (% O2)', 10.0, 40.0, 21.0, 0.1)
    airflow = c3.select_slider('Airflow (cm/s, log scale)', options=AIRFLOW_STEPS,
                               value=10.0,
                               format_func=lambda v: f'{v:g}')

    r = fif_model.predict(material, o2, airflow)
    info = STATE['info'][material]
    icon, badge_text = badge_for(r['trust_badge'])

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### {plain_odds(r)}")
        st.markdown(f"**Call:** <span style='color:{call_colour(r['call'])}'>{r['call']}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**Trust:** {icon} {badge_text}")
        if not info['limit_found']:
            lo_o2, lo_u, lo_id = lowest_burn(material)
            st.warning(f"No limit found: this material never went out in NASA's tests "
                       f"(lowest burn: {lo_o2:g}% O2). The model can't place its limit. "
                       f"The P(burn) above is an extrapolation.")
        st.caption(CAPTION)
        st.caption(f"Yes/no rule: flame-out warning if P(burn) < {r['threshold']:.3f} "
                   f"(weighted burn base rate). Toss-up when the 90% range straddles it. "
                   f"Tested range for this material: {info['o2_min']:g}–{info['o2_max']:g}% O2, "
                   f"{info['airflow_min']:g}–{info['airflow_max']:g} cm/s.")

    with right:
        st.plotly_chart(chart_try(material, o2, airflow, info), width='stretch')

    st.subheader('3 nearest NASA tests for this material')
    st.dataframe(nearest_tests(material, o2, airflow), hide_index=True, width='stretch')
    st.caption('Nearest by distance in O2 (per 3 points) and airflow (per factor of 3); '
               'closest point from each test, usable points only.')


def chart_try(material, o2, airflow, info):
    pts = STATE['df'][STATE['df']['material'] == material]
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=fif_model.NO_DATA_BELOW_CM_S, fillcolor=C_GREY, opacity=0.25, line_width=0)
    fig.add_annotation(x=0, y=39, text='<2 cm/s:<br>boundary unknown', showarrow=False,
                       font=dict(color='#cccccc', size=10), yanchor='top')
    add_ref_lines_h(fig)
    for burned, name, colour in [(1, 'NASA test: burned', C_BURN), (0, 'NASA test: flame-out', C_OUT)]:
        s = pts[pts['burned'] == burned]
        fig.add_trace(go.Scatter(x=s['airflow_cm_s'], y=s['o2_pct'], mode='markers', name=name,
                                 marker=dict(color=colour, size=8, line=dict(width=0)),
                                 text=s['test_id'],
                                 hovertemplate='%{text}: %{y:.1f}% O2, %{x:.3g} cm/s<extra></extra>'))
    u = np.geomspace(0.5, 40, 200)
    b = boundary_o2(material, u)
    if info['limit_found']:
        fig.add_trace(go.Scatter(x=u, y=b, mode='lines', name='yes/no boundary (model)',
                                 line=dict(color=C_OURS, width=2)))
    else:
        fig.add_trace(go.Scatter(x=u, y=b, mode='lines', name='yes/no boundary (extrapolated: no flame-outs)',
                                 line=dict(color=C_GREY, width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=[airflow], y=[o2], mode='markers', name='your point',
                             marker=dict(symbol='star', size=18, color=C_OURS, line=dict(color='black', width=1))))
    fig.update_xaxes(type='log', range=[np.log10(0.5), np.log10(40)], title='Airflow (cm/s, log scale)')
    fig.update_yaxes(range=[10, 40], title='Oxygen (% O2)')
    return dark_layout(fig, height=500)


def nearest_tests(material, o2, airflow, k=3):
    pts = STATE['df'][STATE['df']['material'] == material].copy()
    pts['dist'] = np.hypot((pts['o2_pct'] - o2) / fif_model.O2_MARGIN,
                           np.log(pts['airflow_cm_s'] / airflow) / np.log(fif_model.AIRFLOW_FACTOR))
    best = pts.sort_values('dist').drop_duplicates('test_id').head(k)
    return pd.DataFrame({'Test ID': best['test_id'], 'O2 (%)': best['o2_pct'].round(2),
                         'Airflow (cm/s)': best['airflow_cm_s'].round(2),
                         'Outcome': np.where(best['burned'] == 1, 'burned', 'flame-out')})


# ------------------------------------------------------------------------------
# PAGE 2: Locked predictions
# ------------------------------------------------------------------------------
def page_locked():
    st.header('Locked predictions')
    st.markdown("We froze these predictions **before any of the tests are run**: each registry was committed "
                "to GitHub with its SHA-256 fingerprint. Change a single character and the fingerprint changes. "
                "When NASA runs the tests, anyone can compare the real outcomes with our calls and score us.")
    for ver, meta in REGISTRIES.items():
        st.subheader(f'Registry {ver}')
        live = sha256_of(meta['path'])
        ok = live == meta['sha']
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"Frozen **{meta['frozen']}** · commit: [{meta['commit'].rsplit('/', 1)[-1][:7]}]"
                        f"({meta['commit']}) · file: `{meta['path'].name}`")
            st.markdown(f"Published SHA-256: `{meta['sha']}`  \nLive SHA-256 of the file this app reads: `{live}`")
        with c2:
            if ok:
                st.success('✓ matches: unchanged since freezing')
            else:
                st.error('✗ does not match')
        df = pd.read_csv(meta['path'])
        first = [c for c in ['track', 'material', 'o2_pct', 'airflow_cm_s', 'pressure_psia', 'plain_odds',
                             'p_burn', 'p_low_5', 'p_high_95', 'call'] if c in df.columns]
        df = df[first + [c for c in df.columns if c not in first]]
        st.dataframe(df, hide_index=True, width='stretch')
        st.caption(CAPTION)

    st.subheader('How to check us')
    st.markdown(f"""
1. Download the CSV as it was frozen: open the commit link ([v1]({REGISTRIES['v1']['commit']}),
   [v2]({REGISTRIES['v2']['commit']})), click the file, then **Download raw file**. Direct links:
   [v1 raw]({REGISTRIES['v1']['raw']}), [v2 raw]({REGISTRIES['v2']['raw']}).
2. Compute its fingerprint:
   - Windows: `certutil -hashfile prediction_registry.csv SHA256`
   - Mac/Linux: `shasum -a 256 prediction_registry.csv`
3. Compare with the SHA-256 shown above (and in the commit message). Same fingerprint = same file, unchanged.
""")


# ------------------------------------------------------------------------------
# navigation
# ------------------------------------------------------------------------------
PAGES = [
    st.Page(page_try, title='Try it yourself', icon='🔥', default=True),
    st.Page(page_locked, title='Locked predictions', icon='🔒', url_path='locked'),
]
nav = st.navigation(PAGES)
st.markdown(f"**{HEADLINE}**")
nav.run()
