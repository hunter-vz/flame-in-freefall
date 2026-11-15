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
REF_LINES = [(21.0, 'ISS air 21%'), (26.5, 'Moon habitat 26.5%'), (34.0, 'Moon habitat 34%')]

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
# Streamlit can re-import fif_model on a rerun (e.g. after a redeploy) while the cached STATE survives;
# put the fitted state back into the module so fif_model.predict() always has it
if fif_model.STATE is not STATE:
    fif_model.STATE.clear()
    fif_model.STATE.update(STATE)
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
    lo, hi = pct(r['p_low_5']), pct(r['p_high_95'])
    rng = lo if lo == hi and lo in ('>99%', '<1%') else f'{lo}–{hi}'
    return f"{pct(r['p_burn'])} chance of burning (90% range: {rng})"


def lowest_burn(material, state=None):
    state = STATE if state is None else state
    u = state['df'][(state['df']['material'] == material) & (state['df']['burned'] == 1)]
    r = u.loc[u['o2_pct'].idxmin()]
    return float(r['o2_pct']), float(r['airflow_cm_s']), str(r['test_id'])


def boundary_o2(material, airflow, coefs=None, state=None):
    """O2 where the model's P(burn) equals the yes/no threshold, at the given airflow(s).
    Uses fif_model's feature rows and coefficients; the logit is linear in O2."""
    state = STATE if state is None else state
    coefs = state['model']['coefs'] if coefs is None else coefs
    t = state['model']['threshold']
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
    layout = dict(template='plotly_dark', paper_bgcolor=BG, plot_bgcolor=BG,
                  font=dict(color='#e6e6e6'), margin=dict(l=10, r=10, t=40, b=10),
                  legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    layout.update(kw)
    fig.update_layout(**layout)
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


def chart_try(material, o2, airflow, info, state=None, show_star=True):
    state = STATE if state is None else state
    pts = state['df'][(state['df']['material'] == material) & ~state['df']['test_id'].isin(session_test_ids())]
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
    new = session_entries()
    new = new[new['material'] == material]
    for outcome, colour in [('burned', C_BURN), ('flame-out', C_OUT)]:
        s_ = new[new['outcome'] == outcome]
        if len(s_):
            fig.add_trace(go.Scatter(x=s_['airflow_cm_s'], y=s_['o2_pct'], mode='markers',
                                     name=f'added result: {outcome}', text=s_['test_id'],
                                     marker=dict(color=colour, size=11, line=dict(color=C_OURS, width=2)),
                                     hovertemplate='%{text} (added): %{y:.1f}% O2, %{x:.3g} cm/s<extra></extra>'))
    u = np.geomspace(0.5, 40, 200)
    b = boundary_o2(material, u, state=state)
    if info['limit_found']:
        fig.add_trace(go.Scatter(x=u, y=b, mode='lines', name='yes/no boundary (model)',
                                 line=dict(color=C_OURS, width=2)))
    else:
        lo_o2 = lowest_burn(material, state)[0]
        fig.add_annotation(xref='paper', yref='paper', x=0.99, y=0.03, xanchor='right', yanchor='bottom',
                           showarrow=False, text=f'No flame-outs: limit unknown, below {lo_o2:g}% (lowest burn)',
                           font=dict(color='#dddddd', size=12), bgcolor='rgba(60,60,60,0.8)')
    if show_star:
        fig.add_trace(go.Scatter(x=[airflow], y=[o2], mode='markers', name='your point',
                                 marker=dict(symbol='star', size=18, color=C_OURS, line=dict(color='black', width=1))))
    fig.update_xaxes(type='log', range=[np.log10(0.5), np.log10(40)], title='Airflow (cm/s, log scale)',
                     tickvals=[0.5, 1, 2, 5, 10, 20, 40], ticktext=['0.5', '1', '2', '5', '10', '20', '40'],
                     minor=dict(showgrid=False, ticks=''))
    fig.update_yaxes(range=[10, 40], title='Oxygen (% O2)')
    return dark_layout(fig, height=500)


def nearest_tests(material, o2, airflow, k=3):
    pts = STATE['df'][STATE['df']['material'] == material].copy()
    pts['dist'] = np.hypot((pts['o2_pct'] - o2) / fif_model.O2_MARGIN,
                           np.log(pts['airflow_cm_s'] / airflow) / np.log(fif_model.AIRFLOW_FACTOR))
    best = pts.sort_values('dist').drop_duplicates('test_id').head(k)
    return pd.DataFrame({'Test ID': best['test_id'], 'O2 (%)': best['o2_pct'].map(lambda v: f'{v:.1f}'),
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
        first = [c for c in ['track', 'material', 'o2_pct', 'airflow_cm_s', 'pressure_psia', 'plain_odds', 'call',
                             'p_burn', 'p_low_5', 'p_high_95'] if c in df.columns]
        df = df[first + [c for c in df.columns if c not in first]]
        df = df.map(lambda v: '—' if pd.isna(v) else (f'{v:g}' if isinstance(v, float) else str(v)))
        key = [c for c in ['track', 'material', 'o2_pct', 'airflow_cm_s', 'pressure_psia', 'plain_odds', 'call']
               if c in df.columns]
        if 'plain_odds' not in df.columns:
            key.insert(key.index('call'), 'p_burn')
        st.table(df[key].map(lambda v: v.replace('>', '\\>').replace('<', '\\<')).rename(columns={'o2_pct': 'O2 (%)', 'airflow_cm_s': 'airflow (cm/s)',
                                         'pressure_psia': 'pressure (psia)'}).set_index('track'))
        with st.expander(f'All columns of {meta["path"].name}'):
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
# PAGE 3: Next experiment
# ------------------------------------------------------------------------------
TRACK_WHY = {
    'Refine': 'where the models disagree most, near today\'s limit',
    'Explore': 'one safe step (3 points of O2) towards Moon air',
    'Bracket': 'finds the limit for a material that has never gone out',
}
REPLAY = pd.DataFrame([  # hard-coded from Flame_in_Freefall.ipynb, section 20 (replay findings)
    ('Our method', 11.2, 4, 16),
    ("NASA's real order", 13.0, None, None),
    ('Spread out (careful human)', 15.4, 12, 18),
    ('Random', 17.9, 6, 26),
    ('Highest O2 first', 23.1, 13, 26),
], columns=['strategy', 'mean', 'lo', 'hi'])
BRACKET_STEP = 1.5   # registry v2 rule: Bracket = 1.5 points of O2 below the lowest burn, same airflow


def bracket_test(material, state=None):
    """Bracket for a material with no flame-outs (registry v2 rule). The model can't predict it."""
    o2, u, tid = lowest_burn(material, state)
    return {'track': 'Bracket', 'o2_pct': o2 - BRACKET_STEP, 'airflow_cm_s': u,
            'odds': 'no prediction (no flame-outs to model)', 'call': 'no prediction',
            'why': f"{TRACK_WHY['Bracket']} (lowest burn {o2:g}% at {u:g} cm/s, {tid}; 1.5 points lower)"}


def experiment_rows(material, mod=fif_model, state=None):
    state = STATE if state is None else state
    rows = []
    if not state['info'][material]['limit_found']:
        return [bracket_test(material, state)]     # no flame-outs: the model's boundary is meaningless here
    for track, r in mod.next_experiment(material).items():
        why = TRACK_WHY[track]
        rows.append({'track': track, 'o2_pct': r['o2_pct'], 'airflow_cm_s': r['airflow_cm_s'],
                     'odds': plain_odds(r), 'call': r['call'], 'why': why})
    return rows


def page_next():
    st.header('Next experiment')
    st.markdown('For each material, the tests we recommend NASA runs next. Each is a SoFIE test on the ISS '
                'at 14.7 psia.')
    for name, why in TRACK_WHY.items():
        st.markdown(f'- **{name}**: {why}')
    for material in MATERIALS:
        st.subheader(material)
        df = pd.DataFrame(experiment_rows(material))
        st.dataframe(pd.DataFrame({'Track': df['track'], 'O2 (%)': df['o2_pct'].round(1),
                                   'Airflow (cm/s)': df['airflow_cm_s'].round(2),
                                   'P(burn) (90% range)': df['odds'], 'Call': df['call'], 'Why': df['why']}),
                     hide_index=True, width='stretch')
    st.caption(CAPTION)
    st.caption('Refine and Explore (rods, thin film) and Bracket (slab, SIBAL) match the frozen registry v2.')

    st.subheader('Does the method work? Replay on NASA\'s real tests')
    st.markdown("**Validated on NASA's real BASS-II rod tests: 27% fewer tests than a careful human "
                "(beat it in 14 of 20 runs, tied 3, lost 3); the obvious 'test near Moon air first' strategy "
                "was worse than random.**")
    d = REPLAY.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=d['strategy'], x=d['mean'], orientation='h',
        marker_color=[C_OURS if s == 'Our method' else C_GREY for s in d['strategy']],
        error_x=dict(type='data', symmetric=False, color='#dddddd', thickness=1.5,
                     array=(d['hi'] - d['mean']).fillna(0), arrayminus=(d['mean'] - d['lo']).fillna(0)),
        text=[f'{m:g}' + (f' ({lo:g}–{hi:g})' if pd.notna(lo) else '') for m, lo, hi in
              zip(d['mean'], d['lo'], d['hi'])],
        textposition='inside', insidetextanchor='start',
        textfont=dict(color=['black' if s == 'Our method' else 'white' for s in d['strategy']]),
        hovertemplate='%{y}: %{x} tests<extra></extra>'))
    fig.update_xaxes(title='Average tests needed to get within 10% of the final error, and stay there '
                           '(lower is better)', range=[0, 28])
    st.plotly_chart(dark_layout(fig, height=330, showlegend=False), width='stretch')
    st.caption("Replay covers Olson's PMMA rods only (≤22% O2). Whiskers: range over 20 runs; NASA's real order "
               "is a single run. Source: Flame_in_Freefall.ipynb, section 20 (hard-coded results).")


# ------------------------------------------------------------------------------
# PAGE 4: Materials map
# ------------------------------------------------------------------------------
@st.cache_data
def boot_coefs():
    return np.array(EXPORT['bootstrap_coefficients'])


def page_map():
    st.header('Materials map')
    airflow = st.select_slider('Airflow (cm/s, log scale)', options=AIRFLOW_STEPS, value=5.0,
                               format_func=lambda v: f'{v:g}')
    x_lo, x_hi = 10.0, 40.0
    fig = go.Figure()
    rows = list(reversed(MATERIALS))
    legend_done = set()

    def seg(y, a, b, colour, opacity, name):
        if b <= a:
            return
        fig.add_trace(go.Bar(y=[y], x=[b - a], base=[a], orientation='h', marker_color=colour, opacity=opacity,
                             name=name, showlegend=name not in legend_done, legendgroup=name,
                             hovertemplate=f'{name}: %{{base:.1f}}–{b:.1f}% O2<extra></extra>'))
        legend_done.add(name)

    summary = []
    for m in rows:
        info = STATE['info'][m]
        if not info['limit_found']:
            lo_o2, _, _ = lowest_burn(m)
            seg(m, info['o2_min'], info['o2_max'], C_GREY, 0.9, 'tested range, never went out')
            fig.add_annotation(y=m, x=info['o2_max'] + 0.3, text=f'never went out; limit is below {lo_o2:g}%',
                               showarrow=False, xanchor='left', font=dict(color='#cccccc', size=12))
            summary.append({'Material': m, 'Yes/no O2 (%)': f'below {lo_o2:g} (never went out)',
                            '90% range': 'n/a', 'Highest O2 tested': info['o2_max']})
            continue
        b = float(boundary_o2(m, airflow)[0])
        bb = np.array([boundary_o2(m, airflow, coefs=c)[0] for c in boot_coefs() if c[1] > 0])
        lo, hi = np.percentile(bb, [5, 95])
        top = info['o2_max']
        seg(m, x_lo, min(b, x_hi), C_OUT, 0.9, 'flame-out (below yes/no O2)')
        seg(m, max(b, x_lo), max(b, min(top, x_hi)), C_BURN, 0.9, 'burns (tested)')
        seg(m, max(b, top), x_hi, C_BURN, 0.45, 'burns (above highest O2 tested)')
        fig.add_trace(go.Scatter(y=[m, m], x=[max(lo, x_lo), min(hi, x_hi)], mode='lines+markers',
                                 line=dict(color=C_OURS, width=2), marker=dict(symbol='line-ns', size=14,
                                 line=dict(color=C_OURS, width=2)), name='90% range of yes/no O2',
                                 showlegend='whisker' not in legend_done, legendgroup='whisker',
                                 hovertemplate=f'90% range: {lo:.1f}–{hi:.1f}% O2<extra></extra>'))
        legend_done.add('whisker')
        fig.add_annotation(y=m, x=b, text=f'{b:.1f}%', showarrow=False, yshift=18, font=dict(color=C_OURS, size=11))
        summary.append({'Material': m, 'Yes/no O2 (%)': f'{b:.1f}', '90% range': f'{lo:.1f}–{hi:.1f}',
                        'Highest O2 tested': top})
    for x, label in REF_LINES:
        fig.add_vline(x=x, line_dash='dot', line_color='#bbbbbb', line_width=1, annotation_text=label,
                      annotation_position='top', annotation_font_color='#bbbbbb', annotation_font_size=11)
    fig.update_xaxes(range=[x_lo, x_hi], title='Oxygen (% O2)')
    fig.update_layout(barmode='overlay', bargap=0.45)
    st.plotly_chart(dark_layout(fig, height=430, legend=dict(orientation='h', y=-0.25, x=0)), width='stretch')
    if airflow < fif_model.NO_DATA_BELOW_CM_S:
        st.warning('Below 2 cm/s: no flame-out data, so the boundary here is not known.')
    st.dataframe(pd.DataFrame(summary).iloc[::-1], hide_index=True, width='stretch')
    st.caption(CAPTION)
    st.caption('White whisker: 5th–95th percentile of the yes/no O2 across the 100 bootstrap models in '
               'model_export.json.')
    st.markdown("""
**Notes**
- The opposed rod limit rests on one test (P8, 3 flame-outs); treat it with care.
- Concurrent rods have the lowest limit: concurrent rods survive at the lowest oxygen in this data.
- Thick slab and SIBAL never went out in NASA's tests, so their limit is somewhere below the lowest burn.
""")


# ------------------------------------------------------------------------------
# PAGE 5: Add new results (session only)
# ------------------------------------------------------------------------------
ENTRY_COLS = ['material', 'o2_pct', 'airflow_cm_s', 'pressure_psia', 'gravity', 'outcome', 'test_id', 'notes']
GRAVITY = ['microgravity', 'normal gravity (1 g)', 'lunar (1/6 g)', 'other']
LIMITS = {'o2_pct': (5.0, 60.0), 'airflow_cm_s': (0.1, 100.0), 'pressure_psia': (0.5, 30.0)}
V1_MATERIAL = {'PMMA rod': 'PMMA plastic: rod (concurrent)'}


def session_entries():
    if 'entries' not in st.session_state:
        st.session_state['entries'] = pd.DataFrame(columns=ENTRY_COLS)
    return st.session_state['entries']


def session_test_ids():
    return set(session_entries()['test_id'].astype(str))


def used_for_training(row):
    return abs(float(row['pressure_psia']) - 14.7) <= 0.05 and row['gravity'] == 'microgravity'


def validate(row, existing_ids):
    errs = []
    if row['material'] not in MATERIALS:
        errs.append(f"unknown material '{row['material']}' (use the names exactly as in model_export.json)")
    for col, (lo, hi) in LIMITS.items():
        try:
            v = float(row[col])
            if not lo <= v <= hi:
                errs.append(f'{col} = {v:g} outside {lo:g}–{hi:g}')
        except (TypeError, ValueError):
            errs.append(f'{col} is not a number')
    if row['outcome'] not in ('burned', 'flame-out'):
        errs.append("outcome must be 'burned' or 'flame-out'")
    tid = str(row['test_id']).strip()
    if not tid or tid == 'nan':
        errs.append('test ID is empty')
    elif tid in existing_ids:
        errs.append(f"test ID '{tid}' already exists")
    return errs


def add_entries(df):
    existing = set(STATE['df']['test_id'].astype(str)) | set(load_all_points()['test_id'].astype(str)) | session_test_ids()
    ok, bad = [], []
    for _, r in df.iterrows():
        r = r.reindex(ENTRY_COLS)
        r['pressure_psia'] = 14.7 if pd.isna(r['pressure_psia']) else r['pressure_psia']
        r['gravity'] = 'microgravity' if pd.isna(r['gravity']) else str(r['gravity']).strip()
        r['outcome'] = str(r['outcome']).strip().lower()
        r['notes'] = '' if pd.isna(r['notes']) else r['notes']
        errs = validate(r, existing)
        if errs:
            bad.append((r['test_id'], errs))
        else:
            r['test_id'] = str(r['test_id']).strip()
            existing.add(r['test_id'])
            ok.append(r)
    if ok:
        new = pd.DataFrame(ok)[ENTRY_COLS].astype({'o2_pct': float, 'airflow_cm_s': float, 'pressure_psia': float})
        st.session_state['entries'] = pd.concat([session_entries(), new], ignore_index=True)
        st.session_state.pop('retrained', None)
    return len(ok), bad


def registry_rows():
    out = []
    for ver, meta in REGISTRIES.items():
        df = pd.read_csv(meta['path'])
        for _, r in df.iterrows():
            call = str(r['call'])
            if call.startswith('no prediction') or pd.isna(r['o2_pct']):
                continue
            if call.startswith('toss-up'):
                predicted = 'flame-out' if 'flame-out by' in call else 'burned'
            else:
                predicted = 'burned' if call.startswith('burns') else 'flame-out'
            out.append({'registry': ver, 'track': r['track'],
                        'material': V1_MATERIAL.get(r['material'], r['material']), 'o2_pct': r['o2_pct'],
                        'airflow_cm_s': r['airflow_cm_s'], 'pressure_psia': r['pressure_psia'],
                        'call': call, 'predicted': predicted})
    return pd.DataFrame(out)


def registry_check(entries):
    reg, rows = registry_rows(), []
    for _, e in entries.iterrows():
        m = reg[(reg['material'] == e['material']) & ((reg['o2_pct'] - e['o2_pct']).abs() <= 0.5)
                & ((e['airflow_cm_s'] / reg['airflow_cm_s'] - 1).abs() <= 0.2)
                & ((reg['pressure_psia'] - e['pressure_psia']).abs() <= 0.05)]
        for _, r in m.iterrows():
            rows.append({'Result': '✓' if r['predicted'] == e['outcome'] else '✗', 'Test ID': e['test_id'],
                         'Outcome': e['outcome'], 'Registry': r['registry'], 'Track': r['track'],
                         'Locked at': f"{r['o2_pct']:g}% O2, {r['airflow_cm_s']:g} cm/s", 'Locked call': r['call']})
    return pd.DataFrame(rows)


def retrain(entries):
    """Append usable entries to all_test_points data and refit through fif_model, in a private copy of the
    module so the shared model other visitors see is untouched."""
    import importlib.util
    import tempfile
    use = entries[entries.apply(used_for_training, axis=1)]
    base = load_all_points()
    new = pd.DataFrame({c: pd.Series(dtype=base[c].dtype) for c in base.columns})
    new = pd.DataFrame({'test_id': use['test_id'], 'pi': 'added', 'material': use['material'],
                        'o2_pct': use['o2_pct'].astype(float), 'airflow_cm_s': use['airflow_cm_s'].astype(float),
                        'burned': (use['outcome'] == 'burned').astype(float), 'outcome': use['outcome'],
                        'weight': 1.0, 'usable_for_model': True, 'note': use['notes']})
    combined = pd.concat([base, new], ignore_index=True)
    spec = importlib.util.spec_from_file_location('fif_model_session', HERE / 'fif_model.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.STATE = {}
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8') as f:
        combined.to_csv(f, index=False)
        tmp = f.name
    mod.init(tmp)
    Path(tmp).unlink(missing_ok=True)
    return mod, len(use)


def model_summary(material, mod, state):
    info = state['info'][material]
    if info['limit_found']:
        lim = float(boundary_o2(material, 5.0, state=state)[0])
    else:
        lim = f"never went out; limit below {lowest_burn(material, state)[0]:g}%"
    ex = experiment_rows(material, mod=mod, state=state)
    nxt = '; '.join(f"{r['track']} {r['o2_pct']:.1f}% at {r['airflow_cm_s']:.3g} cm/s" for r in ex)
    return lim, f"{info['flame_outs']} flame-outs / {info['usable_points']} points", nxt


def page_add():
    st.header('Add new results')
    st.markdown('Enter results from new tests (yours, or NASA\'s when they run our predictions). '
                'Entries stay in **this browser session only**: nothing is written to the repository, '
                'and the frozen registries are never touched.')
    c1, c2 = st.columns(2)
    with c1.form('entry', clear_on_submit=False):
        st.markdown('**One result**')
        material = st.selectbox('Material', MATERIALS)
        a, b = st.columns(2)
        o2 = a.number_input('O2 (%)', 5.0, 60.0, 21.0, 0.1)
        u = b.number_input('Airflow (cm/s)', 0.1, 100.0, 5.0, 0.1)
        pressure = a.number_input('Pressure (psia)', 0.5, 30.0, 14.7, 0.1)
        gravity = b.selectbox('Gravity', GRAVITY)
        outcome = a.radio('Outcome', ['burned', 'flame-out'], horizontal=True)
        tid = b.text_input('Test ID', placeholder='e.g. SoFIE-001')
        notes = st.text_input('Notes')
        if st.form_submit_button('Add result'):
            n, bad = add_entries(pd.DataFrame([dict(zip(ENTRY_COLS, [material, o2, u, pressure, gravity,
                                                                      outcome, tid, notes]))]))
            for t, errs in bad:
                st.error(f"Not added ({t}): " + '; '.join(errs))
            if n:
                st.success('Added.')
    with c2:
        st.markdown('**Or upload a CSV** with columns: ' + ', '.join(f'`{c}`' for c in ENTRY_COLS))
        st.download_button('Template CSV', pd.DataFrame(columns=ENTRY_COLS).to_csv(index=False),
                           'new_results_template.csv', 'text/csv')
        up = st.file_uploader('CSV file', type='csv')
        if up is not None and st.button('Add rows from CSV'):
            try:
                df = pd.read_csv(up)
                missing = [c for c in ['material', 'o2_pct', 'airflow_cm_s', 'outcome', 'test_id'] if c not in df.columns]
                if missing:
                    st.error('Missing columns: ' + ', '.join(missing))
                else:
                    n, bad = add_entries(df)
                    st.success(f'Added {n} row(s).') if n else None
                    for t, errs in bad:
                        st.error(f"Row {t} not added: " + '; '.join(errs))
            except Exception as e:  # noqa: BLE001
                st.error(f'Could not read the CSV: {e}')

    entries = session_entries()
    if entries.empty:
        st.info('No results added yet.')
        return
    st.subheader('Your entries (this session)')
    view = entries.copy()
    view['used for training'] = np.where(entries.apply(used_for_training, axis=1), 'yes', 'no: stored only')
    st.dataframe(view, hide_index=True, width='stretch')
    if (view['used for training'] != 'yes').any():
        st.caption('Entries not at 14.7 psia or not in microgravity are stored but not used for training: every '
                   'NASA point in the model is at ISS cabin pressure in microgravity, and the model has no '
                   'pressure or gravity input, so mixing them in would shift the boundary for the wrong reason.')
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button('Download entries (CSV)', entries.to_csv(index=False), 'flame_in_freefall_new_results.csv',
                       'text/csv')
    if d2.button('Reset (clear entries)'):
        st.session_state.pop('entries', None)
        st.session_state.pop('retrained', None)
        st.rerun()

    st.subheader('Registry check')
    rc = registry_check(entries)
    if rc.empty:
        st.caption('No entry is within 0.5% O2 and 20% airflow of a locked prediction for the same material.')
    else:
        st.dataframe(rc, hide_index=True, width='stretch')
        st.caption('Match rule: same material and pressure, within 0.5 points of O2 and 20% of airflow. '
                   'Toss-ups are scored by their base-rate call.')

    st.subheader('Retrain')
    sel = st.selectbox('Show before/after for', MATERIALS, index=MATERIALS.index(entries['material'].iloc[-1]))
    if st.button('Retrain with my entries', type='primary'):
        with st.spinner('Refitting the model and 100 bootstrap models…'):
            st.session_state['retrained'] = retrain(entries)
    if 'retrained' in st.session_state:
        mod, n_used = st.session_state['retrained']
        st.caption(f'Retrained on the NASA data plus {n_used} of your entries.')
        before = list(model_summary(sel, fif_model, STATE))
        after = list(model_summary(sel, mod, mod.STATE))
        if isinstance(before[0], float) and isinstance(after[0], float):
            d = after[0] - before[0]
            after[0] = f"{before[0]:.2f}% → {after[0]:.2f}% ({d:+.2f})".replace('-', '−')
        else:
            after[0] = after[0] if isinstance(after[0], str) else f'{after[0]:.2f}% O2'
        before[0] = before[0] if isinstance(before[0], str) else f'{before[0]:.2f}% O2' 
        st.dataframe(pd.DataFrame({'': ['Yes/no O2 at 5 cm/s', 'Data', 'Next experiment'],
                                   'Before (NASA data)': before, 'After (with your entries)': after}),
                     hide_index=True, width='stretch')
        st.caption('One new test moves a model built on 60 NASA points only a little; repeated results at the '
                   'same condition move it more.')
        st.plotly_chart(chart_try(sel, 21, 5, mod.STATE['info'][sel], state=mod.STATE, show_star=False),
                        width='stretch')
        st.caption('Points with a white outline are your added results. ' + CAPTION)
    else:
        st.plotly_chart(chart_try(sel, 21, 5, STATE['info'][sel], show_star=False), width='stretch')
        st.caption('Points with a white outline are your added results (not yet in the model). ' + CAPTION)


# ------------------------------------------------------------------------------
# navigation
# ------------------------------------------------------------------------------
PAGES = [
    st.Page(page_try, title='Try it yourself', icon='🔥', default=True),
    st.Page(page_locked, title='Locked predictions', icon='🔒', url_path='locked'),
    st.Page(page_next, title='Next experiment', icon='🧪', url_path='next'),
    st.Page(page_map, title='Materials map', icon='🗺️', url_path='map'),
    st.Page(page_add, title='Add new results', icon='➕', url_path='add'),
]
nav = st.navigation(PAGES)
st.markdown(f"**{HEADLINE}**")
nav.run()
