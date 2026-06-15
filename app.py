import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copa del Mundo · Análisis xG",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark pitch background */
.stApp {
    background: #0a0f0d;
    color: #e8f5e9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1a12 !important;
    border-right: 1px solid #1e3a24;
}

[data-testid="stSidebar"] * {
    color: #c8e6c9 !important;
}

/* Headers */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
h1 { color: #69f0ae; letter-spacing: -1px; }
h2 { color: #a5d6a7; }
h3 { color: #81c784; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; }

/* Metric cards */
.metric-card {
    background: #0d1a12;
    border: 1px solid #1e3a24;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #69f0ae; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #69f0ae;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #81c784;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}
.metric-sub {
    font-size: 0.85rem;
    color: #4caf50;
    margin-top: 4px;
}

/* Section divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #388e3c;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid #1e3a24;
    padding-bottom: 8px;
    margin: 32px 0 16px 0;
}

/* Team header */
.team-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0d1a12;
    border: 1px solid #1e3a24;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
}
.team-name {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8f5e9;
}
.team-xg {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: #69f0ae;
}
.vs-badge {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #388e3c;
    background: #1a2e1e;
    padding: 8px 14px;
    border-radius: 6px;
}

/* Warning / info boxes */
.info-box {
    background: #0d1a12;
    border-left: 3px solid #69f0ae;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
    color: #a5d6a7;
    margin: 12px 0;
}

/* Scoreline table */
.score-cell-high { background: #1b5e20 !important; color: #e8f5e9 !important; font-weight: 700; }
.score-cell-mid  { background: #2e7d32 !important; color: #c8e6c9 !important; }
.score-cell-low  { background: #0d1a12 !important; color: #4caf50 !important; }
</style>
""", unsafe_allow_html=True)


# ── Core model ────────────────────────────────────────────────────────────────
MAX_GOALS = 8  # grid size for exact-score matrix

def poisson_matrix(xg_home: float, xg_away: float, n: int = MAX_GOALS) -> np.ndarray:
    """Joint probability matrix P[i,j] = P(home=i, away=j)."""
    ph = [poisson.pmf(k, xg_home) for k in range(n)]
    pa = [poisson.pmf(k, xg_away) for k in range(n)]
    return np.outer(ph, pa)


def calc_1x2(mat: np.ndarray):
    home = float(np.sum(np.tril(mat, -1)))
    draw = float(np.trace(mat))
    away = float(np.sum(np.triu(mat, 1)))
    return home, draw, away


def calc_double_chance(h, d, a):
    return h + d, h + a, d + a


def calc_dnb(h, d, a):
    """Draw No Bet — renormalize excluding draw."""
    base = h + a
    return h / base, a / base


def calc_handicap(xg_home, xg_away, handicap: float):
    """Asian handicap: adjust home xG and recalc 1X2."""
    return calc_1x2(poisson_matrix(xg_home + handicap, xg_away))


def calc_over_under(mat: np.ndarray, line: float):
    n = mat.shape[0]
    over = 0.0
    for i in range(n):
        for j in range(n):
            if i + j > line:
                over += mat[i, j]
    return over, 1 - over


def calc_btts(xg_home, xg_away):
    p_home_scores = 1 - poisson.pmf(0, xg_home)
    p_away_scores = 1 - poisson.pmf(0, xg_away)
    yes = p_home_scores * p_away_scores
    return yes, 1 - yes


def calc_exact_goals(mat: np.ndarray, max_shown: int = 7):
    totals = {}
    n = mat.shape[0]
    for g in range(max_shown + 1):
        p = sum(mat[i, g - i] for i in range(min(g + 1, n)) if g - i < n)
        totals[g] = p
    return totals


def calc_asian_ou(xg_home, xg_away, line: float):
    """Split-line Asian O/U (e.g. 2.25 = half on 2, half on 2.5)."""
    mat = poisson_matrix(xg_home, xg_away)
    lo = int(line - 0.25)
    hi = lo + 1
    over_lo, under_lo = calc_over_under(mat, lo)
    over_hi, under_hi = calc_over_under(mat, hi)
    return (over_lo + over_hi) / 2, (under_lo + under_hi) / 2


def calc_margin_of_victory(mat: np.ndarray):
    n = mat.shape[0]
    margins = {}
    for diff in range(-5, 6):
        p = 0.0
        for i in range(n):
            j = i - diff
            if 0 <= j < n:
                p += mat[i, j]
        margins[diff] = p
    return margins


def calc_multigoal(mat, lo, hi):
    n = mat.shape[0]
    p = 0.0
    for i in range(n):
        for j in range(n):
            if lo <= i + j <= hi:
                p += mat[i, j]
    return p


def calc_expected_points(h, d, a):
    exp_h = 3 * h + d
    exp_a = 3 * a + d
    return exp_h, exp_a


# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Parámetros del Partido")
    st.markdown("---")

    xg_home = st.number_input("xG Equipo 1 (Local)", min_value=0.10, max_value=6.0,
                               value=1.45, step=0.05, format="%.2f")
    xg_away = st.number_input("xG Equipo 2 (Visitante)", min_value=0.10, max_value=6.0,
                               value=1.20, step=0.05, format="%.2f")
    avg_goals = st.number_input("Media de goles Copa del Mundo", min_value=0.5,
                                 max_value=6.0, value=2.52, step=0.01, format="%.2f",
                                 help="Media de goles totales por partido en la edición actual")

    st.markdown("---")
    run = st.button("⚽ Analizar Partido", use_container_width=True, type="primary")

    st.markdown(
        "<div style='font-size:0.75rem;color:#4caf50;margin-top:12px;'>Modelo: Poisson Bivariada<br>"
        "λ₁ = xG local · λ₂ = xG visitante</div>",
        unsafe_allow_html=True,
    )

# Nombres fijos internamente
home_team = "Equipo 1"
away_team = "Equipo 2"

# Bloquea el análisis hasta que se pulse el botón
if not run and "analyzed" not in st.session_state:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:60vh;text-align:center;'>
        <div style='font-size:4rem;margin-bottom:16px;'>⚽</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;color:#69f0ae;font-weight:700;'>
            Copa del Mundo · Análisis xG
        </div>
        <div style='color:#4caf50;margin-top:10px;font-size:0.95rem;'>
            Introduce los tres parámetros en la barra lateral<br>y pulsa <b>Analizar Partido</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Marca que ya se analizó (para no bloquear en reruns tras cambiar inputs)
st.session_state["analyzed"] = True

# Hándicap asiático fijo en 0 (sin control de usuario)
handicap = 0.0


# ── Compute everything ────────────────────────────────────────────────────────
mat = poisson_matrix(xg_home, xg_away)
h, d, a = calc_1x2(mat)
dc_1x, dc_12, dc_x2 = calc_double_chance(h, d, a)
dnb_h, dnb_a = calc_dnb(h, d, a)
hc_h, hc_d, hc_a = calc_handicap(xg_home, xg_away, handicap)

over05, under05 = calc_over_under(mat, 0.5)
over15, under15 = calc_over_under(mat, 1.5)
over25, under25 = calc_over_under(mat, 2.5)
over35, under35 = calc_over_under(mat, 3.5)
over45, under45 = calc_over_under(mat, 4.5)

asian_225_o, asian_225_u = calc_asian_ou(xg_home, xg_away, 2.25)
asian_275_o, asian_275_u = calc_asian_ou(xg_home, xg_away, 2.75)

btts_y, btts_n = calc_btts(xg_home, xg_away)

exact = calc_exact_goals(mat)
margins = calc_margin_of_victory(mat)
exp_h, exp_a = calc_expected_points(h, d, a)

# ── Most likely score ─────────────────────────────────────────────────────────
best_i, best_j = np.unravel_index(np.argmax(mat), mat.shape)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# Copa del Mundo · Análisis Estadístico xG")
st.markdown(
    f"<div class='info-box'>Probabilidades calculadas con Poisson Bivariada · "
    f"λ₁ = <b>{xg_home:.2f}</b> xG (local) · λ₂ = <b>{xg_away:.2f}</b> xG (visitante) · "
    f"Media goles Copa actual: <b>{avg_goals:.2f}</b>/partido · "
    f"xG total del partido: <b>{xg_home + xg_away:.2f}</b></div>",
    unsafe_allow_html=True,
)

# ── Team header ───────────────────────────────────────────────────────────────
col_l, col_vs, col_r = st.columns([5, 2, 5])
with col_l:
    st.markdown(f"""
    <div class='team-header' style='flex-direction:column;align-items:flex-start;'>
        <div class='metric-label'>Local</div>
        <div class='team-name'>{home_team}</div>
        <div class='team-xg'>{xg_home:.2f} <span style='font-size:1rem;color:#388e3c'>xG</span></div>
    </div>""", unsafe_allow_html=True)
with col_vs:
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:100%;padding-top:20px;gap:8px;'>
        <div class='vs-badge'>VS</div>
        <div style='font-size:0.7rem;color:#388e3c;text-align:center;font-family:Space Mono,monospace;'>
            Media Copa<br>{avg_goals:.2f} goles
        </div>
    </div>""", unsafe_allow_html=True)
with col_r:
    st.markdown(f"""
    <div class='team-header' style='flex-direction:column;align-items:flex-end;'>
        <div class='metric-label'>Visitante</div>
        <div class='team-name'>{away_team}</div>
        <div class='team-xg'>{xg_away:.2f} <span style='font-size:1rem;color:#388e3c'>xG</span></div>
    </div>""", unsafe_allow_html=True)


# ── 1X2 ───────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Resultado Final — 1X2</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{h:.1%}</div>
        <div class='metric-label'>Victoria {home_team}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{d:.1%}</div>
        <div class='metric-label'>Empate</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{a:.1%}</div>
        <div class='metric-label'>Victoria {away_team}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{best_i}–{best_j}</div>
        <div class='metric-label'>Marcador más probable</div>
        <div class='metric-sub'>{mat[best_i, best_j]:.1%} de prob.</div>
    </div>""", unsafe_allow_html=True)

# 1X2 bar chart
fig_1x2 = go.Figure(go.Bar(
    x=[f"Victoria {home_team}", "Empate", f"Victoria {away_team}"],
    y=[h * 100, d * 100, a * 100],
    marker_color=["#69f0ae", "#ffd54f", "#ef5350"],
    text=[f"{v:.1f}%" for v in [h*100, d*100, a*100]],
    textposition="outside",
    textfont=dict(family="Space Mono", size=13, color="#e8f5e9"),
))
fig_1x2.update_layout(
    plot_bgcolor="#0d1a12", paper_bgcolor="#0a0f0d",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(h, d, a)*130]),
    xaxis=dict(showgrid=False),
    margin=dict(t=20, b=10, l=10, r=10),
    height=220,
)
st.plotly_chart(fig_1x2, use_container_width=True)


# ── Double chance + DNB + Handicap ───────────────────────────────────────────
st.markdown("<div class='section-title'>Doble Oportunidad · Draw No Bet · Hándicap Asiático</div>",
            unsafe_allow_html=True)

col_dc, col_dnb, col_hc = st.columns(3)

with col_dc:
    st.markdown("**Doble Oportunidad**")
    for label, val in [(f"1X  ({home_team} o Empate)", dc_1x),
                        (f"12  (Cualquier equipo gana)", dc_12),
                        (f"X2  (Empate o {away_team})", dc_x2)]:
        st.markdown(f"""<div class='metric-card' style='margin-bottom:8px;text-align:left;'>
            <span style='font-family:Space Mono;color:#69f0ae;font-size:1.1rem;font-weight:700;'>{val:.1%}</span>
            <span style='color:#81c784;font-size:0.82rem;margin-left:10px;'>{label}</span>
        </div>""", unsafe_allow_html=True)

with col_dnb:
    st.markdown("**Draw No Bet**")
    for label, val in [(f"{home_team} gana (sin empate)", dnb_h),
                        (f"{away_team} gana (sin empate)", dnb_a)]:
        st.markdown(f"""<div class='metric-card' style='margin-bottom:8px;text-align:left;'>
            <span style='font-family:Space Mono;color:#69f0ae;font-size:1.1rem;font-weight:700;'>{val:.1%}</span>
            <span style='color:#81c784;font-size:0.82rem;margin-left:10px;'>{label}</span>
        </div>""", unsafe_allow_html=True)

with col_hc:
    st.markdown(f"**Hándicap Asiático (sin hándicap)**")
    for label, val in [(f"{home_team} (hándicap {handicap:+.1f})", hc_h),
                        ("Empate con hándicap", hc_d),
                        (f"{away_team} (hándicap {handicap:+.1f})", hc_a)]:
        st.markdown(f"""<div class='metric-card' style='margin-bottom:8px;text-align:left;'>
            <span style='font-family:Space Mono;color:#69f0ae;font-size:1.1rem;font-weight:700;'>{val:.1%}</span>
            <span style='color:#81c784;font-size:0.82rem;margin-left:10px;'>{label}</span>
        </div>""", unsafe_allow_html=True)


# ── Over / Under ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Mercado Over / Under</div>", unsafe_allow_html=True)

ou_data = {
    "Línea": ["0.5", "1.5", "2.5", "3.5", "4.5"],
    "Over %": [over05*100, over15*100, over25*100, over35*100, over45*100],
    "Under %": [under05*100, under15*100, under25*100, under35*100, under45*100],
}

fig_ou = go.Figure()
fig_ou.add_trace(go.Bar(name="Over", x=ou_data["Línea"], y=ou_data["Over %"],
                         marker_color="#69f0ae", text=[f"{v:.1f}%" for v in ou_data["Over %"]],
                         textposition="auto", textfont=dict(family="Space Mono", size=11)))
fig_ou.add_trace(go.Bar(name="Under", x=ou_data["Línea"], y=ou_data["Under %"],
                         marker_color="#1b5e20", text=[f"{v:.1f}%" for v in ou_data["Under %"]],
                         textposition="auto", textfont=dict(family="Space Mono", size=11)))
fig_ou.update_layout(
    barmode="group", plot_bgcolor="#0d1a12", paper_bgcolor="#0a0f0d",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    legend=dict(bgcolor="#0d1a12", bordercolor="#1e3a24", borderwidth=1),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
    xaxis=dict(title="Línea de goles totales", showgrid=False),
    margin=dict(t=10, b=10, l=10, r=10), height=260,
)
st.plotly_chart(fig_ou, use_container_width=True)

# Asian O/U split-line
col_a1, col_a2 = st.columns(2)
with col_a1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Asian O/U 2.25 (split 2 / 2.5)</div>
        <div style='display:flex;justify-content:space-around;margin-top:10px;'>
            <div><div class='metric-value'>{asian_225_o:.1%}</div><div class='metric-label'>Over</div></div>
            <div><div class='metric-value'>{asian_225_u:.1%}</div><div class='metric-label'>Under</div></div>
        </div>
    </div>""", unsafe_allow_html=True)
with col_a2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Asian O/U 2.75 (split 2.5 / 3)</div>
        <div style='display:flex;justify-content:space-around;margin-top:10px;'>
            <div><div class='metric-value'>{asian_275_o:.1%}</div><div class='metric-label'>Over</div></div>
            <div><div class='metric-value'>{asian_275_u:.1%}</div><div class='metric-label'>Under</div></div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── BTTS ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Ambos Equipos Marcan (BTTS)</div>", unsafe_allow_html=True)
col_b1, col_b2 = st.columns(2)
with col_b1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:#69f0ae;'>{btts_y:.1%}</div>
        <div class='metric-label'>Sí — Ambos marcan</div>
    </div>""", unsafe_allow_html=True)
with col_b2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:#ef5350;'>{btts_n:.1%}</div>
        <div class='metric-label'>No — Al menos uno no marca</div>
    </div>""", unsafe_allow_html=True)

# BTTS + 1X2 combinado
st.markdown("**BTTS combinado con resultado**")
combos = [
    (f"{home_team} gana + BTTS Sí", h * btts_y),
    (f"{home_team} gana + BTTS No", h * btts_n),
    (f"Empate + BTTS Sí", d * btts_y),
    (f"Empate + BTTS No", d * btts_n),
    (f"{away_team} gana + BTTS Sí", a * btts_y),
    (f"{away_team} gana + BTTS No", a * btts_n),
]
combo_df = pd.DataFrame(combos, columns=["Escenario", "Probabilidad"])
combo_df["Prob %"] = combo_df["Probabilidad"].apply(lambda x: f"{x:.1%}")
combo_df = combo_df.sort_values("Probabilidad", ascending=False)

fig_combo = go.Figure(go.Bar(
    x=combo_df["Probabilidad"] * 100,
    y=combo_df["Escenario"],
    orientation="h",
    marker_color="#2e7d32",
    marker_line_color="#69f0ae",
    marker_line_width=1,
    text=combo_df["Prob %"],
    textposition="outside",
    textfont=dict(family="Space Mono", size=11, color="#e8f5e9"),
))
fig_combo.update_layout(
    plot_bgcolor="#0d1a12", paper_bgcolor="#0a0f0d",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=False),
    margin=dict(t=10, b=10, l=10, r=80), height=280,
)
st.plotly_chart(fig_combo, use_container_width=True)


# ── Exact goals ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Goles Totales Exactos</div>", unsafe_allow_html=True)

labels = [str(k) for k in exact.keys()]
values = [v * 100 for v in exact.values()]
fig_eg = go.Figure(go.Bar(
    x=labels, y=values,
    marker_color="#388e3c",
    marker_line_color="#69f0ae", marker_line_width=1,
    text=[f"{v:.1f}%" for v in values],
    textposition="outside",
    textfont=dict(family="Space Mono", size=11, color="#e8f5e9"),
))
fig_eg.update_layout(
    plot_bgcolor="#0d1a12", paper_bgcolor="#0a0f0d",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    xaxis=dict(title="Goles totales en el partido", showgrid=False),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(values)*1.3]),
    margin=dict(t=10, b=10, l=10, r=10), height=240,
)
st.plotly_chart(fig_eg, use_container_width=True)

# Multigoal ranges
st.markdown("**Rango de Goles (Multigoal)**")
mg_cols = st.columns(4)
ranges = [(0, 1, "0–1 goles"), (2, 3, "2–3 goles"), (3, 4, "3–4 goles"), (2, 4, "2–4 goles")]
for col, (lo, hi, label) in zip(mg_cols, ranges):
    p = calc_multigoal(mat, lo, hi)
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{p:.1%}</div>
        <div class='metric-label'>{label}</div>
    </div>""", unsafe_allow_html=True)


# ── Exact scoreline heatmap ───────────────────────────────────────────────────
st.markdown("<div class='section-title'>Heatmap de Marcadores Exactos</div>", unsafe_allow_html=True)

SHOW = 6
z = mat[:SHOW, :SHOW] * 100
text_mat = [[f"{z[i][j]:.1f}%" for j in range(SHOW)] for i in range(SHOW)]

fig_heat = go.Figure(go.Heatmap(
    z=z,
    x=[f"{away_team} {j}" for j in range(SHOW)],
    y=[f"{home_team} {i}" for i in range(SHOW)],
    colorscale=[[0, "#0a1a0d"], [0.3, "#1b5e20"], [0.7, "#388e3c"], [1.0, "#69f0ae"]],
    text=text_mat,
    texttemplate="%{text}",
    textfont=dict(family="Space Mono", size=11),
    showscale=True,
    colorbar=dict(
        tickfont=dict(family="Space Mono", color="#81c784"),
        bgcolor="#0d1a12",
        bordercolor="#1e3a24",
    ),
))
fig_heat.update_layout(
    plot_bgcolor="#0d1a12", paper_bgcolor="#0a0f0d",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    xaxis=dict(title=f"Goles {away_team}", showgrid=False),
    yaxis=dict(title=f"Goles {home_team}", showgrid=False),
    margin=dict(t=20, b=20, l=20, r=20), height=400,
)
st.plotly_chart(fig_heat, use_container_width=True)


# ── Margin of victory ─────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Margen de Victoria</div>", unsafe_allow_html=True)

mg_labels, mg_values, mg_colors = [], [], []
for diff, p in sorted(margins.items()):
    if diff == 0:
        mg_labels.append("Empate")
        mg_colors.append("#ffd54f")
    elif diff > 0:
        mg_labels.append(f"{home_team} +{diff}")
        mg_colors.append("#69f0ae")
    else:
        mg_labels.append(f"{away_team} +{abs(diff)}")
        mg_colors.append("#ef5350")
    mg_values.append(p * 100)

fig_mg = go.Figure(go.Bar(
    x=mg_labels, y=mg_values,
    marker_color=mg_colors,
    text=[f"{v:.1f}%" for v in mg_values],
    textposition="outside",
    textfont=dict(family="Space Mono", size=10, color="#e8f5e9"),
))
fig_mg.update_layout(
    plot_bgcolor="#0d1a12", paper_bgcolor="#0a0f0d",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(mg_values)*1.3]),
    margin=dict(t=10, b=10, l=10, r=10), height=260,
)
st.plotly_chart(fig_mg, use_container_width=True)


# ── Expected Points ───────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Puntos Esperados (xPts)</div>", unsafe_allow_html=True)
ep1, ep2 = st.columns(2)
with ep1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{exp_h:.2f}</div>
        <div class='metric-label'>xPts — {home_team}</div>
        <div class='metric-sub'>En escala de liga (0–3 pts)</div>
    </div>""", unsafe_allow_html=True)
with ep2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{exp_a:.2f}</div>
        <div class='metric-label'>xPts — {away_team}</div>
        <div class='metric-sub'>En escala de liga (0–3 pts)</div>
    </div>""", unsafe_allow_html=True)


# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Resumen Completo de Mercados</div>", unsafe_allow_html=True)

summary = {
    "Mercado": [
        f"Victoria {home_team}", "Empate", f"Victoria {away_team}",
        f"Doble oportunidad 1X", "Doble oportunidad 12", "Doble oportunidad X2",
        f"DNB {home_team}", f"DNB {away_team}",
        "Over 0.5", "Under 0.5", "Over 1.5", "Under 1.5",
        "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5",
        "Over 4.5", "Under 4.5", "Asian O/U 2.25 Over", "Asian O/U 2.75 Over",
        "BTTS Sí", "BTTS No",
        f"Marcador más probable ({best_i}–{best_j})",
        f"xPts {home_team}", f"xPts {away_team}",
    ],
    "Probabilidad / Valor": [
        f"{h:.1%}", f"{d:.1%}", f"{a:.1%}",
        f"{dc_1x:.1%}", f"{dc_12:.1%}", f"{dc_x2:.1%}",
        f"{dnb_h:.1%}", f"{dnb_a:.1%}",
        f"{over05:.1%}", f"{under05:.1%}", f"{over15:.1%}", f"{under15:.1%}",
        f"{over25:.1%}", f"{under25:.1%}", f"{over35:.1%}", f"{under35:.1%}",
        f"{over45:.1%}", f"{under45:.1%}", f"{asian_225_o:.1%}", f"{asian_275_o:.1%}",
        f"{btts_y:.1%}", f"{btts_n:.1%}",
        f"{mat[best_i,best_j]:.1%}",
        f"{exp_h:.2f} pts", f"{exp_a:.2f} pts",
    ],
}
df_summary = pd.DataFrame(summary)
st.dataframe(
    df_summary,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Mercado": st.column_config.TextColumn("Mercado"),
        "Probabilidad / Valor": st.column_config.TextColumn("Probabilidad / Valor"),
    },
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:32px 0 16px;color:#2e7d32;font-size:0.75rem;font-family:Space Mono,monospace;'>
    Modelo Poisson Bivariada · xG local {xg_home:.2f} · xG visitante {xg_away:.2f} · Media Copa {avg_goals:.2f} goles/partido
</div>
""", unsafe_allow_html=True)
