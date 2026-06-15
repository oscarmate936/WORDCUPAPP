import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import plotly.graph_objects as go

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

.stApp {
    background: #0a0f0d;
    color: #e8f5e9;
}

[data-testid="stSidebar"] {
    background: #0d1a12 !important;
    border-right: 1px solid #1e3a24;
}

[data-testid="stSidebar"] * {
    color: #c8e6c9 !important;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
h1 { color: #69f0ae; letter-spacing: -1px; }
h2 { color: #a5d6a7; }
h3 { color: #81c784; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; }

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

.info-box {
    background: #0d1a12;
    border-left: 3px solid #69f0ae;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
    color: #a5d6a7;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Core model ────────────────────────────────────────────────────────────────
MAX_GOALS = 10

def poisson_matrix(xg1, xg2, n=MAX_GOALS):
    ph = [poisson.pmf(k, xg1) for k in range(n)]
    pa = [poisson.pmf(k, xg2) for k in range(n)]
    return np.outer(ph, pa)

def calc_1x2(mat):
    home = float(np.sum(np.tril(mat, -1)))
    draw = float(np.trace(mat))
    away = float(np.sum(np.triu(mat, 1)))
    return home, draw, away

def calc_double_chance(h, d, a):
    return h + d, h + a, d + a

def calc_dnb(h, d, a):
    base = h + a
    return h / base, a / base

def calc_asian_handicap(mat, handicap):
    n = mat.shape[0]
    home, draw, away = 0.0, 0.0, 0.0
    for i in range(n):
        for j in range(n):
            result = i + handicap - j
            if result > 0:
                home += mat[i, j]
            elif result == 0:
                draw += mat[i, j]
            else:
                away += mat[i, j]
    return home, draw, away

def calc_over_under(mat, line):
    n = mat.shape[0]
    over = 0.0
    for i in range(n):
        for j in range(n):
            if i + j > line:
                over += mat[i, j]
    return over, 1 - over

def calc_btts(xg1, xg2):
    p1 = 1 - poisson.pmf(0, xg1)
    p2 = 1 - poisson.pmf(0, xg2)
    yes = p1 * p2
    return yes, 1 - yes

def calc_exact_goals(mat, max_shown=9):
    totals = {}
    n = mat.shape[0]
    for g in range(max_shown + 1):
        p = sum(mat[i, g - i] for i in range(min(g + 1, n)) if g - i < n)
        totals[g] = p
    return totals

def calc_asian_ou(xg1, xg2, line):
    mat = poisson_matrix(xg1, xg2, MAX_GOALS)
    n = mat.shape[0]
    if line == 2.25:
        over_lower = sum(mat[i, j] for i in range(n) for j in range(n) if i + j > 2)
        push_lower = sum(mat[i, j] for i in range(n) for j in range(n) if i + j == 2)
        over_upper = sum(mat[i, j] for i in range(n) for j in range(n) if i + j >= 3)
        over_win = 0.5 * (over_lower + 0.5 * push_lower) + 0.5 * over_upper
        return over_win, 1 - over_win
    elif line == 2.75:
        over_lower = sum(mat[i, j] for i in range(n) for j in range(n) if i + j >= 3)
        over_upper = sum(mat[i, j] for i in range(n) for j in range(n) if i + j >= 4)
        push_upper = sum(mat[i, j] for i in range(n) for j in range(n) if i + j == 3)
        over_win = 0.5 * over_lower + 0.5 * (over_upper + 0.5 * push_upper)
        return over_win, 1 - over_win
    else:
        over, under = calc_over_under(mat, line)
        return over, under

def calc_margin_of_victory(mat):
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
    return 3 * h + d, 3 * a + d


# ── Sidebar con botón normal y keys ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Parámetros del Partido")
    st.markdown("---")

    team1 = st.text_input("Equipo 1", value="Brasil", key="team1")
    team2 = st.text_input("Equipo 2", value="Argentina", key="team2")

    st.markdown("### xG del Partido")
    xg1_raw = st.number_input(f"xG {team1}", min_value=0.10, max_value=6.0,
                               value=1.45, step=0.05, format="%.2f", key="xg1_raw")
    xg2_raw = st.number_input(f"xG {team2}", min_value=0.10, max_value=6.0,
                               value=1.20, step=0.05, format="%.2f", key="xg2_raw")

    st.markdown("### Torneo Actual")
    avg_total_tournament = st.number_input(
        "Promedio de goles general en la Copa del Mundo",
        min_value=1.0, max_value=5.0, value=2.52, step=0.01,
        help="Media de goles totales por partido en el torneo que estás analizando",
        key="avg_total"
    )

    # Botón de análisis (fuera de cualquier formulario)
    run_analysis = st.button("⚡ Calcular Análisis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#4caf50;'>Modelo: Poisson bivariada con regresión a la media<br>"
        "xG ajustado = (xG_partido + 2 · (promedio_general/2)) / 3</div>",
        unsafe_allow_html=True,
    )


# ── Lógica de cálculo al pulsar el botón ────────────────────────────────────
if run_analysis:
    k = 2.0
    avg_team_prior = st.session_state.avg_total / 2.0

    xg1 = (st.session_state.xg1_raw + k * avg_team_prior) / (1 + k)
    xg2 = (st.session_state.xg2_raw + k * avg_team_prior) / (1 + k)

    mat = poisson_matrix(xg1, xg2)
    h, d, a = calc_1x2(mat)
    dc_1x, dc_12, dc_x2 = calc_double_chance(h, d, a)
    dnb1, dnb2 = calc_dnb(h, d, a)

    over05, under05 = calc_over_under(mat, 0.5)
    over15, under15 = calc_over_under(mat, 1.5)
    over25, under25 = calc_over_under(mat, 2.5)
    over35, under35 = calc_over_under(mat, 3.5)
    over45, under45 = calc_over_under(mat, 4.5)

    asian_225_o, asian_225_u = calc_asian_ou(xg1, xg2, 2.25)
    asian_275_o, asian_275_u = calc_asian_ou(xg1, xg2, 2.75)

    btts_y, btts_n = calc_btts(xg1, xg2)

    exact = calc_exact_goals(mat, max_shown=8)
    margins = calc_margin_of_victory(mat)
    exp1, exp2 = calc_expected_points(h, d, a)

    best_i, best_j = np.unravel_index(np.argmax(mat), mat.shape)

    # Guardar SOLO resultados en session_state (sin tocar las keys de widgets)
    st.session_state.analysis_done = True
    st.session_state.xg1_adj = xg1
    st.session_state.xg2_adj = xg2
    st.session_state.mat = mat
    st.session_state.h = h
    st.session_state.d = d
    st.session_state.a = a
    st.session_state.dc_1x = dc_1x
    st.session_state.dc_12 = dc_12
    st.session_state.dc_x2 = dc_x2
    st.session_state.dnb1 = dnb1
    st.session_state.dnb2 = dnb2
    st.session_state.over05 = over05
    st.session_state.under05 = under05
    st.session_state.over15 = over15
    st.session_state.under15 = under15
    st.session_state.over25 = over25
    st.session_state.under25 = under25
    st.session_state.over35 = over35
    st.session_state.under35 = under35
    st.session_state.over45 = over45
    st.session_state.under45 = under45
    st.session_state.asian_225_o = asian_225_o
    st.session_state.asian_225_u = asian_225_u
    st.session_state.asian_275_o = asian_275_o
    st.session_state.asian_275_u = asian_275_u
    st.session_state.btts_y = btts_y
    st.session_state.btts_n = btts_n
    st.session_state.exact = exact
    st.session_state.margins = margins
    st.session_state.exp1 = exp1
    st.session_state.exp2 = exp2
    st.session_state.best_i = best_i
    st.session_state.best_j = best_j


# ── Mostrar resultados si el análisis ya se ha hecho ────────────────────────
if st.session_state.get("analysis_done", False):
    # Recuperar inputs originales desde las keys de los widgets
    team1 = st.session_state.team1
    team2 = st.session_state.team2
    xg1_raw = st.session_state.xg1_raw
    xg2_raw = st.session_state.xg2_raw
    avg_total = st.session_state.avg_total

    # Recuperar resultados calculados
    xg1 = st.session_state.xg1_adj
    xg2 = st.session_state.xg2_adj
    mat = st.session_state.mat
    h = st.session_state.h
    d = st.session_state.d
    a = st.session_state.a
    dc_1x = st.session_state.dc_1x
    dc_12 = st.session_state.dc_12
    dc_x2 = st.session_state.dc_x2
    dnb1 = st.session_state.dnb1
    dnb2 = st.session_state.dnb2
    over05 = st.session_state.over05
    under05 = st.session_state.under05
    over15 = st.session_state.over15
    under15 = st.session_state.under15
    over25 = st.session_state.over25
    under25 = st.session_state.under25
    over35 = st.session_state.over35
    under35 = st.session_state.under35
    over45 = st.session_state.over45
    under45 = st.session_state.under45
    asian_225_o = st.session_state.asian_225_o
    asian_225_u = st.session_state.asian_225_u
    asian_275_o = st.session_state.asian_275_o
    asian_275_u = st.session_state.asian_275_u
    btts_y = st.session_state.btts_y
    btts_n = st.session_state.btts_n
    exact = st.session_state.exact
    margins = st.session_state.margins
    exp1 = st.session_state.exp1
    exp2 = st.session_state.exp2
    best_i = st.session_state.best_i
    best_j = st.session_state.best_j

    # ── Título ─────────────────────────────────────────────────────────────
    st.markdown("# Copa del Mundo · Análisis Estadístico xG")
    st.markdown(
        "<div class='info-box'>Probabilidades calculadas con Poisson bivariada. "
        "xG ajustados hacia el promedio del torneo (sin ventaja de localía).</div>",
        unsafe_allow_html=True,
    )

    # ── Cabeceras de equipos ───────────────────────────────────────────────
    col1, col_vs, col2 = st.columns([5, 2, 5])
    with col1:
        st.markdown(f"""
        <div class='team-header' style='flex-direction:column;align-items:flex-start;'>
            <div class='metric-label'>Equipo 1</div>
            <div class='team-name'>{team1}</div>
            <div class='team-xg'>{xg1:.2f} <span style='font-size:1rem;color:#388e3c'>xG ajust</span></div>
            <div class='metric-sub'>xG original: {xg1_raw:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col_vs:
        st.markdown(f"""
        <div style='display:flex;align-items:center;justify-content:center;height:100%;padding-top:30px;'>
            <div class='vs-badge'>VS</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='team-header' style='flex-direction:column;align-items:flex-end;'>
            <div class='metric-label'>Equipo 2</div>
            <div class='team-name'>{team2}</div>
            <div class='team-xg'>{xg2:.2f} <span style='font-size:1rem;color:#388e3c'>xG ajust</span></div>
            <div class='metric-sub'>xG original: {xg2_raw:.2f}</div>
        </div>""", unsafe_allow_html=True)

    # ── 1X2 ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Resultado Final — 1X2</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{h:.1%}</div>
            <div class='metric-label'>Victoria {team1}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{d:.1%}</div>
            <div class='metric-label'>Empate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{a:.1%}</div>
            <div class='metric-label'>Victoria {team2}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{best_i}–{best_j}</div>
            <div class='metric-label'>Marcador más probable</div>
            <div class='metric-sub'>{mat[best_i, best_j]:.1%} de prob.</div>
        </div>""", unsafe_allow_html=True)

    fig_1x2 = go.Figure(go.Bar(
        x=[f"Victoria {team1}", "Empate", f"Victoria {team2}"],
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
        margin=dict(t=20, b=10, l=10, r=10), height=220,
    )
    st.plotly_chart(fig_1x2, use_container_width=True)

    # ── Doble oportunidad + DNB ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Doble Oportunidad · Draw No Bet</div>", unsafe_allow_html=True)
    col_dc, col_dnb = st.columns(2)
    with col_dc:
        st.markdown("**Doble Oportunidad**")
        for label, val in [(f"1X  ({team1} o Empate)", dc_1x),
                           (f"12  (Cualquier equipo gana)", dc_12),
                           (f"X2  (Empate o {team2})", dc_x2)]:
            st.markdown(f"""<div class='metric-card' style='margin-bottom:8px;text-align:left;'>
                <span style='font-family:Space Mono;color:#69f0ae;font-size:1.1rem;font-weight:700;'>{val:.1%}</span>
                <span style='color:#81c784;font-size:0.82rem;margin-left:10px;'>{label}</span>
            </div>""", unsafe_allow_html=True)
    with col_dnb:
        st.markdown("**Draw No Bet**")
        for label, val in [(f"{team1} gana (sin empate)", dnb1),
                           (f"{team2} gana (sin empate)", dnb2)]:
            st.markdown(f"""<div class='metric-card' style='margin-bottom:8px;text-align:left;'>
                <span style='font-family:Space Mono;color:#69f0ae;font-size:1.1rem;font-weight:700;'>{val:.1%}</span>
                <span style='color:#81c784;font-size:0.82rem;margin-left:10px;'>{label}</span>
            </div>""", unsafe_allow_html=True)

    # ── Hándicap asiático ─────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Hándicap Asiático</div>", unsafe_allow_html=True)
    handicap = st.select_slider(
        f"Selecciona el hándicap de {team1}:",
        options=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        value=0.0,
        format_func=lambda x: f"{x:+.1f}",
        key="handicap"
    )
    hc_h, hc_d, hc_a = calc_asian_handicap(mat, handicap)
    col_hc1, col_hc2, col_hc3 = st.columns(3)
    with col_hc1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{hc_h:.1%}</div>
            <div class='metric-label'>{team1} (hándicap {handicap:+.1f})</div>
        </div>""", unsafe_allow_html=True)
    with col_hc2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{hc_d:.1%}</div>
            <div class='metric-label'>Empate con hándicap</div>
        </div>""", unsafe_allow_html=True)
    with col_hc3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{hc_a:.1%}</div>
            <div class='metric-label'>{team2} (hándicap {handicap:+.1f})</div>
        </div>""", unsafe_allow_html=True)

    # ── Over/Under ────────────────────────────────────────────────────────
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

    # ── BTTS ──────────────────────────────────────────────────────────────
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

    combos = [
        (f"{team1} gana + BTTS Sí", h * btts_y),
        (f"{team1} gana + BTTS No", h * btts_n),
        (f"Empate + BTTS Sí", d * btts_y),
        (f"Empate + BTTS No", d * btts_n),
        (f"{team2} gana + BTTS Sí", a * btts_y),
        (f"{team2} gana + BTTS No", a * btts_n),
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

    # ── Goles exactos ─────────────────────────────────────────────────────
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

    st.markdown("**Rango de Goles (Multigoal)**")
    mg_cols = st.columns(4)
    ranges = [(0, 1, "0–1 goles"), (2, 3, "2–3 goles"), (3, 4, "3–4 goles"), (2, 4, "2–4 goles")]
    for col, (lo, hi, label) in zip(mg_cols, ranges):
        p = calc_multigoal(mat, lo, hi)
        col.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{p:.1%}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    # ── Heatmap ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Heatmap de Marcadores Exactos</div>", unsafe_allow_html=True)
    SHOW = 7
    z = mat[:SHOW, :SHOW] * 100
    text_mat = [[f"{z[i][j]:.1f}%" for j in range(SHOW)] for i in range(SHOW)]
    fig_heat = go.Figure(go.Heatmap(
        z=z,
        x=[f"{team2} {j}" for j in range(SHOW)],
        y=[f"{team1} {i}" for i in range(SHOW)],
        colorscale=[[0, "#0a1a0d"], [0.3, "#1b5e20"], [0.7, "#388e3c"], [1.0, "#69f0ae"]],
        text=text_mat,
        texttemplate="%{text}",
        textfont=dict(family="Space Mono", size=11),
        showscale=True,
        colorbar=dict(tickfont=dict(family="Space Mono", color="#81c784"), bgcolor="#0d1a12", bordercolor="#1e3a24"),
    ))
    fig_heat.update_layout(
        plot_bgcolor="#0d1a12", paper_bgcolor="#0a0f0d",
        font=dict(family="Space Grotesk", color="#a5d6a7"),
        xaxis=dict(title=f"Goles {team2}", showgrid=False),
        yaxis=dict(title=f"Goles {team1}", showgrid=False),
        margin=dict(t=20, b=20, l=20, r=20), height=400,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Margen de victoria ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Margen de Victoria</div>", unsafe_allow_html=True)
    mg_labels, mg_values, mg_colors = [], [], []
    for diff, p in sorted(margins.items()):
        if diff == 0:
            mg_labels.append("Empate")
            mg_colors.append("#ffd54f")
        elif diff > 0:
            mg_labels.append(f"{team1} +{diff}")
            mg_colors.append("#69f0ae")
        else:
            mg_labels.append(f"{team2} +{abs(diff)}")
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

    # ── xPts ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Puntos Esperados (xPts)</div>", unsafe_allow_html=True)
    ep1, ep2 = st.columns(2)
    with ep1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{exp1:.2f}</div>
            <div class='metric-label'>xPts — {team1}</div>
            <div class='metric-sub'>En escala de liga (0–3 pts)</div>
        </div>""", unsafe_allow_html=True)
    with ep2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{exp2:.2f}</div>
            <div class='metric-label'>xPts — {team2}</div>
            <div class='metric-sub'>En escala de liga (0–3 pts)</div>
        </div>""", unsafe_allow_html=True)

    # ── Tabla resumen ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Resumen Completo de Mercados</div>", unsafe_allow_html=True)
    summary = {
        "Mercado": [
            f"Victoria {team1}", "Empate", f"Victoria {team2}",
            "Doble oportunidad 1X", "Doble oportunidad 12", "Doble oportunidad X2",
            f"DNB {team1}", f"DNB {team2}",
            "Over 0.5", "Under 0.5", "Over 1.5", "Under 1.5",
            "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5",
            "Over 4.5", "Under 4.5", "Asian O/U 2.25 Over", "Asian O/U 2.75 Over",
            "BTTS Sí", "BTTS No",
            f"Marcador más probable ({best_i}–{best_j})",
            f"xPts {team1}", f"xPts {team2}",
        ],
        "Probabilidad / Valor": [
            f"{h:.1%}", f"{d:.1%}", f"{a:.1%}",
            f"{dc_1x:.1%}", f"{dc_12:.1%}", f"{dc_x2:.1%}",
            f"{dnb1:.1%}", f"{dnb2:.1%}",
            f"{over05:.1%}", f"{under05:.1%}", f"{over15:.1%}", f"{under15:.1%}",
            f"{over25:.1%}", f"{under25:.1%}", f"{over35:.1%}", f"{under35:.1%}",
            f"{over45:.1%}", f"{under45:.1%}", f"{asian_225_o:.1%}", f"{asian_275_o:.1%}",
            f"{btts_y:.1%}", f"{btts_n:.1%}",
            f"{mat[best_i,best_j]:.1%}",
            f"{exp1:.2f} pts", f"{exp2:.2f} pts",
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

    st.markdown("""
    <div style='text-align:center;padding:32px 0 16px;color:#2e7d32;font-size:0.75rem;font-family:Space Mono,monospace;'>
        Modelo Poisson Bivariada · Solo uso estadístico · Copa del Mundo xG Analyzer
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Introduce los datos en la barra lateral y pulsa **'Calcular Análisis'** para ver los resultados.")