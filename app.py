import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copa del Mundo · Análisis xG",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
}

.stApp {
    background: #121212;
    color: #E0E0E0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1E1E1E !important;
    border-right: 1px solid #2C2C2C;
}
[data-testid="stSidebar"] * {
    color: #BDBDBD !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #4CAF50 !important;
    color: #FFFFFF !important;
    font-family: 'Roboto', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    margin-top: 16px !important;
    box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #66BB6A !important;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.5);
}

/* ── Section header ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 36px 0 8px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #2C2C2C;
}
.sec-icon {
    font-size: 1.2rem;
    background: rgba(76,175,80,0.15);
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
}
.sec-label {
    font-family: 'Roboto', sans-serif;
    font-size: 0.85rem;
    color: #4CAF50;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.sec-desc {
    font-size: 0.85rem;
    color: #9E9E9E;
    margin: 0 0 20px 0;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: #1E1E1E;
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    margin-bottom: 8px;
    transition: transform 0.2s;
}
.card:active {
    transform: scale(0.98);
}
.card-winner {
    border: 2px solid #4CAF50;
}
.card-val {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.5rem, 5vw, 2.4rem);
    font-weight: 700;
    color: #4CAF50;
    line-height: 1.2;
}
.card-val-sm {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.2rem, 4vw, 1.8rem);
    font-weight: 700;
    color: #4CAF50;
    line-height: 1.2;
}
.card-val-red { color: #EF5350 !important; }
.card-val-yellow { color: #FFC107 !important; }
.card-lbl {
    font-size: 0.75rem;
    color: #BDBDBD;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}
.card-sub {
    font-size: 0.75rem;
    color: #757575;
    margin-top: 4px;
}
.card-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    background: #2C2C2C;
    color: #4CAF50;
    padding: 4px 10px;
    border-radius: 6px;
    margin-top: 8px;
}

/* ── Progress bar ── */
.prob-bar-wrap { margin: 8px 0 12px; }
.prob-bar-row {
    display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
}
.prob-bar-lbl {
    font-size: 0.8rem; color: #BDBDBD; min-width: 100px;
}
.prob-bar-track {
    flex: 1; height: 8px; background: #2C2C2C; border-radius: 4px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 4px; transition: width 0.4s;
}
.prob-bar-num {
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    color: #4CAF50; min-width: 50px; text-align: right;
}

/* ── Table-style rows ── */
.mkt-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid #2C2C2C;
    border-radius: 0;
}
.mkt-row:first-child { border-radius: 12px 12px 0 0; }
.mkt-row:last-child  { border-bottom: none; border-radius: 0 0 12px 12px; }
.mkt-row:hover { background: #2C2C2C; }
.mkt-block {
    background: #1E1E1E; border: 1px solid #2C2C2C; border-radius: 12px;
    margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.mkt-name { font-size: 0.9rem; color: #E0E0E0; }
.mkt-desc { font-size: 0.75rem; color: #757575; margin-top: 2px; }
.mkt-val {
    font-family: 'Space Mono', monospace; font-size: 1rem;
    font-weight: 700; color: #4CAF50;
}
.mkt-badge {
    font-size: 0.65rem; font-family: 'Space Mono', monospace;
    background: #2C2C2C; color: #4CAF50;
    padding: 3px 8px; border-radius: 4px; margin-left: 8px;
}

/* ── Resumen section ── */
.resumen-cat-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: #181818;
    border-radius: 12px 12px 0 0;
    border-bottom: 1px solid #2C2C2C;
    font-family: 'Roboto', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #4CAF50;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
.resumen-cat-icon {
    font-size: 1rem;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(76,175,80,0.12);
    border-radius: 6px;
}
.resumen-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 16px;
    border-bottom: 1px solid #252525;
    transition: background 0.15s;
}
.resumen-row:hover { background: #252525; }
.resumen-row:last-child { border-bottom: none; }
.resumen-name {
    font-size: 0.85rem;
    color: #E0E0E0;
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
}
.resumen-name .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.resumen-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #4CAF50;
    text-align: right;
    min-width: 80px;
    margin-left: 12px;
}
.resumen-val-sm {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: #BDBDBD;
    text-align: right;
    min-width: 80px;
    margin-left: 12px;
}
.resumen-block {
    background: #1E1E1E;
    border: 1px solid #2C2C2C;
    border-radius: 14px;
    margin-bottom: 10px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}

/* ── Match header ── */
.match-header {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0;
    background: #1E1E1E;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.team-block {
    padding: 24px 20px;
    text-align: center;
}
.team-block-away {
    text-align: center;
}
.team-role {
    font-size: 0.7rem;
    color: #757575;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.team-name {
    font-size: clamp(1.2rem, 4vw, 1.8rem);
    font-weight: 700;
    color: #FFFFFF;
    margin: 4px 0 12px;
}
.team-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: #4CAF50;
    border-radius: 50%;
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #121212;
    margin-bottom: 8px;
}
.team-xg {
    font-family: 'Space Mono', monospace;
    font-size: clamp(2rem, 6vw, 3rem);
    font-weight: 700;
    color: #4CAF50;
    line-height: 1;
}
.team-xg-lbl {
    font-size: 0.7rem;
    color: #757575;
    margin-top: 4px;
}
.vs-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0 16px;
    border-left: 1px solid #2C2C2C;
    border-right: 1px solid #2C2C2C;
    background: #181818;
}
.vs-txt {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #757575;
}
.vs-total {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #4CAF50;
    margin: 4px 0;
}
.vs-copa {
    font-size: 0.65rem;
    color: #757575;
    text-align: center;
    margin-top: 4px;
}

/* ── Info pill ── */
.info-pill {
    background: #1E1E1E;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #9E9E9E;
    margin-bottom: 24px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.info-pill span {
    color: #4CAF50;
    font-family: 'Space Mono', monospace;
    font-weight: 500;
}

/* ── OU pill grid ── */
.ou-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
    gap: 8px;
    margin-bottom: 12px;
}
.ou-card {
    background: #1E1E1E;
    border-radius: 12px;
    padding: 16px 8px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.ou-line {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #E0E0E0;
    font-weight: 700;
    margin-bottom: 8px;
}
.ou-over { font-family: 'Space Mono', monospace; font-size: 1.2rem; color: #4CAF50; font-weight: 700; }
.ou-under { font-family: 'Space Mono', monospace; font-size: 1.2rem; color: #EF5350; font-weight: 700; }
.ou-tag-o { font-size: 0.7rem; color: #4CAF50; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0; }
.ou-tag-u { font-size: 0.7rem; color: #EF5350; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0; }

/* ─── BTTS gauge ─── */
.btts-wrap { display: flex; gap: 12px; flex-wrap: wrap; }
.btts-card {
    flex: 1; min-width: 140px; background: #1E1E1E; border-radius: 16px; padding: 20px 16px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.btts-yes { border: 2px solid #4CAF50; }
.btts-no  { border: 2px solid #EF5350; }

/* ─── Top scorelines ─── */
.top10-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
    gap: 8px;
    margin-bottom: 16px;
}
.top10-card {
    background: #1E1E1E;
    border-radius: 12px;
    padding: 16px 8px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.top10-card-1 { border: 2px solid #4CAF50; }
.top10-score {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #4CAF50;
}
.top10-prob { font-size: 0.8rem; color: #BDBDBD; margin-top: 6px; }
.top10-rank {
    font-size: 0.65rem;
    color: #757575;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* ─── Multigoal grid ─── */
.mg-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 8px;
    margin-bottom: 16px;
}
.mg-card {
    background: #1E1E1E;
    border-radius: 12px;
    padding: 16px 10px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* ─── xPts bar ─── */
.xpts-row {
    display: flex; align-items: center; gap: 12px; padding: 12px 0;
}
.xpts-team { font-size: 0.9rem; color: #E0E0E0; min-width: 80px; }
.xpts-track { flex: 1; height: 14px; background: #2C2C2C; border-radius: 7px; overflow: hidden; }
.xpts-fill  { height: 100%; background: #4CAF50; border-radius: 7px; }
.xpts-val   { font-family: 'Space Mono', monospace; font-size: 1rem; color: #4CAF50; min-width: 45px; text-align: right; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #121212; }
::-webkit-scrollbar-thumb { background: #2C2C2C; border-radius: 3px; }

/* Mobile */
@media (max-width: 640px) {
    .card, .btts-card, .mg-card, .top10-card, .ou-card { padding: 14px 10px; }
    .team-name { font-size: 1.1rem; }
    .team-xg { font-size: 1.8rem; }
    .match-header { border-radius: 16px; }
}
</style>
""", unsafe_allow_html=True)


# ── Model functions (unchanged) ───────────────────────────────────────────────
MAX_G = 25

def dixon_coles_matrix(lh, la, rho, n=MAX_G):
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    p_i = poisson.pmf(i, lh)
    p_j = poisson.pmf(j, la)
    indep = p_i * p_j
    tau = np.ones((n, n))
    tau[0, 0] = 1 - lh * la * rho
    tau[0, 1] = 1 + lh * rho
    tau[1, 0] = 1 + la * rho
    tau[1, 1] = 1 - rho
    if np.any(tau < 0):
        max_rho = 1 / (lh * la) if lh * la > 0 else 0.0
        min_rho = -1 / max(lh, la) if max(lh, la) > 0 else 0.0
        safe_rho = max(min(rho, max_rho), min_rho)
        tau[0, 0] = 1 - lh * la * safe_rho
        tau[0, 1] = 1 + lh * safe_rho
        tau[1, 0] = 1 + la * safe_rho
        tau[1, 1] = 1 - safe_rho
        if "rho_warning" not in st.session_state:
            st.warning(
                f"Con λ₁={lh:.2f}, λ₂={la:.2f}, el ρ seleccionado ({rho:.2f}) genera factores τ negativos. "
                f"Se ha ajustado automáticamente a {safe_rho:.2f} para mantener la validez del modelo.",
                icon="⚠️"
            )
            st.session_state["rho_warning"] = True
    mat = indep * tau
    mat /= mat.sum()
    return mat


def calc_1x2(mat):
    h = float(np.sum(np.tril(mat, -1)))
    d = float(np.trace(mat))
    a = float(np.sum(np.triu(mat, 1)))
    return h, d, a


def calc_ou(mat, line):
    idx = np.indices(mat.shape)
    totals = idx[0] + idx[1]
    over = np.sum(mat[totals > line])
    return float(over), float(1 - over)


def calc_asian_ou_full(mat, line):
    lo = line - 0.25
    hi = line + 0.25
    over_lo, _ = calc_ou(mat, lo)
    over_hi, _ = calc_ou(mat, hi)
    exact = calc_exact_total(mat, max_g=MAX_G-1)
    total_probs = {t: p for t, p in exact.items()}
    def prob_range(low, high):
        return sum(total_probs.get(t, 0.0) for t in range(low, high+1))
    if line % 1 == 0.75:
        full_win = prob_range(int(hi)+1, MAX_G)
        half_win = total_probs.get(int(hi), 0.0)
        loss = 1.0 - full_win - half_win
        return {"full_win": full_win, "half_win": half_win, "loss": loss}
    else:
        full_win = prob_range(int(hi)+1, MAX_G)
        half_loss = total_probs.get(int(hi), 0.0)
        loss = 1.0 - full_win - half_loss
        return {"full_win": full_win, "half_loss": half_loss, "loss": loss}


def calc_exact_total(mat, max_g=15):
    n = mat.shape[0]
    return {
        g: float(sum(mat[i, g-i] for i in range(min(g+1, n)) if g-i < n))
        for g in range(max_g+1)
    }


def calc_margin(mat):
    n = mat.shape[0]
    return {
        d: float(sum(mat[i, i-d] for i in range(n) if 0 <= i-d < n))
        for d in range(-5, 6)
    }


def calc_multigoal(mat, lo, hi):
    idx = np.indices(mat.shape)
    totals = idx[0] + idx[1]
    return float(np.sum(mat[(totals >= lo) & (totals <= hi)]))


def top_scorelines(mat, k=10):
    n = mat.shape[0]
    cells = [(mat[i, j], i, j) for i in range(n) for j in range(n)]
    cells.sort(reverse=True)
    return cells[:k]


def bar_html(pct, color="#4CAF50", height=8):
    return (
        f"<div style='height:{height}px;background:#2C2C2C;border-radius:4px;overflow:hidden;margin-top:8px;'>"
        f"<div style='width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;'></div></div>"
    )


def prob_color(p):
    if p >= 0.55: return "#4CAF50"
    if p >= 0.35: return "#FFC107"
    return "#EF5350"


def match_rating(xg_total, btts_yes, over25):
    score = min(xg_total / 4.0, 1.0) * 40 + btts_yes * 30 + over25 * 30
    if score >= 80: return "★★★★★"
    if score >= 60: return "★★★★☆"
    if score >= 40: return "★★★☆☆"
    if score >= 20: return "★★☆☆☆"
    return "★☆☆☆☆"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Parámetros")
    st.markdown("---")
    team1 = st.text_input("Nombre Equipo 1 (Local)", value="Equipo 1")
    team2 = st.text_input("Nombre Equipo 2 (Visitante)", value="Equipo 2")
    st.markdown("---")
    xg_home = st.number_input(
        f"xG {team1} · Local",
        min_value=0.10, max_value=6.0, value=1.45, step=0.05, format="%.2f",
        help="Goles esperados del equipo local según el modelo xG",
    )
    xg_away = st.number_input(
        f"xG {team2} · Visitante",
        min_value=0.10, max_value=6.0, value=1.20, step=0.05, format="%.2f",
        help="Goles esperados del equipo visitante según el modelo xG",
    )
    avg_goals = st.number_input(
        "Media goles/partido del torneo",
        min_value=0.5, max_value=6.0, value=2.52, step=0.01, format="%.2f",
        help="Promedio de goles totales por partido en la edición actual de la Copa del Mundo",
    )
    adjust_avg = st.checkbox(
        "Ajustar total de goles a la media del torneo",
        value=True,
        help="Escala los xG para que la suma coincida exactamente con la media histórica.",
    )
    rho = st.slider(
        "Correlación ρ (modelo Dixon‑Coles)",
        min_value=-0.30, max_value=0.0, value=-0.10, step=0.01,
        help="Parámetro de dependencia entre los goles de ambos equipos.",
    )
    st.markdown("---")
    run = st.button("⚽ Analizar Partido", use_container_width=True, type="primary")
    st.markdown(
        "<div style='font-size:0.75rem;color:#757575;margin-top:12px;line-height:1.6;'>"
        "Modelo · Dixon & Coles (Poisson bivariada)<br>Correlación ajustable · Calibración por media del torneo</div>",
        unsafe_allow_html=True,
    )


# ── Gate ──────────────────────────────────────────────────────────────────────
if not run and "ready" not in st.session_state:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:65vh;text-align:center;gap:12px;'>
        <div style='font-size:5rem;'>⚽</div>
        <div style='font-family:Roboto;font-size:1.5rem;font-weight:700;color:#4CAF50;'>
            Copa del Mundo · Análisis xG
        </div>
        <div style='color:#9E9E9E;font-size:0.9rem;max-width:320px;line-height:1.7;'>
            Ingresa los nombres, xG de ambos equipos y la media de goles<br>
            de la Copa del Mundo en la barra lateral,<br>
            luego pulsa <b style='color:#4CAF50;'>Analizar Partido</b>.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

st.session_state["ready"] = True


# ── Compute ───────────────────────────────────────────────────────────────────
total_xg_input = xg_home + xg_away
if adjust_avg and total_xg_input > 0:
    scale = avg_goals / total_xg_input
    lambda_h = xg_home * scale
    lambda_a = xg_away * scale
else:
    lambda_h = xg_home
    lambda_a = xg_away

@st.cache_data(show_spinner=False)
def build_matrix(lh, la, rho):
    return dixon_coles_matrix(lh, la, rho, MAX_G)

mat = build_matrix(lambda_h, lambda_a, rho)

h, d, a = calc_1x2(mat)
dc_1x = h + d
dc_12 = h + a
dc_x2 = d + a
dnb_h = h / (h + a) if (h + a) > 0 else 0.0
dnb_a = a / (h + a) if (h + a) > 0 else 0.0

over05, under05 = calc_ou(mat, 0.5)
over15, under15 = calc_ou(mat, 1.5)
over25, under25 = calc_ou(mat, 2.5)
over35, under35 = calc_ou(mat, 3.5)
over45, under45 = calc_ou(mat, 4.5)

asian_225 = calc_asian_ou_full(mat, 2.25)
asian_275 = calc_asian_ou_full(mat, 2.75)
asian_325 = calc_asian_ou_full(mat, 3.25)

btts_y = float(np.sum(mat[1:, 1:]))
btts_n = 1.0 - btts_y

home_btts_yes = float(np.sum([mat[i, j] for i in range(1, MAX_G) for j in range(1, MAX_G) if i > j]))
draw_btts_yes = float(np.sum([mat[i, i] for i in range(1, MAX_G)]))
away_btts_yes = float(np.sum([mat[i, j] for i in range(1, MAX_G) for j in range(1, MAX_G) if j > i]))

exact   = calc_exact_total(mat, max_g=15)
margins = calc_margin(mat)
top10   = top_scorelines(mat, 10)

exp_h   = 3 * h + d
exp_a   = 3 * a + d

xg_total = lambda_h + lambda_a
diff_vs_avg = xg_total - avg_goals
rating = match_rating(xg_total, btts_y, over25)

# Auxiliares hándicap asiático
win_by_1 = margins.get(1, 0.0)
win_by_2plus = h - win_by_1
lose_by_1 = margins.get(-1, 0.0)
lose_by_2plus = a - lose_by_1

ah_minus025_win = h
ah_minus025_halfloss = d
ah_minus025_loss = a
ah_plus025_win = a
ah_plus025_halfloss = d
ah_plus025_loss = h
ah_minus05_win = h
ah_minus05_loss = d + a
ah_plus05_win = d + a
ah_plus05_loss = h
ah_minus075_fullwin = win_by_2plus
ah_minus075_halfwin = win_by_1
ah_minus075_loss = d + a
ah_plus075_fullwin = lose_by_2plus
ah_plus075_halfwin = lose_by_1
ah_plus075_loss = d + h


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# Copa del Mundo &nbsp;·&nbsp; Análisis Estadístico xG", unsafe_allow_html=True)

context_arrow = "▲" if diff_vs_avg > 0 else "▼"
context_color = "#4CAF50" if diff_vs_avg > 0 else "#EF5350"
context_txt   = "partido con más goles que la media" if diff_vs_avg > 0 else "partido con menos goles que la media"

st.markdown(f"""
<div class='info-pill'>
    <div>Modelo <span>Dixon & Coles</span></div>
    <div>xG orig. {team1} <span>{xg_home:.2f}</span></div>
    <div>xG orig. {team2} <span>{xg_away:.2f}</span></div>
    <div>xG calib. {team1} <span>{lambda_h:.2f}</span></div>
    <div>xG calib. {team2} <span>{lambda_a:.2f}</span></div>
    <div>Total <span>{xg_total:.2f}</span></div>
    <div>Media <span>{avg_goals:.2f}</span></div>
    <div style='color:{context_color};'>{context_arrow} {abs(diff_vs_avg):.2f} — {context_txt}</div>
    <div style='font-size:1.1rem;margin-left:auto;color:#FFC107;'>{rating}</div>
</div>""", unsafe_allow_html=True)

# Match card con iniciales en círculos
team1_initials = "".join([word[0] for word in team1.split()]).upper()[:3]
team2_initials = "".join([word[0] for word in team2.split()]).upper()[:3]

st.markdown(f"""
<div class='match-header'>
    <div class='team-block'>
        <div class='team-circle'>{team1_initials}</div>
        <div class='team-role'>Local</div>
        <div class='team-name'>{team1}</div>
        <div class='team-xg'>{lambda_h:.2f}</div>
        <div class='team-xg-lbl'>xG calibrados</div>
    </div>
    <div class='vs-block'>
        <div class='vs-txt'>VS</div>
        <div class='vs-total'>{xg_total:.2f}</div>
        <div class='vs-copa'>xG total<br>media {avg_goals:.2f}</div>
    </div>
    <div class='team-block team-block-away'>
        <div class='team-circle' style='background:#EF5350;'>{team2_initials}</div>
        <div class='team-role'>Visitante</div>
        <div class='team-name'>{team2}</div>
        <div class='team-xg' style='color:#EF5350;'>{lambda_a:.2f}</div>
        <div class='team-xg-lbl'>xG calibrados</div>
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RESULTADO FINAL — 1X2
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🏆</span>
  <span class='sec-label'>01 · Resultado Final (1X2)</span>
</div>
<p class='sec-desc'>Probabilidad de cada resultado al término de los 90 minutos.</p>
""", unsafe_allow_html=True)

winner = team1 if h > d and h > a else ("Empate" if d > h and d > a else team2)
w_prob = max(h, d, a)

col1, col2, col3, col4 = st.columns(4)
with col1:
    border = "card-winner" if h == w_prob else ""
    st.markdown(f"""
    <div class='card {border}'>
        <div class='card-lbl'>Victoria {team1}</div>
        <div class='card-val' style='color:{prob_color(h)};'>{h:.1%}</div>
        {bar_html(h*100, prob_color(h))}
    </div>""", unsafe_allow_html=True)
with col2:
    border = "card-winner" if d == w_prob else ""
    st.markdown(f"""
    <div class='card {border}'>
        <div class='card-lbl'>Empate</div>
        <div class='card-val card-val-yellow'>{d:.1%}</div>
        {bar_html(d*100, "#FFC107")}
    </div>""", unsafe_allow_html=True)
with col3:
    border = "card-winner" if a == w_prob else ""
    st.markdown(f"""
    <div class='card {border}'>
        <div class='card-lbl'>Victoria {team2}</div>
        <div class='card-val' style='color:{prob_color(a)};'>{a:.1%}</div>
        {bar_html(a*100, prob_color(a))}
    </div>""", unsafe_allow_html=True)
with col4:
    bi, bj = np.unravel_index(np.argmax(mat), mat.shape)
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>Marcador Más Probable</div>
        <div class='card-val'>{bi}–{bj}</div>
        <div class='card-sub'>{mat[bi,bj]:.1%}</div>
        <div class='card-tag'>{team1[:3]} · {team2[:3]}</div>
    </div>""", unsafe_allow_html=True)

fig1x2 = go.Figure(go.Bar(
    x=[h*100, d*100, a*100],
    y=[f"{team1} gana", "Empate", f"{team2} gana"],
    orientation="h",
    marker_color=[prob_color(h), "#FFC107", prob_color(a)],
    text=[f"{h:.1%}", f"{d:.1%}", f"{a:.1%}"],
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(family="Space Mono", size=14, color="#121212"),
    width=0.55,
))
fig1x2.update_layout(
    plot_bgcolor="#1E1E1E", paper_bgcolor="#121212",
    font=dict(family="Roboto", color="#BDBDBD"),
    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
    yaxis=dict(showgrid=False, tickfont=dict(size=13)),
    margin=dict(t=8, b=8, l=10, r=10), height=140,
)
st.plotly_chart(fig1x2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOBLE OPORTUNIDAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🔀</span>
  <span class='sec-label'>02 · Doble Oportunidad</span>
</div>
<p class='sec-desc'>Cubre dos de los tres posibles resultados simultáneamente.</p>
""", unsafe_allow_html=True)

dc_items = [
    ("1X", f"{team1} gana o Empate", dc_1x, "#4CAF50"),
    ("12", "Cualquiera gana", dc_12, "#4FC3F7"),
    ("X2", f"Empate o {team2} gana", dc_x2, "#CE93D8"),
]
cols_dc = st.columns(3)
for col, (code, desc, val, color) in zip(cols_dc, dc_items):
    with col:
        col.markdown(f"""
        <div class='card'>
            <div class='card-tag' style='color:{color};border:1px solid {color};background:transparent;font-size:0.9rem;font-weight:700;'>{code}</div>
            <div class='card-val' style='color:{color};margin-top:12px;'>{val:.1%}</div>
            <div class='card-sub' style='margin-top:8px;'>{desc}</div>
            {bar_html(val*100, color)}
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DRAW NO BET
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🛡️</span>
  <span class='sec-label'>03 · Draw No Bet (DNB)</span>
</div>
<p class='sec-desc'>Excluye el empate del cálculo y redistribuye su probabilidad.</p>
""", unsafe_allow_html=True)

cols_dnb = st.columns(2)
for col, (lbl, val, color) in zip(cols_dnb, [
    (f"{team1} gana · sin contar empate", dnb_h, "#4CAF50"),
    (f"{team2} gana · sin contar empate", dnb_a, "#EF5350"),
]):
    col.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>{lbl}</div>
        <div class='card-val' style='color:{color};'>{val:.1%}</div>
        {bar_html(val*100, color)}
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='info-pill' style='margin-top:12px;'>
    <div>Prob. empate descartada <span>{d:.1%}</span></div>
    <div>Base de cálculo DNB <span>{h+a:.1%}</span></div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OVER / UNDER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>📊</span>
  <span class='sec-label'>04 · Over / Under — Líneas estándar</span>
</div>
<p class='sec-desc'>Probabilidad de superar o no cada línea de goles totales.</p>
""", unsafe_allow_html=True)

ou_lines = [
    (0.5, over05, under05),
    (1.5, over15, under15),
    (2.5, over25, under25),
    (3.5, over35, under35),
    (4.5, over45, under45),
]
ou_html = "<div class='ou-grid'>"
for line, ov, un in ou_lines:
    highlight = "border:2px solid #4CAF50;" if line == 2.5 else ""
    ou_html += f"""
    <div class='ou-card' style='{highlight}'>
        <div class='ou-line'>{line}</div>
        <div class='ou-tag-o'>Over</div>
        <div class='ou-over'>{ov:.1%}</div>
        <div style='height:4px;'></div>
        <div class='ou-tag-u'>Under</div>
        <div class='ou-under'>{un:.1%}</div>
    </div>"""
ou_html += "</div>"
st.markdown(ou_html, unsafe_allow_html=True)

fig_ou = go.Figure()
ov_vals = [v * 100 for _, v, _ in ou_lines]
un_vals = [v * 100 for _, _, v in ou_lines]
x_labs = [f"O/U {l}" for l, _, _ in ou_lines]
fig_ou.add_trace(go.Scatter(
    x=x_labs, y=ov_vals, name="Over", mode="lines+markers",
    line=dict(color="#4CAF50", width=2),
    marker=dict(size=8, color="#4CAF50"),
    text=[f"{v:.1f}%" for v in ov_vals], textposition="top center",
    textfont=dict(family="Space Mono", size=10, color="#4CAF50"),
))
fig_ou.add_trace(go.Scatter(
    x=x_labs, y=un_vals, name="Under", mode="lines+markers",
    line=dict(color="#EF5350", width=2, dash="dot"),
    marker=dict(size=8, color="#EF5350"),
    text=[f"{v:.1f}%" for v in un_vals], textposition="bottom center",
    textfont=dict(family="Space Mono", size=10, color="#EF5350"),
))
fig_ou.add_hline(y=50, line_dash="dash", line_color="#2C2C2C", line_width=1)
fig_ou.update_layout(
    plot_bgcolor="#1E1E1E", paper_bgcolor="#121212",
    font=dict(family="Roboto", color="#BDBDBD"),
    legend=dict(bgcolor="#1E1E1E", bordercolor="#2C2C2C", borderwidth=1,
                orientation="h", x=0.5, xanchor="center", y=1.1),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
    margin=dict(t=10, b=10, l=10, r=10), height=200,
)
st.plotly_chart(fig_ou, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ASIAN OVER / UNDER (LÍNEAS SPLIT)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🎯</span>
  <span class='sec-label'>05 · Asian Over/Under — Líneas Split</span>
</div>
<p class='sec-desc'>
  Las líneas asiáticas dividen la apuesta en dos mitades. Se muestran las probabilidades de ganancia completa, media ganancia (o media pérdida) y pérdida total.
</p>
""", unsafe_allow_html=True)

def asian_ou_card(line, data, is_over=True):
    if is_over:
        if "half_win" in data:
            full = data["full_win"]
            half = data["half_win"]
            loss = data["loss"]
        else:
            full = data["full_win"]
            half = data["half_loss"]
            loss = data["loss"]
        if "half_win" in data:
            detail = f"Gana: {full:.1%}<br>½ gana: {half:.1%}<br>Pierde: {loss:.1%}"
        else:
            detail = f"Gana: {full:.1%}<br>½ pierde: {half:.1%}<br>Pierde: {loss:.1%}"
    else:
        if "half_win" in data:
            full = data["loss"]
            half = data["half_win"]
            loss = data["full_win"]
            detail = f"Gana: {full:.1%}<br>½ pierde: {half:.1%}<br>Pierde: {loss:.1%}"
        else:
            full = data["loss"]
            half = data["half_loss"]
            loss = data["full_win"]
            detail = f"Gana: {full:.1%}<br>½ gana: {half:.1%}<br>Pierde: {loss:.1%}"
    return detail

cols_as = st.columns(3)
asian_pairs = [
    (2.25, asian_225),
    (2.75, asian_275),
    (3.25, asian_325),
]
for col, (line, data) in zip(cols_as, asian_pairs):
    over_detail = asian_ou_card(line, data, is_over=True)
    under_detail = asian_ou_card(line, data, is_over=False)
    col.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>Asian {line}</div>
        <div style='display:flex;justify-content:space-around;align-items:flex-start;margin-top:10px;'>
            <div>
                <div class='ou-tag-o' style='margin-bottom:4px;'>Over</div>
                <div style='font-family:Space Mono;font-size:0.9rem;color:#4CAF50;line-height:1.6;'>{over_detail}</div>
            </div>
            <div style='width:1px;background:#2C2C2C;align-self:stretch;'></div>
            <div>
                <div class='ou-tag-u' style='margin-bottom:4px;'>Under</div>
                <div style='font-family:Space Mono;font-size:0.9rem;color:#EF5350;line-height:1.6;'>{under_detail}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ASIAN HANDICAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🧧</span>
  <span class='sec-label'>06 · Hándicap Asiático</span>
</div>
<p class='sec-desc'>
  Probabilidades para las líneas de hándicap asiático más comunes. Referencia: {team1} como local.
</p>
""", unsafe_allow_html=True)

cols_ah = st.columns(3)
with cols_ah[0]:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>AH -0.25 ({team1})</div>
        <div style='margin:10px 0;'>
            <div style='color:#4CAF50;font-family:Space Mono;font-size:1.1rem;'>Gana: {ah_minus025_win:.1%}</div>
            <div style='color:#FFC107;font-family:Space Mono;font-size:0.9rem;'>½ pierde: {ah_minus025_halfloss:.1%}</div>
            <div style='color:#EF5350;font-family:Space Mono;font-size:0.9rem;'>Pierde: {ah_minus025_loss:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)

with cols_ah[1]:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>AH -0.5 ({team1})</div>
        <div style='margin:10px 0;'>
            <div style='color:#4CAF50;font-family:Space Mono;font-size:1.1rem;'>Gana: {ah_minus05_win:.1%}</div>
            <div style='color:#EF5350;font-family:Space Mono;font-size:0.9rem;'>Pierde: {ah_minus05_loss:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)

with cols_ah[2]:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>AH -0.75 ({team1})</div>
        <div style='margin:10px 0;'>
            <div style='color:#4CAF50;font-family:Space Mono;font-size:1rem;'>Gana completo: {ah_minus075_fullwin:.1%}</div>
            <div style='color:#FFC107;font-family:Space Mono;font-size:0.85rem;'>½ gana: {ah_minus075_halfwin:.1%}</div>
            <div style='color:#EF5350;font-family:Space Mono;font-size:0.85rem;'>Pierde: {ah_minus075_loss:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)

cols_ah2 = st.columns(3)
with cols_ah2[0]:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>AH +0.25 ({team2})</div>
        <div style='margin:10px 0;'>
            <div style='color:#4CAF50;font-family:Space Mono;font-size:1.1rem;'>Gana: {ah_plus025_win:.1%}</div>
            <div style='color:#FFC107;font-family:Space Mono;font-size:0.9rem;'>½ pierde: {ah_plus025_halfloss:.1%}</div>
            <div style='color:#EF5350;font-family:Space Mono;font-size:0.9rem;'>Pierde: {ah_plus025_loss:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)

with cols_ah2[1]:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>AH +0.5 ({team2})</div>
        <div style='margin:10px 0;'>
            <div style='color:#4CAF50;font-family:Space Mono;font-size:1.1rem;'>Gana: {ah_plus05_win:.1%}</div>
            <div style='color:#EF5350;font-family:Space Mono;font-size:0.9rem;'>Pierde: {ah_plus05_loss:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)

with cols_ah2[2]:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>AH +0.75 ({team2})</div>
        <div style='margin:10px 0;'>
            <div style='color:#4CAF50;font-family:Space Mono;font-size:1rem;'>Gana completo: {ah_plus075_fullwin:.1%}</div>
            <div style='color:#FFC107;font-family:Space Mono;font-size:0.85rem;'>½ gana: {ah_plus075_halfwin:.1%}</div>
            <div style='color:#EF5350;font-family:Space Mono;font-size:0.85rem;'>Pierde: {ah_plus075_loss:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BTTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>⚡</span>
  <span class='sec-label'>07 · Ambos Equipos Marcan (BTTS)</span>
</div>
<p class='sec-desc'>
  Probabilidad de que tanto {team1} como {team2} anoten al menos un gol.
</p>
""", unsafe_allow_html=True)

p_e1_marca = 1 - poisson.pmf(0, lambda_h)
p_e2_marca = 1 - poisson.pmf(0, lambda_a)

col_b1, col_b2, col_b3 = st.columns([2, 2, 3])
with col_b1:
    st.markdown(f"""
    <div class='btts-card btts-yes'>
        <div class='card-lbl' style='color:#4CAF50;'>✔ BTTS — Sí</div>
        <div class='card-val' style='font-size:2.4rem;'>{btts_y:.1%}</div>
        <div class='card-sub'>Ambos equipos marcan</div>
        {bar_html(btts_y*100)}
    </div>""", unsafe_allow_html=True)
with col_b2:
    st.markdown(f"""
    <div class='btts-card btts-no'>
        <div class='card-lbl' style='color:#EF5350;'>✘ BTTS — No</div>
        <div class='card-val card-val-red' style='font-size:2.4rem;'>{btts_n:.1%}</div>
        <div class='card-sub'>Al menos uno no marca</div>
        {bar_html(btts_n*100, "#EF5350")}
    </div>""", unsafe_allow_html=True)
with col_b3:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>Probabilidad individual de marcar</div>
        <div style='margin-top:12px;'>
            <div class='prob-bar-row'>
                <div class='prob-bar-lbl'>{team1} marca</div>
                <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{p_e1_marca*100:.1f}%;background:#4CAF50;'></div></div>
                <div class='prob-bar-num'>{p_e1_marca:.1%}</div>
            </div>
            <div class='prob-bar-row'>
                <div class='prob-bar-lbl'>{team2} marca</div>
                <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{p_e2_marca*100:.1f}%;background:#4FC3F7;'></div></div>
                <div class='prob-bar-num' style='color:#4FC3F7;'>{p_e2_marca:.1%}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class='card' style='text-align:left;padding:16px 20px;'>
    <div class='card-lbl' style='margin-bottom:12px;'>Composición del BTTS Sí</div>
    <div class='prob-bar-row'>
        <div class='prob-bar-lbl'>{team1} gana + BTTS</div>
        <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{home_btts_yes/btts_y*100:.1f}%;background:#4CAF50;'></div></div>
        <div class='prob-bar-num'>{home_btts_yes:.1%} ({(home_btts_yes/btts_y)*100:.0f}% del Sí)</div>
    </div>
    <div class='prob-bar-row'>
        <div class='prob-bar-lbl'>Empate + BTTS</div>
        <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{draw_btts_yes/btts_y*100:.1f}%;background:#FFC107;'></div></div>
        <div class='prob-bar-num' style='color:#FFC107;'>{draw_btts_yes:.1%} ({(draw_btts_yes/btts_y)*100:.0f}%)</div>
    </div>
    <div class='prob-bar-row'>
        <div class='prob-bar-lbl'>{team2} gana + BTTS</div>
        <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{away_btts_yes/btts_y*100:.1f}%;background:#EF5350;'></div></div>
        <div class='prob-bar-num' style='color:#EF5350;'>{away_btts_yes:.1%} ({(away_btts_yes/btts_y)*100:.0f}%)</div>
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GOLES TOTALES EXACTOS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🔢</span>
  <span class='sec-label'>08 · Goles Totales Exactos</span>
</div>
<p class='sec-desc'>Probabilidad de que el partido termine con exactamente N goles en total.</p>
""", unsafe_allow_html=True)

eg_labels = [f"{k} gol{'es' if k!=1 else ''}" for k in exact.keys()]
eg_values = [v * 100 for v in exact.values()]
peak_g    = max(exact, key=exact.get)
colors_eg = ["#4CAF50" if k == peak_g else "#1E4D2B" for k in exact.keys()]

fig_eg = go.Figure(go.Bar(
    x=eg_labels, y=eg_values,
    marker_color=colors_eg,
    marker_line_color="#121212", marker_line_width=2,
    text=[f"{v:.1f}%" for v in eg_values],
    textposition="outside",
    textfont=dict(family="Space Mono", size=11, color="#BDBDBD"),
))
fig_eg.add_annotation(
    x=f"{peak_g} gol{'es' if peak_g!=1 else ''}",
    y=exact[peak_g]*100 + 3,
    text="★ más probable",
    showarrow=False,
    font=dict(family="Space Mono", size=10, color="#4CAF50"),
    yshift=12,
)
fig_eg.update_layout(
    plot_bgcolor="#1E1E1E", paper_bgcolor="#121212",
    font=dict(family="Roboto", color="#BDBDBD"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(eg_values)*1.4]),
    margin=dict(t=30, b=10, l=10, r=10), height=250,
)
st.plotly_chart(fig_eg, use_container_width=True)

# Multigoal ranges
st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
mg_ranges = [
    ("0–1 goles",  0, 1,  "Partido muy cerrado"),
    ("2–3 goles",  2, 3,  "Rango más frecuente"),
    ("3–4 goles",  3, 4,  "Partido abierto"),
    ("2–4 goles",  2, 4,  "Rango amplio central"),
]
mg_html = "<div class='mg-grid'>"
for lbl, lo, hi, desc in mg_ranges:
    p = calc_multigoal(mat, lo, hi)
    mg_html += f"""
    <div class='mg-card'>
        <div class='card-lbl'>{lbl}</div>
        <div class='card-val'>{p:.1%}</div>
        <div class='card-sub'>{desc}</div>
        {bar_html(p*100)}
    </div>"""
mg_html += "</div>"
st.markdown(mg_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TOP 10 MARCADORES + HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🎯</span>
  <span class='sec-label'>09 · Marcadores Exactos</span>
</div>
<p class='sec-desc'>
  Los 10 marcadores más probables y el mapa de calor completo de la matriz de Poisson.
</p>
""", unsafe_allow_html=True)

top10_html = "<div class='top10-grid'>"
for rank, (prob, i, j) in enumerate(top10):
    cls = "top10-card-1" if rank == 0 else ""
    top10_html += f"""
    <div class='top10-card {cls}'>
        <div class='top10-rank'>#{rank+1}</div>
        <div class='top10-score'>{i}–{j}</div>
        <div class='top10-prob'>{prob:.1%}</div>
        <div class='card-sub' style='font-size:0.68rem;'>{team1[:3]} – {team2[:3]}</div>
    </div>"""
top10_html += "</div>"
st.markdown(top10_html, unsafe_allow_html=True)

SHOW = min(10, max(6, int(np.ceil(max(lambda_h, lambda_a) * 3))))
z = mat[:SHOW, :SHOW] * 100
text_mat = [[f"{z[i][j]:.1f}%" for j in range(SHOW)] for i in range(SHOW)]
fig_heat = go.Figure(go.Heatmap(
    z=z,
    x=[f"{team2[:3]} · {j}" for j in range(SHOW)],
    y=[f"{team1[:3]} · {i}" for i in range(SHOW)],
    colorscale=[[0,"#121212"],[0.2,"#1B3A1B"],[0.5,"#2E7D32"],[0.8,"#43A047"],[1.0,"#4CAF50"]],
    text=text_mat, texttemplate="%{text}",
    textfont=dict(family="Space Mono", size=11, color="#E0E0E0"),
    showscale=True,
    colorbar=dict(
        tickfont=dict(family="Space Mono", color="#BDBDBD", size=10),
        bgcolor="#1E1E1E", bordercolor="#2C2C2C", thickness=12,
        title=dict(text="%", font=dict(color="#BDBDBD", size=10)),
    ),
))
fig_heat.update_layout(
    plot_bgcolor="#1E1E1E", paper_bgcolor="#121212",
    font=dict(family="Roboto", color="#BDBDBD"),
    xaxis=dict(title=f"Goles {team2}", showgrid=False, side="top"),
    yaxis=dict(title=f"Goles {team1}", showgrid=False, autorange="reversed"),
    margin=dict(t=40, b=10, l=10, r=10), height=400,
)
st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MARGEN DE VICTORIA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>📐</span>
  <span class='sec-label'>10 · Margen de Victoria</span>
</div>
<p class='sec-desc'>
  Diferencia de goles entre los equipos al final del partido.
</p>
""", unsafe_allow_html=True)

mg_items = sorted(margins.items())
mg_x, mg_y, mg_c, mg_t = [], [], [], []
for diff, p in mg_items:
    if diff > 0:   label = f"{team1} +{diff}";  color = "#4CAF50"
    elif diff < 0: label = f"{team2} +{abs(diff)}"; color = "#EF5350"
    else:           label = "Empate";        color = "#FFC107"
    mg_x.append(label); mg_y.append(p*100); mg_c.append(color); mg_t.append(f"{p:.1%}")

fig_mg = go.Figure(go.Bar(
    x=mg_x, y=mg_y, marker_color=mg_c,
    marker_line_color="#121212", marker_line_width=2,
    text=mg_t, textposition="outside",
    textfont=dict(family="Space Mono", size=10, color="#BDBDBD"),
))
fig_mg.update_layout(
    plot_bgcolor="#1E1E1E", paper_bgcolor="#121212",
    font=dict(family="Roboto", color="#BDBDBD"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(mg_y)*1.35]),
    margin=dict(t=10, b=10, l=10, r=10), height=250,
)
st.plotly_chart(fig_mg, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PUNTOS ESPERADOS (xPts)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>📈</span>
  <span class='sec-label'>11 · Puntos Esperados (xPts)</span>
</div>
<p class='sec-desc'>
  Valor esperado de puntos que obtendría cada equipo si se jugase este partido muchas veces.
</p>
""", unsafe_allow_html=True)

col_xp1, col_xp2 = st.columns(2)
xp_max = 3.0
for col, (team, xp, color) in zip(
    [col_xp1, col_xp2],
    [(f"{team1} · Local", exp_h, "#4CAF50"), (f"{team2} · Visitante", exp_a, "#4FC3F7")]
):
    col.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>{team}</div>
        <div class='card-val' style='color:{color};font-size:2.8rem;'>{xp:.2f}</div>
        <div class='card-sub'>puntos esperados de 3.00 posibles</div>
        <div style='margin-top:12px;height:16px;background:#2C2C2C;border-radius:8px;overflow:hidden;'>
            <div style='width:{xp/xp_max*100:.1f}%;height:100%;background:{color};border-radius:8px;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:4px;'>
            <span style='font-size:0.65rem;color:#757575;font-family:Space Mono;'>0</span>
            <span style='font-size:0.65rem;color:#757575;font-family:Space Mono;'>{xp:.2f} / 3.00</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='info-pill' style='margin-top:12px;'>
    <div>{team1} · 3×P(win) = <span>{3*h:.2f}</span></div>
    <div>{team1} · 1×P(draw) = <span>{d:.2f}</span></div>
    <div>{team2} · 3×P(win) = <span>{3*a:.2f}</span></div>
    <div>{team2} · 1×P(draw) = <span>{d:.2f}</span></div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TABLA RESUMEN COMPLETA (REDISEÑADA)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>📋</span>
  <span class='sec-label'>12 · Resumen Completo de Mercados</span>
</div>
<p class='sec-desc'>Todos los mercados organizados por categoría en un formato visual claro y profesional.</p>
""", unsafe_allow_html=True)

bi, bj = np.unravel_index(np.argmax(mat), mat.shape)

# ── Helper para Asian O/U ──
def asian_over_resumen(data):
    if "half_win" in data:
        return f"<span style='color:#4CAF50;'>Full</span> {data['full_win']:.1%} &nbsp;|&nbsp; <span style='color:#FFC107;'>½Win</span> {data['half_win']:.1%} &nbsp;|&nbsp; <span style='color:#EF5350;'>Lose</span> {data['loss']:.1%}"
    else:
        return f"<span style='color:#4CAF50;'>Full</span> {data['full_win']:.1%} &nbsp;|&nbsp; <span style='color:#FFC107;'>½Loss</span> {data['half_loss']:.1%} &nbsp;|&nbsp; <span style='color:#EF5350;'>Lose</span> {data['loss']:.1%}"

def asian_under_resumen(data):
    if "half_win" in data:
        return f"<span style='color:#4CAF50;'>Full</span> {data['loss']:.1%} &nbsp;|&nbsp; <span style='color:#FFC107;'>½Loss</span> {data['half_win']:.1%} &nbsp;|&nbsp; <span style='color:#EF5350;'>Lose</span> {data['full_win']:.1%}"
    else:
        return f"<span style='color:#4CAF50;'>Full</span> {data['loss']:.1%} &nbsp;|&nbsp; <span style='color:#FFC107;'>½Win</span> {data['half_loss']:.1%} &nbsp;|&nbsp; <span style='color:#EF5350;'>Lose</span> {data['full_win']:.1%}"

# ── Construcción de bloques por categoría ──
bloques_html = ""

# --- RESULTADO FINAL ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>🏆</span> Resultado Final (1X2)
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>Victoria {team1}</span>
        <span class='resumen-val'>{h:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#FFC107;'></span>Empate</span>
        <span class='resumen-val' style='color:#FFC107;'>{d:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>Victoria {team2}</span>
        <span class='resumen-val' style='color:#EF5350;'>{a:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4FC3F7;'></span>Marcador más probable</span>
        <span class='resumen-val-sm'>{bi}–{bj} &nbsp;<span style='color:#757575;font-size:0.7rem;'>({mat[bi,bj]:.1%})</span></span>
    </div>
</div>"""

# --- DOBLE OPORTUNIDAD ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>🔀</span> Doble Oportunidad
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>1X — {team1} o Empate</span>
        <span class='resumen-val'>{dc_1x:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4FC3F7;'></span>12 — Cualquier ganador</span>
        <span class='resumen-val' style='color:#4FC3F7;'>{dc_12:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#CE93D8;'></span>X2 — Empate o {team2}</span>
        <span class='resumen-val' style='color:#CE93D8;'>{dc_x2:.1%}</span>
    </div>
</div>"""

# --- DRAW NO BET ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>🛡️</span> Draw No Bet
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>DNB {team1}</span>
        <span class='resumen-val'>{dnb_h:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>DNB {team2}</span>
        <span class='resumen-val' style='color:#EF5350;'>{dnb_a:.1%}</span>
    </div>
</div>"""

# --- OVER / UNDER ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>📊</span> Over / Under
    </div>"""
for line, ov, un in [(0.5, over05, under05), (1.5, over15, under15),
                       (2.5, over25, under25), (3.5, over35, under35),
                       (4.5, over45, under45)]:
    highlight_style = "background:#1A2E1A;" if line == 2.5 else ""
    bloques_html += f"""
    <div class='resumen-row' style='{highlight_style}'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>Over {line}</span>
        <span class='resumen-val'>{ov:.1%}</span>
    </div>
    <div class='resumen-row' style='{highlight_style}'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>Under {line}</span>
        <span class='resumen-val' style='color:#EF5350;'>{un:.1%}</span>
    </div>"""
bloques_html += "</div>"

# --- ASIAN OVER/UNDER ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>🎯</span> Asian Over/Under
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>Asian O/U 2.25 Over</span>
        <span class='resumen-val-sm'>{asian_over_resumen(asian_225)}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>Asian O/U 2.25 Under</span>
        <span class='resumen-val-sm'>{asian_under_resumen(asian_225)}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>Asian O/U 2.75 Over</span>
        <span class='resumen-val-sm'>{asian_over_resumen(asian_275)}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>Asian O/U 2.75 Under</span>
        <span class='resumen-val-sm'>{asian_under_resumen(asian_275)}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>Asian O/U 3.25 Over</span>
        <span class='resumen-val-sm'>{asian_over_resumen(asian_325)}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>Asian O/U 3.25 Under</span>
        <span class='resumen-val-sm'>{asian_under_resumen(asian_325)}</span>
    </div>
</div>"""

# --- ASIAN HANDICAP ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>🧧</span> Hándicap Asiático
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>AH -0.25 ({team1})</span>
        <span class='resumen-val-sm'>Win {ah_minus025_win:.1%} &nbsp;|&nbsp; ½Loss {ah_minus025_halfloss:.1%} &nbsp;|&nbsp; Loss {ah_minus025_loss:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>AH -0.5 ({team1})</span>
        <span class='resumen-val-sm'>Win {ah_minus05_win:.1%} &nbsp;|&nbsp; Loss {ah_minus05_loss:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>AH -0.75 ({team1})</span>
        <span class='resumen-val-sm'>Full {ah_minus075_fullwin:.1%} &nbsp;|&nbsp; ½Win {ah_minus075_halfwin:.1%} &nbsp;|&nbsp; Loss {ah_minus075_loss:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>AH +0.25 ({team2})</span>
        <span class='resumen-val-sm'>Win {ah_plus025_win:.1%} &nbsp;|&nbsp; ½Loss {ah_plus025_halfloss:.1%} &nbsp;|&nbsp; Loss {ah_plus025_loss:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>AH +0.5 ({team2})</span>
        <span class='resumen-val-sm'>Win {ah_plus05_win:.1%} &nbsp;|&nbsp; Loss {ah_plus05_loss:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>AH +0.75 ({team2})</span>
        <span class='resumen-val-sm'>Full {ah_plus075_fullwin:.1%} &nbsp;|&nbsp; ½Win {ah_plus075_halfwin:.1%} &nbsp;|&nbsp; Loss {ah_plus075_loss:.1%}</span>
    </div>
</div>"""

# --- BTTS ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>⚡</span> Ambos Equipos Marcan (BTTS)
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>BTTS — Sí</span>
        <span class='resumen-val'>{btts_y:.1%}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#EF5350;'></span>BTTS — No</span>
        <span class='resumen-val' style='color:#EF5350;'>{btts_n:.1%}</span>
    </div>
</div>"""

# --- GOLES EXACTOS ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>🔢</span> Goles Totales Exactos (más probables)
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>{peak_g} gol{'es' if peak_g != 1 else ''} → más probable</span>
        <span class='resumen-val'>{exact[peak_g]:.1%}</span>
    </div>"""
# Añadir top 5 goles exactos
sorted_exact = sorted(exact.items(), key=lambda x: x[1], reverse=True)
for g, p in sorted_exact[1:6]:
    bloques_html += f"""
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#424242;'></span>{g} gol{'es' if g != 1 else ''}</span>
        <span class='resumen-val-sm'>{p:.1%}</span>
    </div>"""
bloques_html += "</div>"

# --- PUNTOS ESPERADOS ---
bloques_html += f"""
<div class='resumen-block'>
    <div class='resumen-cat-header'>
        <span class='resumen-cat-icon'>📈</span> Puntos Esperados (xPts)
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4CAF50;'></span>xPts {team1}</span>
        <span class='resumen-val'>{exp_h:.2f}</span>
    </div>
    <div class='resumen-row'>
        <span class='resumen-name'><span class='dot' style='background:#4FC3F7;'></span>xPts {team2}</span>
        <span class='resumen-val' style='color:#4FC3F7;'>{exp_a:.2f}</span>
    </div>
</div>"""

st.markdown(bloques_html, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:40px 0 20px;border-top:1px solid #2C2C2C;margin-top:32px;'>
    <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#757575;letter-spacing:2px;'>
        COPA DEL MUNDO · ANÁLISIS xG · MODELO DIXON & COLES (POISSON BIVARIADA)
    </div>
    <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#424242;margin-top:6px;'>
        λ₁ = {lambda_h:.2f} ({team1}) · λ₂ = {lambda_a:.2f} ({team2}) · ρ = {rho:.2f} · Media torneo {avg_goals:.2f} goles/partido · xG total calibrado {xg_total:.2f}
    </div>
</div>""", unsafe_allow_html=True)