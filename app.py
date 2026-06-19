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

/* ─── Summary redesign ─── */
.summary-container {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 800px;
    overflow-y: auto;
    padding-right: 4px;
}
.summary-item {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #1E1E1E;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    transition: transform 0.2s, box-shadow 0.2s;
    border-left: 4px solid #2C2C2C;
}
.summary-item:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(76,175,80,0.2);
}
.summary-item.top-pick {
    border-left: 4px solid #FFC107;
    background: #1A1A1A;
}
.summary-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #757575;
    min-width: 28px;
    text-align: right;
}
.summary-info {
    flex: 1;
    min-width: 0;
}
.summary-category {
    font-size: 0.65rem;
    color: #616161;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
    margin-bottom: 2px;
}
.summary-name {
    font-size: 0.9rem;
    color: #E0E0E0;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.summary-type-badge {
    display: inline-block;
    font-size: 0.6rem;
    font-family: 'Space Mono', monospace;
    background: #2C2C2C;
    color: #4CAF50;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 4px;
}
.summary-prob-section {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 200px;
}
.summary-bar-wrap {
    flex: 1;
    height: 10px;
    background: #2C2C2C;
    border-radius: 5px;
    overflow: hidden;
}
.summary-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.4s;
}
.summary-prob-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    min-width: 55px;
    text-align: right;
}
.summary-trend {
    font-size: 0.8rem;
    min-width: 24px;
    text-align: center;
}

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
    .summary-item { flex-wrap: wrap; gap: 8px; padding: 12px 14px; }
    .summary-prob-section { min-width: 100%; }
}
</style>
""", unsafe_allow_html=True)


# ── Model functions ───────────────────────────────────────────────────────────
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


def asian_over_summary(data):
    if "half_win" in data:
        return f"Full {data['full_win']:.1%} / ½ {data['half_win']:.1%} / Lose {data['loss']:.1%}"
    else:
        return f"Full {data['full_win']:.1%} / ½Loss {data['half_loss']:.1%} / Lose {data['loss']:.1%}"


def asian_under_summary(data):
    if "half_win" in data:
        return f"Full {data['loss']:.1%} / ½Loss {data['half_win']:.1%} / Lose {data['full_win']:.1%}"
    else:
        return f"Full {data['loss']:.1%} / ½Win {data['half_loss']:.1%} / Lose {data['full_win']:.1%}"


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
    border = "card
