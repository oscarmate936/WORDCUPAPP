import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import urllib.parse

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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0D0D0D;
    color: #E0E0E0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #141414 !important;
    border-right: 1px solid #1F1F1F;
}
[data-testid="stSidebar"] * {
    color: #BDBDBD !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #2E7D32, #43A047) !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    margin-top: 16px !important;
    box-shadow: 0 4px 16px rgba(46, 125, 50, 0.35);
    transition: all 0.25s ease;
    letter-spacing: 0.3px;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #388E3C, #4CAF50) !important;
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.45);
    transform: translateY(-1px);
}

/* ── Section header ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 44px 0 6px 0;
    padding-bottom: 14px;
    border-bottom: 1px solid #1A1A1A;
}
.sec-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #2E7D32;
    letter-spacing: 2px;
    font-weight: 700;
    background: rgba(46,125,50,0.1);
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid rgba(46,125,50,0.2);
}
.sec-icon {
    font-size: 1.1rem;
}
.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    color: #E8E8E8;
    font-weight: 600;
    letter-spacing: 0.2px;
}
.sec-desc {
    font-size: 0.82rem;
    color: #666;
    margin: 0 0 22px 0;
    line-height: 1.65;
    font-weight: 400;
}

/* ── Cards ── */
.card {
    background: #141414;
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    border: 1px solid #1F1F1F;
    margin-bottom: 8px;
    transition: border-color 0.2s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: transparent;
    transition: background 0.2s;
}
.card:hover::before { background: rgba(76,175,80,0.4); }
.card:hover { border-color: #2A2A2A; transform: translateY(-1px); }

.card-winner {
    border: 1px solid rgba(76,175,80,0.5) !important;
    box-shadow: 0 0 0 1px rgba(76,175,80,0.1), 0 8px 24px rgba(0,0,0,0.4);
}
.card-winner::before { background: #4CAF50 !important; }

.card-val {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.6rem, 5vw, 2.5rem);
    font-weight: 700;
    color: #4CAF50;
    line-height: 1.1;
    letter-spacing: -1px;
}
.card-val-sm {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.2rem, 4vw, 1.9rem);
    font-weight: 700;
    color: #4CAF50;
    line-height: 1.1;
}
.card-val-red { color: #EF5350 !important; }
.card-val-yellow { color: #FFC107 !important; }
.card-lbl {
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
    font-weight: 500;
}
.card-sub {
    font-size: 0.75rem;
    color: #555;
    margin-top: 6px;
    font-weight: 400;
}
.card-tag {
    display: inline-block;
    font-size: 0.68rem;
    font-family: 'Space Mono', monospace;
    background: #1A1A1A;
    color: #4CAF50;
    padding: 4px 10px;
    border-radius: 5px;
    margin-top: 10px;
    border: 1px solid #222;
}

/* ── Progress bar ── */
.prob-bar-wrap { margin: 8px 0 12px; }
.prob-bar-row {
    display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
}
.prob-bar-lbl {
    font-size: 0.78rem; color: #999; min-width: 110px; font-weight: 400;
}
.prob-bar-track {
    flex: 1; height: 6px; background: #1A1A1A; border-radius: 3px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 3px; transition: width 0.5s cubic-bezier(.4,0,.2,1);
}
.prob-bar-num {
    font-family: 'Space Mono', monospace; font-size: 0.78rem;
    color: #4CAF50; min-width: 52px; text-align: right;
}

/* ── Match header ── */
.match-header {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    background: #141414;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 14px;
    border: 1px solid #1F1F1F;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.team-block {
    padding: 28px 24px;
    text-align: center;
}
.team-role {
    font-size: 0.65rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 10px;
    font-weight: 500;
}
.team-name {
    font-size: clamp(1.1rem, 3.5vw, 1.7rem);
    font-weight: 700;
    color: #FFFFFF;
    margin: 6px 0 14px;
    letter-spacing: -0.3px;
}
.team-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: #4CAF50;
    border-radius: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #0D0D0D;
    margin-bottom: 10px;
}
.team-xg {
    font-family: 'Space Mono', monospace;
    font-size: clamp(2.2rem, 6vw, 3.2rem);
    font-weight: 700;
    color: #4CAF50;
    line-height: 1;
    letter-spacing: -2px;
}
.team-xg-lbl {
    font-size: 0.65rem;
    color: #555;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.vs-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 0 20px;
    border-left: 1px solid #1A1A1A;
    border-right: 1px solid #1A1A1A;
    background: #111;
    min-width: 90px;
}
.vs-txt {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    letter-spacing: 3px;
}
.vs-total {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #4CAF50;
    margin: 6px 0;
    letter-spacing: -1px;
}
.vs-copa {
    font-size: 0.6rem;
    color: #444;
    text-align: center;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
    line-height: 1.6;
}

/* ── Info pill ── */
.info-pill {
    background: #141414;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.78rem;
    color: #666;
    margin-bottom: 20px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    border: 1px solid #1F1F1F;
    align-items: center;
}
.info-pill span {
    color: #4CAF50;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.82rem;
}

/* ── OU grid ── */
.ou-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 14px;
}
.ou-card {
    background: #141414;
    border-radius: 12px;
    padding: 18px 8px;
    text-align: center;
    border: 1px solid #1F1F1F;
    transition: border-color 0.2s;
}
.ou-card:hover { border-color: #2A2A2A; }
.ou-card-highlight {
    border: 1px solid rgba(76,175,80,0.4) !important;
    background: #121F12;
}
.ou-line {
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    color: #999;
    font-weight: 700;
    margin-bottom: 10px;
}
.ou-sep { width: 24px; height: 1px; background: #1F1F1F; margin: 8px auto; }
.ou-over { font-family: 'Space Mono', monospace; font-size: 1.15rem; color: #4CAF50; font-weight: 700; }
.ou-under { font-family: 'Space Mono', monospace; font-size: 1.15rem; color: #EF5350; font-weight: 700; }
.ou-tag-o { font-size: 0.62rem; color: #4CAF50; text-transform: uppercase; letter-spacing: 1.5px; margin: 4px 0 2px; font-weight: 600; }
.ou-tag-u { font-size: 0.62rem; color: #EF5350; text-transform: uppercase; letter-spacing: 1.5px; margin: 4px 0 2px; font-weight: 600; }

/* ── BTTS ── */
.btts-big {
    background: #141414;
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    border: 1px solid #1F1F1F;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.btts-big::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.btts-yes::after { background: linear-gradient(90deg, #2E7D32, #4CAF50); }
.btts-no::after  { background: linear-gradient(90deg, #B71C1C, #EF5350); }
.btts-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
    font-weight: 600;
}
.btts-pct {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -2px;
}
.btts-sub {
    font-size: 0.75rem;
    margin-top: 8px;
    color: #555;
}

/* ── Top scorelines ── */
.top10-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 16px;
}
.top10-card {
    background: #141414;
    border-radius: 12px;
    padding: 16px 8px;
    text-align: center;
    border: 1px solid #1F1F1F;
    transition: border-color 0.2s;
}
.top10-card:hover { border-color: #2A2A2A; }
.top10-card-1 {
    border: 1px solid rgba(76,175,80,0.5) !important;
    background: #121F12;
}
.top10-score {
    font-family: 'Space Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: #4CAF50;
    letter-spacing: -0.5px;
}
.top10-prob { font-size: 0.8rem; color: #777; margin-top: 6px; font-family: 'Space Mono', monospace; }
.top10-rank {
    font-size: 0.62rem;
    color: #444;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Multigoal ── */
.mg-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 16px;
}
.mg-card {
    background: #141414;
    border-radius: 12px;
    padding: 18px 12px;
    text-align: center;
    border: 1px solid #1F1F1F;
}

/* ── xPts ── */
.xpts-card {
    background: #141414;
    border-radius: 14px;
    padding: 22px 20px;
    border: 1px solid #1F1F1F;
    margin-bottom: 8px;
}
.xpts-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; font-weight: 500; }
.xpts-big { font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; line-height: 1; letter-spacing: -2px; }
.xpts-sub { font-size: 0.75rem; color: #444; margin-top: 10px; }
.xpts-track { height: 8px; background: #1A1A1A; border-radius: 4px; overflow: hidden; margin-top: 14px; }
.xpts-fill  { height: 100%; border-radius: 4px; }

/* ── WhatsApp button ── */
.wa-section {
    background: linear-gradient(135deg, #0d1f0d, #0f2810);
    border: 1px solid rgba(37,211,102,0.25);
    border-radius: 18px;
    padding: 32px 28px;
    margin-top: 44px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.wa-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #25D366, transparent);
}
.wa-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #E8E8E8;
    margin-bottom: 6px;
    letter-spacing: -0.2px;
}
.wa-sub {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 20px;
    line-height: 1.6;
}
.wa-btn {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: #25D366;
    color: #0D0D0D;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 14px 28px;
    border-radius: 12px;
    text-decoration: none;
    letter-spacing: 0.2px;
    box-shadow: 0 4px 20px rgba(37,211,102,0.3);
    transition: all 0.2s ease;
}
.wa-btn:hover {
    background: #22c55e;
    box-shadow: 0 6px 28px rgba(37,211,102,0.45);
    transform: translateY(-1px);
    color: #0D0D0D;
}

/* ── Misc ── */
.divider-line {
    height: 1px; background: #1A1A1A; margin: 6px 0;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D0D0D; }
::-webkit-scrollbar-thumb { background: #1F1F1F; border-radius: 3px; }

/* Mobile */
@media (max-width: 640px) {
    .ou-grid { grid-template-columns: repeat(3, 1fr); }
    .top10-grid { grid-template-columns: repeat(3, 1fr); }
    .mg-grid { grid-template-columns: repeat(2, 1fr); }
    .team-name { font-size: 1rem; }
    .team-xg { font-size: 2rem; }
    .match-header { border-radius: 14px; }
    .btts-pct { font-size: 2rem; }
    .xpts-big { font-size: 2.2rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Model functions ────────────────────────────────────────────────────────────
MAX_G = 25

def dixon_coles_matrix(lh, la, rho, n=MAX_G):
    """
    Devuelve (matriz, rho_ajustado).
    rho_ajustado es None si el rho pedido era válido; si no, es el valor
    al que se recortó para mantener factores tau >= 0.
    NOTA: la advertencia al usuario se emite FUERA de esta función porque
    st.warning dentro de una función cacheada no se re-renderiza en cache hits.
    """
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    p_i = poisson.pmf(i, lh)
    p_j = poisson.pmf(j, la)
    indep = p_i * p_j

    adjusted_rho = None
    eff_rho = rho
    # Condiciones de validez: 1 - lh*la*rho >= 0 ; 1 + lh*rho >= 0 ; 1 + la*rho >= 0 ; 1 - rho >= 0
    max_rho = min(1.0, 1.0 / (lh * la)) if lh * la > 0 else 1.0
    min_rho = -1.0 / max(lh, la) if max(lh, la) > 0 else 0.0
    if rho > max_rho or rho < min_rho:
        eff_rho = max(min(rho, max_rho), min_rho)
        adjusted_rho = eff_rho

    tau = np.ones((n, n))
    tau[0, 0] = 1 - lh * la * eff_rho
    tau[0, 1] = 1 + lh * eff_rho
    tau[1, 0] = 1 + la * eff_rho
    tau[1, 1] = 1 - eff_rho

    mat = indep * tau
    mat /= mat.sum()
    return mat, adjusted_rho


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


def calc_exact_total(mat, max_g=None):
    n = mat.shape[0]
    total_max = 2 * n - 2
    if max_g is None or max_g > total_max:
        max_g = total_max
    res = {}
    for g in range(max_g + 1):
        s = 0.0
        for i in range(min(g + 1, n)):
            j = g - i
            if 0 <= j < n:
                s += mat[i, j]
        res[g] = s
    return res


def calc_margin(mat):
    n = mat.shape[0]
    margins = {}
    for d in range(-5, 6):
        s = 0.0
        for i in range(n):
            j = i - d
            if 0 <= j < n:
                s += mat[i, j]
        margins[d] = s
    return margins


def calc_multigoal(mat, lo, hi):
    idx = np.indices(mat.shape)
    totals = idx[0] + idx[1]
    return float(np.sum(mat[(totals >= lo) & (totals <= hi)]))


def top_scorelines(mat, k=10):
    n = mat.shape[0]
    cells = [(mat[i, j], i, j) for i in range(n) for j in range(n)]
    cells.sort(reverse=True)
    return cells[:k]


def calc_asian_handicap(mat, H):
    """Devuelve (win, push, loss) para el LOCAL con hándicap H (vectorizado)."""
    n = mat.shape[0]
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    adj = i + H - j
    win = float(mat[adj > 1e-9].sum())
    push = float(mat[np.abs(adj) <= 1e-9].sum())
    loss = float(mat[adj < -1e-9].sum())
    return win, push, loss


def bar_html(pct, color="#4CAF50", height=6):
    return (
        f"<div style='height:{height}px;background:#1A1A1A;border-radius:3px;overflow:hidden;margin-top:10px;'>"
        f"<div style='width:{pct:.1f}%;height:100%;background:{color};border-radius:3px;transition:width .5s;'></div></div>"
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
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
        "<div style='font-size:0.72rem;color:#444;margin-top:12px;line-height:1.7;'>"
        "Modelo · Dixon & Coles (Poisson bivariada)<br>Correlación ajustable · Calibración por media del torneo</div>",
        unsafe_allow_html=True,
    )


# ── Gate ──────────────────────────────────────────────────────────────────────
if not run and "ready" not in st.session_state:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:65vh;text-align:center;gap:14px;'>
        <div style='font-size:5rem;line-height:1;'>⚽</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.6rem;font-weight:700;
                    color:#E8E8E8;letter-spacing:-0.5px;'>
            Copa del Mundo · Análisis xG
        </div>
        <div style='color:#555;font-size:0.88rem;max-width:320px;line-height:1.75;'>
            Ingresa los nombres, xG de ambos equipos y la media de goles
            del torneo en la barra lateral,<br>
            luego pulsa <b style='color:#4CAF50;'>Analizar Partido</b>.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

st.session_state["ready"] = True


# ── Compute ────────────────────────────────────────────────────────────────────
total_xg_input = xg_home + xg_away
if adjust_avg and total_xg_input > 0:
    scale = avg_goals / total_xg_input
    lambda_h = xg_home * scale
    lambda_a = xg_away * scale
else:
    lambda_h = xg_home
    lambda_a = xg_away

@st.cache_data(show_spinner=False, max_entries=128)
def build_matrix(lh, la, rho):
    return dixon_coles_matrix(lh, la, rho, MAX_G)

mat, rho_adjusted = build_matrix(lambda_h, lambda_a, rho)

if rho_adjusted is not None:
    st.warning(
        f"Con λ₁={lambda_h:.2f}, λ₂={lambda_a:.2f}, el ρ seleccionado ({rho:.2f}) genera factores τ negativos. "
        f"Se ha ajustado automáticamente a {rho_adjusted:.2f} para mantener la validez del modelo.",
        icon="⚠️"
    )

rho_eff = rho_adjusted if rho_adjusted is not None else rho

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

btts_y = float(np.sum(mat[1:, 1:]))
btts_n = 1.0 - btts_y
btts_y_safe = max(btts_y, 1e-12)

# Composición del BTTS Sí (vectorizado sobre la submatriz i>=1, j>=1)
sub = mat[1:, 1:]
home_btts_yes = float(np.sum(np.tril(sub, -1)))
draw_btts_yes = float(np.trace(sub))
away_btts_yes = float(np.sum(np.triu(sub, 1)))

exact   = calc_exact_total(mat, max_g=15)
margins = calc_margin(mat)
top10   = top_scorelines(mat, 10)

exp_h   = 3 * h + d
exp_a   = 3 * a + d

xg_total = lambda_h + lambda_a
diff_vs_avg = xg_total - avg_goals
rating = match_rating(xg_total, btts_y, over25)

# Probabilidad de marcar COHERENTE con el modelo Dixon-Coles
# (antes se usaba la Poisson independiente, que ignora el ajuste tau en los
#  marcadores bajos; los marginales de la matriz son la fuente correcta)
p_e1_marca = 1.0 - float(mat[0, :].sum())
p_e2_marca = 1.0 - float(mat[:, 0].sum())

bi, bj = np.unravel_index(np.argmax(mat), mat.shape)
peak_g = max(exact, key=exact.get)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# Copa del Mundo &nbsp;·&nbsp; Análisis Estadístico xG", unsafe_allow_html=True)

context_arrow = "▲" if diff_vs_avg > 0 else "▼"
context_color = "#4CAF50" if diff_vs_avg > 0 else "#EF5350"
context_txt   = "más goles que la media" if diff_vs_avg > 0 else "menos goles que la media"

st.markdown(f"""
<div class='info-pill'>
    <div>Modelo <span>Dixon & Coles</span></div>
    <div>xG orig. {team1} <span>{xg_home:.2f}</span></div>
    <div>xG orig. {team2} <span>{xg_away:.2f}</span></div>
    <div>xG calib. {team1} <span>{lambda_h:.2f}</span></div>
    <div>xG calib. {team2} <span>{lambda_a:.2f}</span></div>
    <div>Total <span>{xg_total:.2f}</span></div>
    <div>Media <span>{avg_goals:.2f}</span></div>
    <div style='color:{context_color};font-family:Space Mono,monospace;font-size:0.78rem;'>
        {context_arrow} {abs(diff_vs_avg):.2f} — {context_txt}
    </div>
    <div style='font-size:1rem;margin-left:auto;color:#FFC107;letter-spacing:1px;'>{rating}</div>
</div>""", unsafe_allow_html=True)

team1_initials = "".join([w[0] for w in team1.split()]).upper()[:3] or "???"
team2_initials = "".join([w[0] for w in team2.split()]).upper()[:3] or "???"

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
    <div class='team-block' style='text-align:center;'>
        <div class='team-circle' style='background:#EF5350;color:#0D0D0D;border-radius:12px;'>{team2_initials}</div>
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
  <span class='sec-num'>01</span>
  <span class='sec-icon'>🏆</span>
  <span class='sec-label'>Resultado Final (1X2)</span>
</div>
<p class='sec-desc'>Probabilidad de cada resultado al término de los 90 minutos.</p>
""", unsafe_allow_html=True)

w_prob = max(h, d, a)
col1, col2, col3, col4 = st.columns(4)
for col, (lbl, val, color) in zip(
    [col1, col2, col3],
    [
        (f"Victoria {team1}", h, prob_color(h)),
        ("Empate", d, "#FFC107"),
        (f"Victoria {team2}", a, prob_color(a)),
    ]
):
    is_winner = (val == w_prob)
    border_cls = "card-winner" if is_winner else ""
    winner_badge = "<div style='font-size:0.6rem;color:#4CAF50;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;'>◆ FAVORITO</div>" if is_winner else "<div style='height:22px;'></div>"
    col.markdown(f"""
    <div class='card {border_cls}'>
        {winner_badge}
        <div class='card-lbl'>{lbl}</div>
        <div class='card-val' style='color:{color};'>{val:.1%}</div>
        {bar_html(val*100, color)}
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='card'>
        <div style='height:22px;'></div>
        <div class='card-lbl'>Marcador Más Probable</div>
        <div class='card-val'>{bi}–{bj}</div>
        <div class='card-sub'>{mat[bi,bj]:.1%} de probabilidad</div>
        <div class='card-tag'>{team1[:4]} · {team2[:4]}</div>
    </div>""", unsafe_allow_html=True)

fig1x2 = go.Figure(go.Bar(
    x=[h*100, d*100, a*100],
    y=[f"{team1}", "Empate", f"{team2}"],
    orientation="h",
    marker_color=[prob_color(h), "#FFC107", prob_color(a)],
    marker_line_width=0,
    text=[f"{h:.1%}", f"{d:.1%}", f"{a:.1%}"],
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(family="Space Mono", size=14, color="#0D0D0D"),
    width=0.5,
))
fig1x2.update_layout(
    plot_bgcolor="#141414", paper_bgcolor="#0D0D0D",
    font=dict(family="Space Grotesk", color="#999"),
    xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
    yaxis=dict(showgrid=False, tickfont=dict(size=13, color="#BDBDBD")),
    margin=dict(t=8, b=8, l=8, r=8), height=130,
)
st.plotly_chart(fig1x2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOBLE OPORTUNIDAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>02</span>
  <span class='sec-icon'>🔀</span>
  <span class='sec-label'>Doble Oportunidad</span>
</div>
<p class='sec-desc'>Cubre dos de los tres posibles resultados simultáneamente.</p>
""", unsafe_allow_html=True)

dc_items = [
    ("1X", f"{team1} gana o Empate", dc_1x, "#4CAF50"),
    ("12", "Cualquiera gana · sin empate", dc_12, "#4FC3F7"),
    ("X2", f"Empate o {team2} gana", dc_x2, "#CE93D8"),
]
cols_dc = st.columns(3)
for col, (code, desc, val, color) in zip(cols_dc, dc_items):
    col.markdown(f"""
    <div class='card' style='border-color:rgba(255,255,255,0.04);'>
        <div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;
                    color:{color};border:1px solid {color}33;border-radius:8px;
                    padding:6px 16px;display:inline-block;margin-bottom:12px;letter-spacing:2px;'>{code}</div>
        <div class='card-val' style='color:{color};'>{val:.1%}</div>
        <div class='card-sub' style='margin-top:8px;'>{desc}</div>
        {bar_html(val*100, color)}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DRAW NO BET
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>03</span>
  <span class='sec-icon'>🛡️</span>
  <span class='sec-label'>Draw No Bet (DNB)</span>
</div>
<p class='sec-desc'>Excluye el empate del cálculo y redistribuye su probabilidad entre ambos equipos.</p>
""", unsafe_allow_html=True)

cols_dnb = st.columns([3, 3, 2])
for col, (lbl, val, color) in zip(cols_dnb[:2], [
    (f"{team1}", dnb_h, "#4CAF50"),
    (f"{team2}", dnb_a, "#EF5350"),
]):
    col.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>DNB · {lbl}</div>
        <div class='card-val' style='color:{color};'>{val:.1%}</div>
        <div class='card-sub'>probabilidad sin empate</div>
        {bar_html(val*100, color)}
    </div>""", unsafe_allow_html=True)

cols_dnb[2].markdown(f"""
<div class='card' style='text-align:left;'>
    <div class='card-lbl'>Desglose</div>
    <div style='margin-top:10px;'>
        <div style='display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1A1A1A;'>
            <span style='font-size:0.78rem;color:#666;'>Empate excluido</span>
            <span style='font-family:Space Mono,monospace;font-size:0.82rem;color:#FFC107;font-weight:700;'>{d:.1%}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:8px 0;'>
            <span style='font-size:0.78rem;color:#666;'>Base de cálculo</span>
            <span style='font-family:Space Mono,monospace;font-size:0.82rem;color:#4CAF50;font-weight:700;'>{h+a:.1%}</span>
        </div>
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OVER / UNDER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>04</span>
  <span class='sec-icon'>📊</span>
  <span class='sec-label'>Over / Under — Líneas estándar</span>
</div>
<p class='sec-desc'>Probabilidad de superar o no cada línea de goles totales en el partido.</p>
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
    hl = "ou-card-highlight" if line == 2.5 else ""
    ou_html += f"""
    <div class='ou-card {hl}'>
        <div class='ou-line'>{line}</div>
        <div class='ou-tag-o'>Over</div>
        <div class='ou-over'>{ov:.1%}</div>
        <div class='ou-sep'></div>
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
    line=dict(color="#4CAF50", width=2.5),
    marker=dict(size=8, color="#4CAF50", line=dict(color="#0D0D0D", width=2)),
    text=[f"{v:.1f}%" for v in ov_vals], textposition="top center",
    textfont=dict(family="Space Mono", size=10, color="#4CAF50"),
))
fig_ou.add_trace(go.Scatter(
    x=x_labs, y=un_vals, name="Under", mode="lines+markers",
    line=dict(color="#EF5350", width=2.5, dash="dot"),
    marker=dict(size=8, color="#EF5350", line=dict(color="#0D0D0D", width=2)),
    text=[f"{v:.1f}%" for v in un_vals], textposition="bottom center",
    textfont=dict(family="Space Mono", size=10, color="#EF5350"),
))
fig_ou.add_hline(y=50, line_dash="dash", line_color="#1F1F1F", line_width=1)
fig_ou.update_layout(
    plot_bgcolor="#141414", paper_bgcolor="#0D0D0D",
    font=dict(family="Space Grotesk", color="#999"),
    legend=dict(bgcolor="#141414", bordercolor="#1F1F1F", borderwidth=1,
                orientation="h", x=0.5, xanchor="center", y=1.12,
                font=dict(size=12)),
    xaxis=dict(showgrid=False, tickfont=dict(color="#666", size=11)),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, 115]),
    margin=dict(t=16, b=10, l=10, r=10), height=200,
)
st.plotly_chart(fig_ou, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HÁNDICAP DE GOLES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>05</span>
  <span class='sec-icon'>🎯</span>
  <span class='sec-label'>Hándicap de Goles</span>
</div>
<p class='sec-desc'>
Probabilidad de que el equipo <b>cubra</b> el hándicap indicado.
En líneas enteras (1.0, 2.0, 3.0) existe <b>push</b> (devolución de la apuesta) y se muestra por separado.
</p>
""", unsafe_allow_html=True)

def desc_handicap(L, negative):
    """Condición exacta para CUBRIR el hándicap (win)."""
    if negative:
        d = {
            0.5: "gana por 1+ gol",
            1.0: "gana por 2+ · push si gana por 1",
            1.5: "gana por 2+ goles",
            2.0: "gana por 3+ · push si gana por 2",
            2.5: "gana por 3+ goles",
            3.0: "gana por 4+ · push si gana por 3",
            3.5: "gana por 4+ goles",
        }
    else:
        d = {
            0.5: "empata o gana",
            1.0: "gana o empata · push si pierde por 1",
            1.5: "no pierde por 2+ goles",
            2.0: "pierde por máx. 1 · push si pierde por 2",
            2.5: "no pierde por 3+ goles",
            3.0: "pierde por máx. 2 · push si pierde por 3",
            3.5: "no pierde por 4+ goles",
        }
    return d.get(L, "")

def push_html(p):
    if p > 1e-4:
        return f"<div style='font-size:0.65rem;color:#FFC107;font-family:Space Mono,monospace;margin-top:4px;'>push {p:.1%}</div>"
    return ""

def ah_card(sign, L, desc, cover, push, color):
    # NOTA: el HTML se genera SIN saltos de línea ni indentación.
    # st.markdown interpreta una línea indentada 4+ espacios tras una línea
    # en blanco como bloque de código Markdown, y mostraba el HTML crudo.
    return (
        "<div style='flex:1; background:#141414; border:1px solid #1F1F1F; border-radius:12px; padding:12px;'>"
        f"<div style='font-family:Space Mono,monospace; font-size:0.85rem; font-weight:700; color:#E0E0E0;'>{sign}{L}</div>"
        f"<div style='font-size:0.7rem; color:#999; margin:4px 0 8px 0;'>{desc}</div>"
        f"<div style='font-family:Space Mono,monospace; font-size:1.3rem; font-weight:700; color:{color};'>{cover:.1%}</div>"
        f"{push_html(push)}"
        f"<div style='margin-top:6px;'>{bar_html(cover*100, color=color, height=4)}</div>"
        "</div>"
    )

handicap_lines = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

col_loc, col_vis = st.columns(2)

# ── LOCAL ──
with col_loc:
    st.markdown("<div style='font-family:Space Mono,monospace;font-weight:700;color:#4CAF50;margin-bottom:16px;'>🏠 LOCAL</div>", unsafe_allow_html=True)
    for L in handicap_lines:
        # Local -L: gana si adj = i - L - j > 0
        w_neg, p_neg, _ = calc_asian_handicap(mat, -L)
        # Local +L: gana si adj = i + L - j > 0
        w_pos, p_pos, _ = calc_asian_handicap(mat, L)

        html = (
            "<div style='display:flex; gap:10px; margin-bottom:12px;'>"
            + ah_card('-', L, desc_handicap(L, True),  w_neg, p_neg, '#4CAF50')
            + ah_card('+', L, desc_handicap(L, False), w_pos, p_pos, '#4CAF50')
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

# ── VISITANTE ──
with col_vis:
    st.markdown("<div style='font-family:Space Mono,monospace;font-weight:700;color:#4FC3F7;margin-bottom:16px;'>✈️ VISITANTE</div>", unsafe_allow_html=True)
    for L in handicap_lines:
        # Visitante -L cubre si j - L - i > 0  ⇔  i + L - j < 0  ⇔  PÉRDIDA del Local +L
        _, p_neg, w_neg = calc_asian_handicap(mat, L)
        # Visitante +L cubre si j + L - i > 0  ⇔  i - L - j < 0  ⇔  PÉRDIDA del Local -L
        _, p_pos, w_pos = calc_asian_handicap(mat, -L)

        html = (
            "<div style='display:flex; gap:10px; margin-bottom:12px;'>"
            + ah_card('-', L, desc_handicap(L, True),  w_neg, p_neg, '#4FC3F7')
            + ah_card('+', L, desc_handicap(L, False), w_pos, p_pos, '#4FC3F7')
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BTTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='sec-header'>
  <span class='sec-num'>06</span>
  <span class='sec-icon'>⚡</span>
  <span class='sec-label'>Ambos Equipos Marcan (BTTS)</span>
</div>
<p class='sec-desc'>Probabilidad de que tanto {team1} como {team2} anoten al menos un gol.</p>
""", unsafe_allow_html=True)

col_b1, col_b2, col_b3 = st.columns([2, 2, 3])
with col_b1:
    st.markdown(f"""
    <div class='btts-big btts-yes'>
        <div class='btts-label' style='color:#4CAF50;'>✔ BTTS — Sí</div>
        <div class='btts-pct' style='color:#4CAF50;'>{btts_y:.1%}</div>
        <div class='btts-sub'>Ambos equipos marcan</div>
        {bar_html(btts_y*100)}
    </div>""", unsafe_allow_html=True)

with col_b2:
    st.markdown(f"""
    <div class='btts-big btts-no'>
        <div class='btts-label' style='color:#EF5350;'>✘ BTTS — No</div>
        <div class='btts-pct' style='color:#EF5350;'>{btts_n:.1%}</div>
        <div class='btts-sub'>Al menos uno no marca</div>
        {bar_html(btts_n*100, "#EF5350")}
    </div>""", unsafe_allow_html=True)

with col_b3:
    st.markdown(f"""
    <div class='card' style='text-align:left;padding:20px;'>
        <div class='card-lbl' style='margin-bottom:14px;'>Probabilidad individual de marcar</div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team1}</div>
            <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{p_e1_marca*100:.1f}%;background:#4CAF50;'></div></div>
            <div class='prob-bar-num'>{p_e1_marca:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team2}</div>
            <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{p_e2_marca*100:.1f}%;background:#4FC3F7;'></div></div>
            <div class='prob-bar-num' style='color:#4FC3F7;'>{p_e2_marca:.1%}</div>
        </div>
        <div class='divider-line' style='margin:12px 0;'></div>
        <div class='card-lbl' style='margin-bottom:14px;'>Composición del BTTS Sí</div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team1} gana</div>
            <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{home_btts_yes/btts_y_safe*100:.1f}%;background:#4CAF50;'></div></div>
            <div class='prob-bar-num'>{home_btts_yes:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>Empate</div>
            <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{draw_btts_yes/btts_y_safe*100:.1f}%;background:#FFC107;'></div></div>
            <div class='prob-bar-num' style='color:#FFC107;'>{draw_btts_yes:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team2} gana</div>
            <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{away_btts_yes/btts_y_safe*100:.1f}%;background:#EF5350;'></div></div>
            <div class='prob-bar-num' style='color:#EF5350;'>{away_btts_yes:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GOLES TOTALES EXACTOS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>07</span>
  <span class='sec-icon'>🔢</span>
  <span class='sec-label'>Goles Totales Exactos</span>
</div>
<p class='sec-desc'>Probabilidad de que el partido termine con exactamente N goles en total.</p>
""", unsafe_allow_html=True)

eg_labels = [f"{k}G" for k in exact.keys()]
eg_values = [v * 100 for v in exact.values()]
colors_eg = ["#4CAF50" if k == peak_g else "#1E3A1E" for k in exact.keys()]

fig_eg = go.Figure(go.Bar(
    x=eg_labels, y=eg_values,
    marker_color=colors_eg,
    marker_line_width=0,
    text=[f"{v:.1f}%" for v in eg_values],
    textposition="outside",
    textfont=dict(family="Space Mono", size=10, color="#888"),
))
fig_eg.add_annotation(
    x=f"{peak_g}G",
    y=exact[peak_g]*100,
    text="★ más probable",
    showarrow=True,
    arrowhead=0,
    arrowcolor="#4CAF50",
    arrowwidth=1,
    ax=0, ay=-36,
    font=dict(family="Space Mono", size=10, color="#4CAF50"),
)
fig_eg.update_layout(
    plot_bgcolor="#141414", paper_bgcolor="#0D0D0D",
    font=dict(family="Space Grotesk", color="#999"),
    xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#666")),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(eg_values)*1.45]),
    margin=dict(t=20, b=10, l=10, r=10), height=250,
)
st.plotly_chart(fig_eg, use_container_width=True)

mg_ranges = [
    ("0–1 goles", 0, 1, "Partido muy cerrado"),
    ("2–3 goles", 2, 3, "Rango más frecuente"),
    ("3–4 goles", 3, 4, "Partido abierto"),
    ("2–4 goles", 2, 4, "Rango amplio central"),
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
# 8. TOP 10 MARCADORES + HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>08</span>
  <span class='sec-icon'>🎯</span>
  <span class='sec-label'>Marcadores Exactos</span>
</div>
<p class='sec-desc'>Los 10 marcadores más probables y el mapa de calor completo de la matriz de Poisson.</p>
""", unsafe_allow_html=True)

top10_html = "<div class='top10-grid'>"
for rank, (prob, i, j) in enumerate(top10):
    cls = "top10-card-1" if rank == 0 else ""
    score_color = "#4CAF50" if rank == 0 else ("#E0E0E0" if rank < 3 else "#888")
    top10_html += f"""
    <div class='top10-card {cls}'>
        <div class='top10-rank'>#{rank+1}</div>
        <div class='top10-score' style='color:{score_color};'>{i}–{j}</div>
        <div class='top10-prob'>{prob:.1%}</div>
    </div>"""
top10_html += "</div>"
st.markdown(top10_html, unsafe_allow_html=True)

SHOW = min(10, max(6, int(np.ceil(max(lambda_h, lambda_a) * 3))))
z = mat[:SHOW, :SHOW] * 100
text_mat = [[f"{z[i][j]:.1f}%" for j in range(SHOW)] for i in range(SHOW)]
fig_heat = go.Figure(go.Heatmap(
    z=z,
    x=[f"{team2[:4]} {j}" for j in range(SHOW)],
    y=[f"{team1[:4]} {i}" for i in range(SHOW)],
    colorscale=[[0,"#0D0D0D"],[0.15,"#142814"],[0.45,"#1E5C1E"],[0.75,"#2E7D32"],[1.0,"#4CAF50"]],
    text=text_mat, texttemplate="%{text}",
    textfont=dict(family="Space Mono", size=11, color="#E0E0E0"),
    showscale=True,
    colorbar=dict(
        tickfont=dict(family="Space Mono", color="#666", size=10),
        bgcolor="#141414", bordercolor="#1F1F1F", thickness=10,
        title=dict(text="%", font=dict(color="#666", size=10)),
    ),
))
fig_heat.update_layout(
    plot_bgcolor="#141414", paper_bgcolor="#0D0D0D",
    font=dict(family="Space Grotesk", color="#999"),
    xaxis=dict(title=dict(text=f"Goles {team2}", font=dict(color="#666")), showgrid=False, side="top", tickfont=dict(color="#666")),
    yaxis=dict(title=dict(text=f"Goles {team1}", font=dict(color="#666")), showgrid=False, autorange="reversed", tickfont=dict(color="#666")),
    margin=dict(t=40, b=10, l=10, r=10), height=400,
)
st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MARGEN DE VICTORIA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>09</span>
  <span class='sec-icon'>📐</span>
  <span class='sec-label'>Margen de Victoria</span>
</div>
<p class='sec-desc'>Diferencia de goles entre los equipos al final del partido (se muestran márgenes de hasta ±5).</p>
""", unsafe_allow_html=True)

mg_items = sorted(margins.items())
mg_x, mg_y, mg_c, mg_t = [], [], [], []
for diff, p in mg_items:
    if diff > 0:    label = f"{team1[:6]} +{diff}";  color = "#4CAF50"
    elif diff < 0:  label = f"{team2[:6]} +{abs(diff)}"; color = "#EF5350"
    else:            label = "Empate";        color = "#FFC107"
    mg_x.append(label); mg_y.append(p*100); mg_c.append(color); mg_t.append(f"{p:.1%}")

fig_mg = go.Figure(go.Bar(
    x=mg_x, y=mg_y, marker_color=mg_c,
    marker_line_width=0,
    text=mg_t, textposition="outside",
    textfont=dict(family="Space Mono", size=10, color="#888"),
))
fig_mg.update_layout(
    plot_bgcolor="#141414", paper_bgcolor="#0D0D0D",
    font=dict(family="Space Grotesk", color="#999"),
    xaxis=dict(showgrid=False, tickfont=dict(color="#666", size=11)),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(mg_y)*1.38]),
    margin=dict(t=10, b=10, l=10, r=10), height=250,
)
st.plotly_chart(fig_mg, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PUNTOS ESPERADOS (xPts)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-num'>10</span>
  <span class='sec-icon'>📈</span>
  <span class='sec-label'>Puntos Esperados (xPts)</span>
</div>
<p class='sec-desc'>Valor esperado de puntos que obtendría cada equipo si se jugase este partido muchas veces.</p>
""", unsafe_allow_html=True)

col_xp1, col_xp2 = st.columns(2)
for col, (team, xp, color) in zip(
    [col_xp1, col_xp2],
    [(team1, exp_h, "#4CAF50"), (team2, exp_a, "#4FC3F7")]
):
    col.markdown(f"""
    <div class='xpts-card'>
        <div class='xpts-label'>{team}</div>
        <div class='xpts-big' style='color:{color};'>{xp:.2f}</div>
        <div class='xpts-sub'>puntos esperados de 3.00 posibles</div>
        <div class='xpts-track'>
            <div class='xpts-fill' style='width:{xp/3*100:.1f}%;background:{color};'></div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:5px;'>
            <span style='font-size:0.62rem;color:#333;font-family:Space Mono;'>0</span>
            <span style='font-size:0.62rem;color:{color};font-family:Space Mono;'>{xp:.2f} / 3.00</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='info-pill' style='margin-top:4px;'>
    <div>{team1} xPts = 3×{h:.2f} + {d:.2f} = <span>{exp_h:.2f}</span></div>
    <div>{team2} xPts = 3×{a:.2f} + {d:.2f} = <span>{exp_a:.2f}</span></div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. BOTÓN WHATSAPP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header' style='margin-top:44px;'>
  <span class='sec-num'>11</span>
  <span class='sec-icon'>📤</span>
  <span class='sec-label'>Compartir Análisis</span>
</div>
""", unsafe_allow_html=True)

def pct(v): return f"{v:.1%}"

sorted_exact_top5 = sorted(exact.items(), key=lambda x: x[1], reverse=True)[:5]

wa_message = f"""⚽ *ANÁLISIS xG · COPA DEL MUNDO*
━━━━━━━━━━━━━━━━━━━━━━
🏟️ *{team1}* (Local) vs *{team2}* (Visitante)
📊 Modelo: Dixon & Coles · Poisson bivariada
xG {team1}: {lambda_h:.2f} | xG {team2}: {lambda_a:.2f}
xG Total: {xg_total:.2f} | Media torneo: {avg_goals:.2f}
Valoración: {rating}

━━━━━━━━━━━━━━━━━━━━━━
🏆 *RESULTADO FINAL (1X2)*
✅ {team1} gana: {pct(h)}
🤝 Empate: {pct(d)}
❌ {team2} gana: {pct(a)}
🎯 Marcador más probable: {bi}–{bj} ({mat[bi,bj]:.1%})

━━━━━━━━━━━━━━━━━━━━━━
🔀 *DOBLE OPORTUNIDAD*
1X ({team1} o Empate): {pct(dc_1x)}
12 (Cualquier ganador): {pct(dc_12)}
X2 (Empate o {team2}): {pct(dc_x2)}

━━━━━━━━━━━━━━━━━━━━━━
🛡️ *DRAW NO BET*
DNB {team1}: {pct(dnb_h)}
DNB {team2}: {pct(dnb_a)}

━━━━━━━━━━━━━━━━━━━━━━
📊 *OVER / UNDER*
O/U 0.5 → Over {pct(over05)} | Under {pct(under05)}
O/U 1.5 → Over {pct(over15)} | Under {pct(under15)}
⭐ O/U 2.5 → Over {pct(over25)} | Under {pct(under25)}
O/U 3.5 → Over {pct(over35)} | Under {pct(under35)}
O/U 4.5 → Over {pct(over45)} | Under {pct(under45)}

━━━━━━━━━━━━━━━━━━━━━━
⚡ *BTTS (AMBOS MARCAN)*
✅ BTTS Sí: {pct(btts_y)}
❌ BTTS No: {pct(btts_n)}
{team1} marca: {pct(p_e1_marca)} | {team2} marca: {pct(p_e2_marca)}

━━━━━━━━━━━━━━━━━━━━━━
🔢 *GOLES TOTALES MÁS PROBABLES*
{chr(10).join(f"  {g} gol{'es' if g!=1 else ''}:  {pct(p)}" for g, p in sorted_exact_top5)}

━━━━━━━━━━━━━━━━━━━━━━
📈 *PUNTOS ESPERADOS (xPts)*
{team1}: {exp_h:.2f} pts
{team2}: {exp_a:.2f} pts

━━━━━━━━━━━━━━━━━━━━━━
_Generado con modelo Dixon & Coles_
_ρ={rho_eff:.2f} · λ₁={lambda_h:.2f} · λ₂={lambda_a:.2f}_"""

wa_encoded = urllib.parse.quote(wa_message)
wa_url = f"https://wa.me/?text={wa_encoded}"

st.markdown(f"""
<div class='wa-section'>
    <div class='wa-title'>Compartir análisis completo</div>
    <div class='wa-sub'>
        Envía todos los mercados organizados por categoría en un mensaje profesional.<br>
        Incluye 1X2, Doble Oportunidad, DNB, O/U, BTTS, Goles exactos y xPts.
    </div>
    <a class='wa-btn' href='{wa_url}' target='_blank'>
        <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='currentColor'>
          <path d='M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z'/>
        </svg>
        Enviar por WhatsApp
    </a>
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:40px 0 20px;border-top:1px solid #1A1A1A;margin-top:32px;'>
    <div style='font-family:Space Mono,monospace;font-size:0.62rem;color:#333;letter-spacing:2.5px;text-transform:uppercase;'>
        Copa del Mundo · Análisis xG · Modelo Dixon & Coles
    </div>
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#2A2A2A;margin-top:6px;'>
        λ₁ = {lambda_h:.2f} ({team1}) · λ₂ = {lambda_a:.2f} ({team2}) · ρ = {rho_eff:.2f} · Media {avg_goals:.2f} g/p · Total {xg_total:.2f}
    </div>
</div>""", unsafe_allow_html=True) 