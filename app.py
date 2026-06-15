cat > /home/claude/app.py << 'ENDOFFILE'
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp { background: #080e0a; color: #e8f5e9; }

[data-testid="stSidebar"] {
    background: #0b1610 !important;
    border-right: 1px solid #1a3320;
}
[data-testid="stSidebar"] * { color: #b9dbbe !important; }
[data-testid="stSidebar"] .stButton button {
    background: #69f0ae !important;
    color: #040a05 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px !important;
    margin-top: 8px !important;
    transition: background 0.2s !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #00e676 !important;
}

/* ─── Section header ─── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 36px 0 4px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1a3320;
}
.sec-icon { font-size: 1.1rem; }
.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #4caf50;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 700;
}
.sec-desc {
    font-size: 0.8rem;
    color: #4a7a50;
    margin: 0 0 16px 0;
    line-height: 1.5;
}

/* ─── Cards ─── */
.card {
    background: #0b1610;
    border: 1px solid #1a3320;
    border-radius: 10px;
    padding: 16px 18px 14px;
    text-align: center;
    height: 100%;
}
.card-winner { border-color: #69f0ae; }
.card-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #69f0ae;
    line-height: 1;
}
.card-val-sm {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #69f0ae;
    line-height: 1;
}
.card-val-red { color: #ef5350 !important; }
.card-val-yellow { color: #ffd54f !important; }
.card-lbl {
    font-size: 0.7rem;
    color: #4caf50;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 5px;
}
.card-sub { font-size: 0.78rem; color: #388e3c; margin-top: 4px; }
.card-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    background: #1a3320;
    color: #69f0ae;
    padding: 2px 8px;
    border-radius: 3px;
    margin-top: 6px;
}

/* ─── Progress bar ─── */
.prob-bar-wrap { margin: 6px 0 12px; }
.prob-bar-row {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.prob-bar-lbl { font-size: 0.8rem; color: #81c784; min-width: 120px; }
.prob-bar-track {
    flex: 1; height: 10px; background: #1a3320; border-radius: 5px; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.4s; }
.prob-bar-num {
    font-family: 'Space Mono', monospace; font-size: 0.78rem;
    color: #69f0ae; min-width: 46px; text-align: right;
}

/* ─── Table-style rows ─── */
.mkt-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 14px; border-bottom: 1px solid #111f14;
    border-radius: 0;
}
.mkt-row:first-child { border-radius: 8px 8px 0 0; }
.mkt-row:last-child  { border-bottom: none; border-radius: 0 0 8px 8px; }
.mkt-row:hover { background: #0f2015; }
.mkt-block { background: #0b1610; border: 1px solid #1a3320; border-radius: 8px; margin-bottom: 8px; }
.mkt-name { font-size: 0.85rem; color: #a5d6a7; }
.mkt-desc { font-size: 0.72rem; color: #4a7a50; margin-top: 1px; }
.mkt-val {
    font-family: 'Space Mono', monospace; font-size: 1rem;
    font-weight: 700; color: #69f0ae;
}
.mkt-badge {
    font-size: 0.62rem; font-family: 'Space Mono', monospace;
    background: #1a3320; color: #4caf50;
    padding: 2px 7px; border-radius: 3px; margin-left: 6px;
}

/* ─── Team header ─── */
.match-header {
    display: grid; grid-template-columns: 1fr auto 1fr;
    gap: 0; background: #0b1610;
    border: 1px solid #1a3320; border-radius: 12px;
    overflow: hidden; margin-bottom: 8px;
}
.team-block { padding: 22px 28px; }
.team-block-away { text-align: right; }
.team-role { font-size: 0.65rem; color: #4caf50; text-transform: uppercase; letter-spacing: 2px; }
.team-name { font-size: 1.5rem; font-weight: 700; color: #e8f5e9; margin: 2px 0 8px; }
.team-xg  { font-family: 'Space Mono', monospace; font-size: 2.6rem; font-weight: 700; color: #69f0ae; line-height: 1; }
.team-xg-lbl { font-size: 0.7rem; color: #388e3c; }
.vs-block {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 0 20px;
    border-left: 1px solid #1a3320; border-right: 1px solid #1a3320;
    background: #080e0a;
}
.vs-txt { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #2e7d32; }
.vs-total { font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; color: #4caf50; }
.vs-copa { font-size: 0.65rem; color: #2e7d32; text-align: center; margin-top: 4px; }

/* ─── Info pill ─── */
.info-pill {
    background: #0b1610; border: 1px solid #1a3320; border-radius: 6px;
    padding: 10px 16px; font-size: 0.8rem; color: #4a7a50;
    margin-bottom: 24px; display: flex; gap: 20px; flex-wrap: wrap;
}
.info-pill span { color: #69f0ae; font-family: 'Space Mono', monospace; }

/* ─── OU pill grid ─── */
.ou-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 8px; }
.ou-card {
    background: #0b1610; border: 1px solid #1a3320; border-radius: 8px;
    padding: 12px 8px; text-align: center;
}
.ou-line { font-family: 'Space Mono', monospace; font-size: 0.95rem; color: #e8f5e9; font-weight: 700; }
.ou-over { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #69f0ae; font-weight: 700; }
.ou-under { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #ef5350; font-weight: 700; }
.ou-tag-o { font-size: 0.6rem; color: #69f0ae; text-transform: uppercase; letter-spacing: 1px; }
.ou-tag-u { font-size: 0.6rem; color: #ef5350; text-transform: uppercase; letter-spacing: 1px; }

/* ─── BTTS gauge ─── */
.btts-wrap { display: flex; gap: 12px; }
.btts-card {
    flex: 1; background: #0b1610; border-radius: 10px; padding: 18px 16px; text-align: center;
}
.btts-yes { border: 1px solid #2e7d32; }
.btts-no  { border: 1px solid #3a1010; }

/* ─── Summary table rows ─── */
.sum-group-label {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    color: #2e7d32; text-transform: uppercase; letter-spacing: 3px;
    padding: 14px 14px 4px; background: #080e0a;
}

/* ─── Scoreline top-5 ─── */
.top5-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
.top5-card {
    background: #0b1610; border: 1px solid #1a3320; border-radius: 8px;
    padding: 14px 8px; text-align: center;
}
.top5-card-1 { border-color: #69f0ae; }
.top5-score {
    font-family: 'Space Mono', monospace; font-size: 1.4rem;
    font-weight: 700; color: #69f0ae;
}
.top5-prob { font-size: 0.78rem; color: #4caf50; margin-top: 4px; }
.top5-rank { font-size: 0.6rem; color: #2e7d32; margin-top: 2px; font-family: 'Space Mono', monospace; }

/* ─── Multigoal grid ─── */
.mg-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.mg-card {
    background: #0b1610; border: 1px solid #1a3320; border-radius: 8px;
    padding: 14px 10px; text-align: center;
}

/* ─── xPts bar ─── */
.xpts-row {
    display: flex; align-items: center; gap: 14px; padding: 10px 0;
}
.xpts-team { font-size: 0.85rem; color: #a5d6a7; min-width: 80px; }
.xpts-track { flex: 1; height: 14px; background: #1a3320; border-radius: 7px; overflow: hidden; }
.xpts-fill  { height: 100%; background: #69f0ae; border-radius: 7px; }
.xpts-val   { font-family: 'Space Mono', monospace; font-size: 1rem; color: #69f0ae; min-width: 40px; text-align: right; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080e0a; }
::-webkit-scrollbar-thumb { background: #1a3320; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Model functions ───────────────────────────────────────────────────────────
MAX_G = 9

def poisson_matrix(lh, la, n=MAX_G):
    return np.outer([poisson.pmf(k, lh) for k in range(n)],
                    [poisson.pmf(k, la) for k in range(n)])

def calc_1x2(mat):
    h = float(np.sum(np.tril(mat, -1)))
    d = float(np.trace(mat))
    a = float(np.sum(np.triu(mat, 1)))
    return h, d, a

def calc_ou(mat, line):
    n = mat.shape[0]
    over = sum(mat[i,j] for i in range(n) for j in range(n) if i+j > line)
    return float(over), float(1 - over)

def calc_btts(lh, la):
    yes = (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
    return float(yes), float(1 - yes)

def calc_exact_total(mat, max_g=8):
    n = mat.shape[0]
    return {g: float(sum(mat[i, g-i] for i in range(min(g+1,n)) if g-i < n))
            for g in range(max_g+1)}

def calc_asian_ou(lh, la, line):
    mat = poisson_matrix(lh, la)
    lo, hi = int(line - 0.25), int(line - 0.25) + 1
    o1, u1 = calc_ou(mat, lo)
    o2, u2 = calc_ou(mat, hi)
    return (o1+o2)/2, (u1+u2)/2

def calc_margin(mat):
    n = mat.shape[0]
    return {d: float(sum(mat[i, i-d] for i in range(n) if 0 <= i-d < n))
            for d in range(-5, 6)}

def calc_multigoal(mat, lo, hi):
    n = mat.shape[0]
    return float(sum(mat[i,j] for i in range(n) for j in range(n) if lo <= i+j <= hi))

def top_scorelines(mat, k=5):
    n = mat.shape[0]
    cells = [(mat[i,j], i, j) for i in range(n) for j in range(n)]
    cells.sort(reverse=True)
    return cells[:k]

def bar_html(pct, color="#69f0ae", height=8):
    return (f"<div style='height:{height}px;background:#1a3320;border-radius:4px;overflow:hidden;margin-top:5px;'>"
            f"<div style='width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;'></div></div>")

def prob_color(p):
    if p >= 0.55: return "#69f0ae"
    if p >= 0.35: return "#ffd54f"
    return "#ef5350"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Parámetros")
    st.markdown("---")
    xg_home = st.number_input("xG Equipo 1 · Local", min_value=0.10, max_value=6.0,
                               value=1.45, step=0.05, format="%.2f",
                               help="Goles esperados del equipo local según el modelo xG")
    xg_away = st.number_input("xG Equipo 2 · Visitante", min_value=0.10, max_value=6.0,
                               value=1.20, step=0.05, format="%.2f",
                               help="Goles esperados del equipo visitante según el modelo xG")
    avg_goals = st.number_input("Media goles · Copa del Mundo", min_value=0.5,
                                 max_value=6.0, value=2.52, step=0.01, format="%.2f",
                                 help="Promedio de goles totales por partido en la edición actual")
    st.markdown("---")
    run = st.button("⚽  Analizar Partido", use_container_width=True, type="primary")
    st.markdown(
        "<div style='font-size:0.72rem;color:#2e7d32;margin-top:10px;line-height:1.6;'>"
        "Modelo · Poisson Bivariada<br>λ = xG como tasa de goles</div>",
        unsafe_allow_html=True)

# ── Gate ──────────────────────────────────────────────────────────────────────
if not run and "ready" not in st.session_state:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:65vh;text-align:center;gap:12px;'>
        <div style='font-size:5rem;'>⚽</div>
        <div style='font-family:Space Grotesk;font-size:1.5rem;font-weight:700;color:#69f0ae;'>
            Copa del Mundo · Análisis xG
        </div>
        <div style='color:#2e7d32;font-size:0.9rem;max-width:340px;line-height:1.7;'>
            Ingresa los xG de ambos equipos y la media de goles<br>
            de la Copa del Mundo en la barra lateral,<br>
            luego pulsa <b style='color:#69f0ae;'>Analizar Partido</b>.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

st.session_state["ready"] = True


# ── Compute ───────────────────────────────────────────────────────────────────
mat  = poisson_matrix(xg_home, xg_away)
h, d, a = calc_1x2(mat)

dc_1x = h + d
dc_12 = h + a
dc_x2 = d + a
dnb_h = h / (h + a)
dnb_a = a / (h + a)

over05, under05 = calc_ou(mat, 0.5)
over15, under15 = calc_ou(mat, 1.5)
over25, under25 = calc_ou(mat, 2.5)
over35, under35 = calc_ou(mat, 3.5)
over45, under45 = calc_ou(mat, 4.5)

asian_225_o, asian_225_u = calc_asian_ou(xg_home, xg_away, 2.25)
asian_275_o, asian_275_u = calc_asian_ou(xg_home, xg_away, 2.75)
asian_325_o, asian_325_u = calc_asian_ou(xg_home, xg_away, 3.25)

btts_y, btts_n = calc_btts(xg_home, xg_away)
exact   = calc_exact_total(mat)
margins = calc_margin(mat)
top5    = top_scorelines(mat, 5)
exp_h   = 3*h + d
exp_a   = 3*a + d
xg_total = xg_home + xg_away
diff_vs_avg = xg_total - avg_goals


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# Copa del Mundo &nbsp;·&nbsp; Análisis Estadístico xG", unsafe_allow_html=True)

context_arrow = "▲" if diff_vs_avg > 0 else "▼"
context_color = "#69f0ae" if diff_vs_avg > 0 else "#ef5350"
context_txt   = "partido con más goles que la media" if diff_vs_avg > 0 else "partido con menos goles que la media"

st.markdown(f"""
<div class='info-pill'>
    <div>Modelo <span>Poisson Bivariada</span></div>
    <div>xG Local <span>{xg_home:.2f}</span></div>
    <div>xG Visitante <span>{xg_away:.2f}</span></div>
    <div>xG Total <span>{xg_total:.2f}</span></div>
    <div>Media Copa <span>{avg_goals:.2f}</span></div>
    <div style='color:{context_color};'>{context_arrow} {abs(diff_vs_avg):.2f} — {context_txt}</div>
</div>""", unsafe_allow_html=True)

# Match card
st.markdown(f"""
<div class='match-header'>
    <div class='team-block'>
        <div class='team-role'>Local · Equipo 1</div>
        <div class='team-name'>Equipo 1</div>
        <div class='team-xg'>{xg_home:.2f}</div>
        <div class='team-xg-lbl'>Expected Goals</div>
    </div>
    <div class='vs-block'>
        <div class='vs-txt'>VS</div>
        <div class='vs-total'>{xg_total:.2f}</div>
        <div class='vs-copa'>xG total<br>media Copa {avg_goals:.2f}</div>
    </div>
    <div class='team-block team-block-away'>
        <div class='team-role'>Visitante · Equipo 2</div>
        <div class='team-name'>Equipo 2</div>
        <div class='team-xg'>{xg_away:.2f}</div>
        <div class='team-xg-lbl'>Expected Goals</div>
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

winner = "Equipo 1" if h > d and h > a else ("Empate" if d > h and d > a else "Equipo 2")
w_prob = max(h, d, a)

col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    border = "card-winner" if h == w_prob else ""
    st.markdown(f"""
    <div class='card {border}'>
        <div class='card-lbl'>Victoria Local</div>
        <div class='card-val' style='color:{prob_color(h)};'>{h:.1%}</div>
        <div class='card-sub'>Equipo 1</div>
        {bar_html(h*100, prob_color(h))}
    </div>""", unsafe_allow_html=True)
with col2:
    border = "card-winner" if d == w_prob else ""
    st.markdown(f"""
    <div class='card {border}'>
        <div class='card-lbl'>Empate</div>
        <div class='card-val card-val-yellow'>{d:.1%}</div>
        <div class='card-sub'>Ningún ganador</div>
        {bar_html(d*100, "#ffd54f")}
    </div>""", unsafe_allow_html=True)
with col3:
    border = "card-winner" if a == w_prob else ""
    st.markdown(f"""
    <div class='card {border}'>
        <div class='card-lbl'>Victoria Visitante</div>
        <div class='card-val' style='color:{prob_color(a)};'>{a:.1%}</div>
        <div class='card-sub'>Equipo 2</div>
        {bar_html(a*100, prob_color(a))}
    </div>""", unsafe_allow_html=True)
with col4:
    bi, bj = np.unravel_index(np.argmax(mat), mat.shape)
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>Marcador Más Probable</div>
        <div class='card-val'>{bi}–{bj}</div>
        <div class='card-sub'>{mat[bi,bj]:.1%} de probabilidad</div>
        <div class='card-tag'>E1 · E2</div>
    </div>""", unsafe_allow_html=True)

# Gauge visual
fig1x2 = go.Figure(go.Bar(
    x=[h*100, d*100, a*100],
    y=["Equipo 1 gana", "Empate", "Equipo 2 gana"],
    orientation="h",
    marker_color=[prob_color(h), "#ffd54f", prob_color(a)],
    text=[f"{h:.1%}", f"{d:.1%}", f"{a:.1%}"],
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(family="Space Mono", size=14, color="#040a05"),
    width=0.55,
))
fig1x2.update_layout(
    plot_bgcolor="#0b1610", paper_bgcolor="#080e0a",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
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
    ("1X", "Equipo 1 gana o Empate", dc_1x, "#69f0ae"),
    ("12", "Cualquiera de los dos equipos gana", dc_12, "#4fc3f7"),
    ("X2", "Empate o Equipo 2 gana", dc_x2, "#ce93d8"),
]
cols_dc = st.columns(3)
for col, (code, desc, val, color) in zip(cols_dc, dc_items):
    with col:
        col.markdown(f"""
        <div class='card'>
            <div class='card-tag' style='color:{color};border-color:{color};background:transparent;font-size:0.85rem;font-weight:700;'>{code}</div>
            <div class='card-val' style='color:{color};margin-top:8px;'>{val:.1%}</div>
            <div class='card-sub' style='margin-top:4px;'>{desc}</div>
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
<p class='sec-desc'>Excluye el empate del cálculo y redistribuye su probabilidad.
Si hay empate, se anula — aquí mostramos la probabilidad relativa de que gane cada equipo.</p>
""", unsafe_allow_html=True)

cols_dnb = st.columns(2)
for col, (lbl, val, color) in zip(cols_dnb, [
    ("Equipo 1 gana · sin contar empate", dnb_h, "#69f0ae"),
    ("Equipo 2 gana · sin contar empate", dnb_a, "#ef5350"),
]):
    col.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>{lbl}</div>
        <div class='card-val' style='color:{color};'>{val:.1%}</div>
        {bar_html(val*100, color)}
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='info-pill' style='margin-top:8px;'>
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
<p class='sec-desc'>
  Probabilidad de que el total de goles supere (Over) o no alcance (Under) cada línea.
  La línea más habitual en la Copa del Mundo es la 2.5.
</p>
""", unsafe_allow_html=True)

ou_lines = [
    (0.5,  over05,  under05),
    (1.5,  over15,  under15),
    (2.5,  over25,  under25),
    (3.5,  over35,  under35),
    (4.5,  over45,  under45),
]
ou_html = "<div class='ou-grid'>"
for line, ov, un in ou_lines:
    highlight = "border-color:#69f0ae;" if line == 2.5 else ""
    ou_html += f"""
    <div class='ou-card' style='{highlight}'>
        <div class='ou-line'>{line}</div>
        <div style='height:1px;background:#1a3320;margin:6px 0;'></div>
        <div class='ou-tag-o'>Over</div>
        <div class='ou-over'>{ov:.1%}</div>
        <div style='margin:4px 0;'></div>
        <div class='ou-tag-u'>Under</div>
        <div class='ou-under'>{un:.1%}</div>
    </div>"""
ou_html += "</div>"
st.markdown(ou_html, unsafe_allow_html=True)

# Waterfall chart Over progression
fig_ou = go.Figure()
ov_vals = [v*100 for _,v,_ in ou_lines]
un_vals = [v*100 for _,_,v in ou_lines]
x_labs  = [f"O/U {l}" for l,_,_ in ou_lines]
fig_ou.add_trace(go.Scatter(
    x=x_labs, y=ov_vals, name="Over", mode="lines+markers",
    line=dict(color="#69f0ae", width=2),
    marker=dict(size=8, color="#69f0ae"),
    text=[f"{v:.1f}%" for v in ov_vals], textposition="top center",
    textfont=dict(family="Space Mono", size=10, color="#69f0ae"),
))
fig_ou.add_trace(go.Scatter(
    x=x_labs, y=un_vals, name="Under", mode="lines+markers",
    line=dict(color="#ef5350", width=2, dash="dot"),
    marker=dict(size=8, color="#ef5350"),
    text=[f"{v:.1f}%" for v in un_vals], textposition="bottom center",
    textfont=dict(family="Space Mono", size=10, color="#ef5350"),
))
fig_ou.add_hline(y=50, line_dash="dash", line_color="#1a3320", line_width=1)
fig_ou.update_layout(
    plot_bgcolor="#0b1610", paper_bgcolor="#080e0a",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    legend=dict(bgcolor="#0b1610", bordercolor="#1a3320", borderwidth=1,
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
  Las líneas asiáticas dividen la apuesta en dos mitades entre dos líneas consecutivas
  (ej: 2.25 = mitad en 2.0 y mitad en 2.5). Ofrecen mayor precisión estadística.
</p>
""", unsafe_allow_html=True)

asian_items = [
    (2.25, "Split 2.0 / 2.5", asian_225_o, asian_225_u),
    (2.75, "Split 2.5 / 3.0", asian_275_o, asian_275_u),
    (3.25, "Split 3.0 / 3.5", asian_325_o, asian_325_u),
]
cols_as = st.columns(3)
for col, (line, desc, ov, un) in zip(cols_as, asian_items):
    col.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>Línea {line}</div>
        <div class='card-sub' style='margin-bottom:10px;'>{desc}</div>
        <div style='display:flex;justify-content:space-around;'>
            <div>
                <div class='ou-tag-o'>Over</div>
                <div style='font-family:Space Mono;font-size:1.3rem;font-weight:700;color:#69f0ae;'>{ov:.1%}</div>
            </div>
            <div style='width:1px;background:#1a3320;'></div>
            <div>
                <div class='ou-tag-u'>Under</div>
                <div style='font-family:Space Mono;font-size:1.3rem;font-weight:700;color:#ef5350;'>{un:.1%}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BTTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>⚡</span>
  <span class='sec-label'>06 · Ambos Equipos Marcan (BTTS)</span>
</div>
<p class='sec-desc'>
  Probabilidad de que tanto el Equipo 1 como el Equipo 2 anoten al menos un gol.
  Se calcula de forma independiente para cada equipo con la distribución de Poisson.
</p>
""", unsafe_allow_html=True)

p_e1_marca  = 1 - poisson.pmf(0, xg_home)
p_e2_marca  = 1 - poisson.pmf(0, xg_away)

col_b1, col_b2, col_b3 = st.columns([2, 2, 3])
with col_b1:
    st.markdown(f"""
    <div class='card' style='border-color:#1e5c30;'>
        <div class='card-lbl' style='color:#69f0ae;'>✔ BTTS — Sí</div>
        <div class='card-val' style='font-size:2.4rem;'>{btts_y:.1%}</div>
        <div class='card-sub'>Ambos equipos marcan</div>
        {bar_html(btts_y*100)}
    </div>""", unsafe_allow_html=True)
with col_b2:
    st.markdown(f"""
    <div class='card' style='border-color:#3a1010;'>
        <div class='card-lbl' style='color:#ef5350;'>✘ BTTS — No</div>
        <div class='card-val card-val-red' style='font-size:2.4rem;'>{btts_n:.1%}</div>
        <div class='card-sub'>Al menos uno no marca</div>
        {bar_html(btts_n*100, "#ef5350")}
    </div>""", unsafe_allow_html=True)
with col_b3:
    st.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>Probabilidad individual de marcar</div>
        <div style='margin-top:12px;'>
            <div class='prob-bar-row'>
                <div class='prob-bar-lbl'>Equipo 1 marca</div>
                <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{p_e1_marca*100:.1f}%;background:#69f0ae;'></div></div>
                <div class='prob-bar-num'>{p_e1_marca:.1%}</div>
            </div>
            <div class='prob-bar-row'>
                <div class='prob-bar-lbl'>Equipo 2 marca</div>
                <div class='prob-bar-track'><div class='prob-bar-fill' style='width:{p_e2_marca*100:.1f}%;background:#4fc3f7;'></div></div>
                <div class='prob-bar-num' style='color:#4fc3f7;'>{p_e2_marca:.1%}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# BTTS × Resultado combinado
st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
combos = [
    ("Equipo 1 gana + BTTS Sí",  h * btts_y,  "#69f0ae"),
    ("Equipo 1 gana + BTTS No",  h * btts_n,  "#388e3c"),
    ("Empate + BTTS Sí",         d * btts_y,  "#ffd54f"),
    ("Empate + BTTS No",         d * btts_n,  "#f9a825"),
    ("Equipo 2 gana + BTTS Sí",  a * btts_y,  "#ef5350"),
    ("Equipo 2 gana + BTTS No",  a * btts_n,  "#b71c1c"),
]
combos.sort(key=lambda x: x[1], reverse=True)

combo_html = "<div class='mkt-block'>"
for name, val, color in combos:
    combo_html += f"""
    <div class='mkt-row'>
        <div><div class='mkt-name'>{name}</div></div>
        <div style='display:flex;align-items:center;gap:12px;'>
            <div style='width:120px;height:6px;background:#1a3320;border-radius:3px;overflow:hidden;'>
                <div style='width:{val*100:.1f}%;height:100%;background:{color};border-radius:3px;'></div>
            </div>
            <div class='mkt-val' style='color:{color};'>{val:.1%}</div>
        </div>
    </div>"""
combo_html += "</div>"
st.markdown(combo_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GOLES TOTALES EXACTOS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🔢</span>
  <span class='sec-label'>07 · Goles Totales Exactos</span>
</div>
<p class='sec-desc'>Probabilidad de que el partido termine con exactamente N goles en total (ambos equipos).</p>
""", unsafe_allow_html=True)

eg_labels = [f"{k} gol{'es' if k!=1 else ''}" for k in exact.keys()]
eg_values = [v * 100 for v in exact.values()]
peak_g    = max(exact, key=exact.get)
colors_eg = ["#69f0ae" if k == peak_g else "#1e4d2b" for k in exact.keys()]

fig_eg = go.Figure(go.Bar(
    x=eg_labels, y=eg_values,
    marker_color=colors_eg,
    marker_line_color="#0b1610", marker_line_width=2,
    text=[f"{v:.1f}%" for v in eg_values],
    textposition="outside",
    textfont=dict(family="Space Mono", size=11, color="#a5d6a7"),
))
fig_eg.add_annotation(
    x=f"{peak_g} gol{'es' if peak_g!=1 else ''}",
    y=exact[peak_g]*100 + 3,
    text="★ más probable",
    showarrow=False,
    font=dict(family="Space Mono", size=10, color="#69f0ae"),
    yshift=12,
)
fig_eg.update_layout(
    plot_bgcolor="#0b1610", paper_bgcolor="#080e0a",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
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
# 8. TOP 5 MARCADORES + HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>🎯</span>
  <span class='sec-label'>08 · Marcadores Exactos</span>
</div>
<p class='sec-desc'>
  Los 5 marcadores más probables y el mapa de calor completo de la matriz de Poisson (E1 × E2).
  Cada celda muestra la probabilidad de ese marcador exacto.
</p>
""", unsafe_allow_html=True)

# Top 5
top5_html = "<div class='top5-grid'>"
for rank, (prob, i, j) in enumerate(top5):
    cls = "top5-card-1" if rank == 0 else ""
    top5_html += f"""
    <div class='top5-card {cls}'>
        <div class='top5-rank'>#{rank+1}</div>
        <div class='top5-score'>{i}–{j}</div>
        <div class='top5-prob'>{prob:.1%}</div>
        <div class='card-sub' style='font-size:0.68rem;'>E1 – E2</div>
    </div>"""
top5_html += "</div>"
st.markdown(top5_html, unsafe_allow_html=True)

# Heatmap
SHOW = 6
z = mat[:SHOW, :SHOW] * 100
text_mat = [[f"{z[i][j]:.1f}%" for j in range(SHOW)] for i in range(SHOW)]
fig_heat = go.Figure(go.Heatmap(
    z=z,
    x=[f"E2 · {j}" for j in range(SHOW)],
    y=[f"E1 · {i}" for i in range(SHOW)],
    colorscale=[[0,"#080e0a"],[0.2,"#0d2b14"],[0.5,"#1b5e20"],[0.8,"#388e3c"],[1.0,"#69f0ae"]],
    text=text_mat, texttemplate="%{text}",
    textfont=dict(family="Space Mono", size=11, color="#e8f5e9"),
    showscale=True,
    colorbar=dict(
        tickfont=dict(family="Space Mono", color="#4caf50", size=10),
        bgcolor="#0b1610", bordercolor="#1a3320", thickness=12,
        title=dict(text="%", font=dict(color="#4caf50", size=10)),
    ),
))
fig_heat.update_layout(
    plot_bgcolor="#0b1610", paper_bgcolor="#080e0a",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    xaxis=dict(title="Goles Equipo 2", showgrid=False, side="top"),
    yaxis=dict(title="Goles Equipo 1", showgrid=False, autorange="reversed"),
    margin=dict(t=40, b=10, l=10, r=10), height=400,
)
st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MARGEN DE VICTORIA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>📐</span>
  <span class='sec-label'>09 · Margen de Victoria</span>
</div>
<p class='sec-desc'>
  Diferencia de goles entre los equipos al final del partido.
  Los valores positivos favorecen al Equipo 1 y los negativos al Equipo 2.
</p>
""", unsafe_allow_html=True)

mg_items = sorted(margins.items())
mg_x, mg_y, mg_c, mg_t = [], [], [], []
for diff, p in mg_items:
    if diff > 0:   label = f"E1 +{diff}";  color = "#69f0ae"
    elif diff < 0: label = f"E2 +{abs(diff)}"; color = "#ef5350"
    else:           label = "Empate";        color = "#ffd54f"
    mg_x.append(label); mg_y.append(p*100); mg_c.append(color); mg_t.append(f"{p:.1%}")

fig_mg = go.Figure(go.Bar(
    x=mg_x, y=mg_y, marker_color=mg_c,
    marker_line_color="#080e0a", marker_line_width=2,
    text=mg_t, textposition="outside",
    textfont=dict(family="Space Mono", size=10, color="#a5d6a7"),
))
fig_mg.update_layout(
    plot_bgcolor="#0b1610", paper_bgcolor="#080e0a",
    font=dict(family="Space Grotesk", color="#a5d6a7"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(mg_y)*1.35]),
    margin=dict(t=10, b=10, l=10, r=10), height=250,
)
st.plotly_chart(fig_mg, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PUNTOS ESPERADOS (xPts)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>📈</span>
  <span class='sec-label'>10 · Puntos Esperados (xPts)</span>
</div>
<p class='sec-desc'>
  Valor esperado de puntos que obtendría cada equipo si se jugase este partido muchas veces.
  Se calcula como 3 × P(victoria) + 1 × P(empate). El máximo posible es 3.00 pts.
</p>
""", unsafe_allow_html=True)

col_xp1, col_xp2 = st.columns(2)
xp_max = 3.0
for col, (team, xp, color) in zip(
    [col_xp1, col_xp2],
    [("Equipo 1 · Local", exp_h, "#69f0ae"), ("Equipo 2 · Visitante", exp_a, "#4fc3f7")]
):
    col.markdown(f"""
    <div class='card'>
        <div class='card-lbl'>{team}</div>
        <div class='card-val' style='color:{color};font-size:2.8rem;'>{xp:.2f}</div>
        <div class='card-sub'>puntos esperados de 3.00 posibles</div>
        <div style='margin-top:12px;height:16px;background:#1a3320;border-radius:8px;overflow:hidden;'>
            <div style='width:{xp/xp_max*100:.1f}%;height:100%;background:{color};border-radius:8px;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:4px;'>
            <span style='font-size:0.65rem;color:#2e7d32;font-family:Space Mono;'>0</span>
            <span style='font-size:0.65rem;color:#2e7d32;font-family:Space Mono;'>{xp:.2f} / 3.00</span>
        </div>
    </div>""", unsafe_allow_html=True)

# Desglose xPts
st.markdown(f"""
<div class='info-pill' style='margin-top:12px;'>
    <div>E1 · 3×P(win) = <span>{3*h:.2f}</span></div>
    <div>E1 · 1×P(draw) = <span>{d:.2f}</span></div>
    <div>E2 · 3×P(win) = <span>{3*a:.2f}</span></div>
    <div>E2 · 1×P(draw) = <span>{d:.2f}</span></div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TABLA RESUMEN COMPLETA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='sec-header'>
  <span class='sec-icon'>📋</span>
  <span class='sec-label'>11 · Resumen Completo de Mercados</span>
</div>
<p class='sec-desc'>Todos los mercados en una sola tabla para consulta rápida.</p>
""", unsafe_allow_html=True)

bi, bj = np.unravel_index(np.argmax(mat), mat.shape)
summary_rows = [
    ("RESULTADO FINAL",  "Victoria Equipo 1",     f"{h:.1%}",         "1X2"),
    ("RESULTADO FINAL",  "Empate",                f"{d:.1%}",         "1X2"),
    ("RESULTADO FINAL",  "Victoria Equipo 2",     f"{a:.1%}",         "1X2"),
    ("RESULTADO FINAL",  "Marcador más probable", f"{bi}–{bj} ({mat[bi,bj]:.1%})", "Score"),
    ("DOBLE OPORTUNIDAD","1X — E1 o Empate",      f"{dc_1x:.1%}",     "DC"),
    ("DOBLE OPORTUNIDAD","12 — Cualquier ganador",f"{dc_12:.1%}",     "DC"),
    ("DOBLE OPORTUNIDAD","X2 — Empate o E2",      f"{dc_x2:.1%}",     "DC"),
    ("DRAW NO BET",      "DNB Equipo 1",          f"{dnb_h:.1%}",     "DNB"),
    ("DRAW NO BET",      "DNB Equipo 2",          f"{dnb_a:.1%}",     "DNB"),
    ("OVER / UNDER",     "Over 0.5",              f"{over05:.1%}",    "O/U"),
    ("OVER / UNDER",     "Under 0.5",             f"{under05:.1%}",   "O/U"),
    ("OVER / UNDER",     "Over 1.5",              f"{over15:.1%}",    "O/U"),
    ("OVER / UNDER",     "Under 1.5",             f"{under15:.1%}",   "O/U"),
    ("OVER / UNDER",     "Over 2.5",              f"{over25:.1%}",    "O/U"),
    ("OVER / UNDER",     "Under 2.5",             f"{under25:.1%}",   "O/U"),
    ("OVER / UNDER",     "Over 3.5",              f"{over35:.1%}",    "O/U"),
    ("OVER / UNDER",     "Under 3.5",             f"{under35:.1%}",   "O/U"),
    ("OVER / UNDER",     "Over 4.5",              f"{over45:.1%}",    "O/U"),
    ("OVER / UNDER",     "Under 4.5",             f"{under45:.1%}",   "O/U"),
    ("ASIAN O/U",        "Asian O/U 2.25 Over",   f"{asian_225_o:.1%}","A O/U"),
    ("ASIAN O/U",        "Asian O/U 2.25 Under",  f"{asian_225_u:.1%}","A O/U"),
    ("ASIAN O/U",        "Asian O/U 2.75 Over",   f"{asian_275_o:.1%}","A O/U"),
    ("ASIAN O/U",        "Asian O/U 2.75 Under",  f"{asian_275_u:.1%}","A O/U"),
    ("ASIAN O/U",        "Asian O/U 3.25 Over",   f"{asian_325_o:.1%}","A O/U"),
    ("ASIAN O/U",        "Asian O/U 3.25 Under",  f"{asian_325_u:.1%}","A O/U"),
    ("BTTS",             "Ambos marcan — Sí",     f"{btts_y:.1%}",    "BTTS"),
    ("BTTS",             "Ambos marcan — No",     f"{btts_n:.1%}",    "BTTS"),
    ("GOLES EXACTOS",    f"Goles totales: {peak_g} (más probable)", f"{exact[peak_g]:.1%}", "Exact"),
    ("PUNTOS ESPERADOS", "xPts Equipo 1",         f"{exp_h:.2f} pts", "xPts"),
    ("PUNTOS ESPERADOS", "xPts Equipo 2",         f"{exp_a:.2f} pts", "xPts"),
]

df = pd.DataFrame(summary_rows, columns=["Categoría", "Mercado", "Probabilidad", "Tipo"])
st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Categoría":    st.column_config.TextColumn("Categoría",    width="medium"),
        "Mercado":      st.column_config.TextColumn("Mercado",      width="large"),
        "Probabilidad": st.column_config.TextColumn("Probabilidad", width="small"),
        "Tipo":         st.column_config.TextColumn("Tipo",         width="small"),
    },
)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:40px 0 20px;border-top:1px solid #1a3320;margin-top:32px;'>
    <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#2e7d32;letter-spacing:2px;'>
        COPA DEL MUNDO · ANÁLISIS xG · MODELO POISSON BIVARIADA
    </div>
    <div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#1a3320;margin-top:6px;'>
        λ₁ = {xg_home:.2f} · λ₂ = {xg_away:.2f} · Media Copa {avg_goals:.2f} goles/partido · xG total {xg_total:.2f}
    </div>
</div>""", unsafe_allow_html=True)
ENDOFFILE
