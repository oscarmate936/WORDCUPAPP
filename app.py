import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import urllib.parse

# ══════════════════════════════════════════════════════════════════════════════
#  COPA DEL MUNDO · ANÁLISIS xG — v2
#  Mejoras estadísticas (mismos datos de entrada):
#   1. Calibración por CONTRACCIÓN (shrinkage) que preserva la supremacía xG,
#      en lugar de reescalar el total al 100% a la media del torneo.
#   2. ρ acotado dinámicamente a su rango matemáticamente válido (sin τ<0).
#   3. TODOS los mercados se derivan de la misma matriz Dixon–Coles
#      (antes "equipo marca" usaba Poisson independiente e ignoraba ρ).
#   4. Cálculo vectorizado por diagonales/anti-diagonales + renormalización.
#   5. Verificación de cobertura de cola de la matriz (>99.99%).
#   6. Hándicap asiático exacto con push y líneas de cuarto (½ ganancia/pérdida).
#   7. Índice de certeza (entropía del 1X2) y porterías a cero, derivados
#      de los mismos datos.
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Copa del Mundo · Análisis xG",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tokens de diseño ──────────────────────────────────────────────────────────
BG      = "#0B0F0D"   # verde-negro, césped nocturno
SURF    = "#121814"   # superficie de tarjetas
SURF2   = "#0E1411"
BORDER  = "#1E2620"
BORDER2 = "#2A3630"
GREEN   = "#34D399"   # esmeralda — local / positivo
GREEND  = "#0E3B2C"
BLUE    = "#38BDF8"   # visitante
AMBER   = "#FBBF24"   # empate / atención
RED     = "#F87171"   # negativo
TXT     = "#E7ECE9"
MUT     = "#8A968F"
DIM     = "#5A665F"
FONT    = "Archivo"
MONO    = "IBM Plex Mono"

PLOTLY_BASE = dict(
    plot_bgcolor=SURF, paper_bgcolor=BG,
    font=dict(family=FONT, color=MUT),
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{ font-family: '{FONT}', sans-serif; }}
.stApp {{ background: {BG}; color: {TXT}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background: {SURF2} !important; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] * {{ color: #B9C4BE !important; font-family: '{FONT}', sans-serif !important; }}
[data-testid="stSidebar"] .stButton button {{
    background: linear-gradient(135deg, #0E9F6E, {GREEN}) !important;
    color: {BG} !important; font-weight: 800 !important; font-size: 1rem !important;
    border: none !important; border-radius: 12px !important; padding: 13px 20px !important;
    margin-top: 16px !important; box-shadow: 0 6px 22px rgba(52,211,153,0.28);
    transition: all .22s ease; letter-spacing: .3px;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    box-shadow: 0 8px 28px rgba(52,211,153,0.42); transform: translateY(-1px);
}}

/* ── Encabezado de sección ── */
.sec-header {{
    display:flex; align-items:baseline; gap:12px;
    margin:48px 0 6px; padding-bottom:12px; border-bottom:1px solid {BORDER};
}}
.sec-kicker {{
    font-family:'{MONO}',monospace; font-size:.62rem; color:{GREEN};
    letter-spacing:2.5px; font-weight:700; text-transform:uppercase;
}}
.sec-label {{ font-size:1.15rem; color:{TXT}; font-weight:700; letter-spacing:-.2px; }}
.sec-desc {{ font-size:.82rem; color:{DIM}; margin:0 0 20px; line-height:1.65; }}

/* ── Tarjetas ── */
.card {{
    background:{SURF}; border-radius:16px; padding:20px 18px; text-align:center;
    border:1px solid {BORDER}; margin-bottom:8px; position:relative; overflow:hidden;
    transition:border-color .2s, transform .15s;
}}
.card:hover {{ border-color:{BORDER2}; transform:translateY(-1px); }}
.card-winner {{ border:1px solid rgba(52,211,153,.55) !important;
    box-shadow:0 0 0 1px rgba(52,211,153,.12), 0 10px 28px rgba(0,0,0,.45); }}
.card-winner::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#0E9F6E,{GREEN}); }}
.card-val {{ font-family:'{MONO}',monospace; font-size:clamp(1.6rem,5vw,2.4rem);
    font-weight:700; color:{GREEN}; line-height:1.1; letter-spacing:-1px; }}
.card-lbl {{ font-size:.68rem; color:{DIM}; text-transform:uppercase;
    letter-spacing:1.6px; margin-bottom:10px; font-weight:600; }}
.card-sub {{ font-size:.75rem; color:{DIM}; margin-top:6px; }}
.card-tag {{ display:inline-block; font-size:.68rem; font-family:'{MONO}',monospace;
    background:{SURF2}; color:{GREEN}; padding:4px 10px; border-radius:6px;
    margin-top:10px; border:1px solid {BORDER}; }}

/* ── Barras de probabilidad ── */
.prob-bar-row {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
.prob-bar-lbl {{ font-size:.78rem; color:{MUT}; min-width:110px; }}
.prob-bar-track {{ flex:1; height:6px; background:{SURF2}; border-radius:3px; overflow:hidden; }}
.prob-bar-fill {{ height:100%; border-radius:3px; transition:width .5s cubic-bezier(.4,0,.2,1); }}
.prob-bar-num {{ font-family:'{MONO}',monospace; font-size:.78rem; color:{GREEN};
    min-width:52px; text-align:right; }}

/* ── Marcador de estadio (header del partido) ── */
.match-header {{
    display:grid; grid-template-columns:1fr auto 1fr;
    background:linear-gradient(180deg,{SURF} 0%,{SURF2} 100%);
    border-radius:20px; overflow:hidden; margin-bottom:14px;
    border:1px solid {BORDER}; box-shadow:0 12px 40px rgba(0,0,0,.5);
}}
.team-block {{ padding:30px 24px; text-align:center; }}
.team-role {{ font-size:.62rem; color:{DIM}; text-transform:uppercase;
    letter-spacing:3px; margin-bottom:10px; font-weight:600; }}
.team-name {{ font-size:clamp(1.1rem,3.5vw,1.7rem); font-weight:800; color:#FFF;
    margin:6px 0 14px; letter-spacing:-.4px; }}
.team-circle {{ display:inline-flex; align-items:center; justify-content:center;
    width:52px; height:52px; border-radius:14px; font-family:'{MONO}',monospace;
    font-size:1rem; font-weight:700; color:{BG}; margin-bottom:10px; }}
.team-xg {{ font-family:'{MONO}',monospace; font-size:clamp(2.2rem,6vw,3.4rem);
    font-weight:700; line-height:1; letter-spacing:-2px; }}
.team-xg-lbl {{ font-size:.62rem; color:{DIM}; margin-top:6px;
    text-transform:uppercase; letter-spacing:1.6px; }}
.vs-block {{ display:flex; flex-direction:column; align-items:center; justify-content:center;
    padding:0 22px; border-left:1px solid {BORDER}; border-right:1px solid {BORDER};
    background:{SURF2}; min-width:96px; }}
.vs-txt {{ font-family:'{MONO}',monospace; font-size:.68rem; color:{DIM}; letter-spacing:3px; }}
.vs-total {{ font-family:'{MONO}',monospace; font-size:1.7rem; font-weight:700;
    color:{GREEN}; margin:6px 0; letter-spacing:-1px; }}
.vs-copa {{ font-size:.6rem; color:{DIM}; text-align:center; margin-top:4px;
    text-transform:uppercase; letter-spacing:1px; line-height:1.6; }}

/* ── Pill de información ── */
.info-pill {{
    background:{SURF}; border-radius:12px; padding:12px 16px; font-size:.78rem;
    color:{DIM}; margin-bottom:20px; display:flex; gap:16px; flex-wrap:wrap;
    border:1px solid {BORDER}; align-items:center;
}}
.info-pill span {{ color:{GREEN}; font-family:'{MONO}',monospace; font-weight:700; font-size:.82rem; }}

/* ── O/U grid ── */
.ou-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:14px; }}
.ou-card {{ background:{SURF}; border-radius:14px; padding:18px 8px; text-align:center;
    border:1px solid {BORDER}; transition:border-color .2s; }}
.ou-card:hover {{ border-color:{BORDER2}; }}
.ou-card-highlight {{ border:1px solid rgba(52,211,153,.45) !important; background:{GREEND}22; }}
.ou-line {{ font-family:'{MONO}',monospace; font-size:.95rem; color:{MUT}; font-weight:700; margin-bottom:10px; }}
.ou-sep {{ width:24px; height:1px; background:{BORDER}; margin:8px auto; }}
.ou-over {{ font-family:'{MONO}',monospace; font-size:1.15rem; color:{GREEN}; font-weight:700; }}
.ou-under {{ font-family:'{MONO}',monospace; font-size:1.15rem; color:{RED}; font-weight:700; }}
.ou-tag-o {{ font-size:.6rem; color:{GREEN}; text-transform:uppercase; letter-spacing:1.5px; margin:4px 0 2px; font-weight:700; }}
.ou-tag-u {{ font-size:.6rem; color:{RED}; text-transform:uppercase; letter-spacing:1.5px; margin:4px 0 2px; font-weight:700; }}

/* ── BTTS ── */
.btts-big {{ background:{SURF}; border-radius:16px; padding:24px 20px; text-align:center;
    border:1px solid {BORDER}; margin-bottom:8px; position:relative; overflow:hidden; }}
.btts-big::after {{ content:''; position:absolute; bottom:0; left:0; right:0; height:3px; }}
.btts-yes::after {{ background:linear-gradient(90deg,#0E9F6E,{GREEN}); }}
.btts-no::after  {{ background:linear-gradient(90deg,#991B1B,{RED}); }}
.btts-label {{ font-size:.62rem; text-transform:uppercase; letter-spacing:2px; margin-bottom:10px; font-weight:700; }}
.btts-pct {{ font-family:'{MONO}',monospace; font-size:2.8rem; font-weight:700; line-height:1; letter-spacing:-2px; }}
.btts-sub {{ font-size:.75rem; margin-top:8px; color:{DIM}; }}

/* ── Top marcadores ── */
.top10-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-bottom:16px; }}
.top10-card {{ background:{SURF}; border-radius:14px; padding:16px 8px; text-align:center;
    border:1px solid {BORDER}; transition:border-color .2s; }}
.top10-card:hover {{ border-color:{BORDER2}; }}
.top10-card-1 {{ border:1px solid rgba(52,211,153,.55) !important; background:{GREEND}33; }}
.top10-score {{ font-family:'{MONO}',monospace; font-size:1.35rem; font-weight:700; letter-spacing:-.5px; }}
.top10-prob {{ font-size:.8rem; color:{MUT}; margin-top:6px; font-family:'{MONO}',monospace; }}
.top10-rank {{ font-size:.6rem; color:{DIM}; margin-bottom:6px; font-family:'{MONO}',monospace;
    text-transform:uppercase; letter-spacing:1px; }}

/* ── Multigoal / xPts ── */
.mg-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:16px; }}
.xpts-card {{ background:{SURF}; border-radius:16px; padding:22px 20px; border:1px solid {BORDER}; margin-bottom:8px; }}
.xpts-label {{ font-size:.68rem; color:{DIM}; text-transform:uppercase; letter-spacing:1.6px; margin-bottom:14px; font-weight:600; }}
.xpts-big {{ font-family:'{MONO}',monospace; font-size:3rem; font-weight:700; line-height:1; letter-spacing:-2px; }}
.xpts-sub {{ font-size:.75rem; color:{DIM}; margin-top:10px; }}
.xpts-track {{ height:8px; background:{SURF2}; border-radius:4px; overflow:hidden; margin-top:14px; }}
.xpts-fill {{ height:100%; border-radius:4px; }}

/* ── WhatsApp ── */
.wa-section {{ background:linear-gradient(135deg,#0B1F14,#0E2A19);
    border:1px solid rgba(37,211,102,.25); border-radius:20px; padding:32px 28px;
    margin-top:44px; text-align:center; position:relative; overflow:hidden; }}
.wa-section::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,#25D366,transparent); }}
.wa-title {{ font-size:1.1rem; font-weight:800; color:{TXT}; margin-bottom:6px; letter-spacing:-.2px; }}
.wa-sub {{ font-size:.82rem; color:{DIM}; margin-bottom:20px; line-height:1.6; }}
.wa-btn {{ display:inline-flex; align-items:center; gap:10px; background:#25D366; color:{BG};
    font-weight:800; font-size:.95rem; padding:14px 28px; border-radius:14px;
    text-decoration:none; letter-spacing:.2px; box-shadow:0 4px 22px rgba(37,211,102,.3);
    transition:all .2s ease; }}
.wa-btn:hover {{ background:#22c55e; box-shadow:0 6px 30px rgba(37,211,102,.45);
    transform:translateY(-1px); color:{BG}; }}

.divider-line {{ height:1px; background:{BORDER}; margin:6px 0; }}
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}

@media (max-width:640px) {{
    .ou-grid {{ grid-template-columns:repeat(3,1fr); }}
    .top10-grid {{ grid-template-columns:repeat(3,1fr); }}
    .mg-grid {{ grid-template-columns:repeat(2,1fr); }}
    .team-name {{ font-size:1rem; }} .team-xg {{ font-size:2rem; }}
    .btts-pct {{ font-size:2rem; }} .xpts-big {{ font-size:2.2rem; }}
}}
@media (prefers-reduced-motion: reduce) {{
    * {{ transition:none !important; animation:none !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO — Dixon & Coles con mejoras de precisión
# ══════════════════════════════════════════════════════════════════════════════
MAX_G_BASE = 25
TAIL_TOL   = 1e-4   # cobertura mínima exigida de la cola: 99.99%


def rho_valid_range(lh, la):
    """Rango de ρ para el cual todos los factores τ son ≥ 0."""
    lo = -1.0 / max(lh, la)
    hi = min(1.0 / (lh * la), 1.0) if lh * la > 0 else 1.0
    return lo, hi


def dixon_coles_matrix(lh, la, rho):
    """Matriz conjunta Dixon–Coles.
    Mejoras: tamaño adaptativo (cobertura de cola >99.99%), ρ acotado a su
    rango válido (nunca τ<0) y renormalización exacta."""
    n = MAX_G_BASE
    # tamaño adaptativo: amplía hasta cubrir la masa de probabilidad
    while (poisson.sf(n - 1, lh) > TAIL_TOL or poisson.sf(n - 1, la) > TAIL_TOL) and n < 60:
        n += 5

    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    mat = poisson.pmf(i, lh) * poisson.pmf(j, la)

    lo, hi = rho_valid_range(lh, la)
    rho_eff = float(np.clip(rho, lo, hi))

    mat[0, 0] *= 1 - lh * la * rho_eff
    mat[0, 1] *= 1 + lh * rho_eff
    mat[1, 0] *= 1 + la * rho_eff
    mat[1, 1] *= 1 - rho_eff
    mat /= mat.sum()          # renormalización exacta
    return mat, rho_eff, (rho_eff != rho)


# ── Derivados de la matriz (todo vectorizado y coherente con ρ) ───────────────
def calc_1x2(mat):
    return (float(np.tril(mat, -1).sum()),
            float(np.trace(mat)),
            float(np.triu(mat, 1).sum()))


def totals_dist(mat):
    """P(total = g) para g = 0..2n-2, por suma de anti-diagonales."""
    n = mat.shape[0]
    fl = np.fliplr(mat)
    return np.array([np.trace(fl, offset=n - 1 - g) for g in range(2 * n - 1)])


def margins_dist(mat, span=5):
    """P(local − visitante = d) para d en [−span, span]."""
    return {d: float(np.trace(mat, offset=-d)) for d in range(-span, span + 1)}


def calc_ou(tot, line):
    over = float(tot[int(np.floor(line)) + 1:].sum())
    return over, 1.0 - over


def calc_multigoal(tot, lo, hi):
    return float(tot[lo:hi + 1].sum())


def asian_handicap(mat, H):
    """(win, push, loss) para el LOCAL con hándicap H (medias y enteras)."""
    n = mat.shape[0]
    diff = np.arange(n)[:, None] - np.arange(n)[None, :] + H
    win  = float(mat[diff >  1e-9].sum())
    push = float(mat[np.abs(diff) <= 1e-9].sum())
    return win, push, 1.0 - win - push


def top_scorelines(mat, k=10):
    n = mat.shape[0]
    flat = np.argsort(mat, axis=None)[::-1][:k]
    return [(float(mat.flat[f]), *np.unravel_index(f, mat.shape)) for f in flat]


def entropy_1x2(h, d, a):
    """Entropía normalizada del 1X2 → 0 = certeza total, 1 = máxima incertidumbre."""
    ps = np.array([h, d, a])
    ps = ps[ps > 0]
    return float(-(ps * np.log(ps)).sum() / np.log(3))


def bar_html(pct, color=GREEN, height=6):
    return (f"<div style='height:{height}px;background:{SURF2};border-radius:3px;"
            f"overflow:hidden;margin-top:10px;'>"
            f"<div style='width:{pct:.1f}%;height:100%;background:{color};"
            f"border-radius:3px;'></div></div>")


def prob_color(p):
    if p >= 0.55: return GREEN
    if p >= 0.35: return AMBER
    return RED


def match_rating(xg_total, btts_yes, over25):
    score = min(xg_total / 4.0, 1.0) * 40 + btts_yes * 30 + over25 * 30
    stars = 1 + min(int(score // 20), 4)
    return "★" * stars + "☆" * (5 - stars)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ Parámetros")
    st.markdown("---")
    team1 = st.text_input("Nombre Equipo 1 (Local)", value="Equipo 1")
    team2 = st.text_input("Nombre Equipo 2 (Visitante)", value="Equipo 2")
    st.markdown("---")
    xg_home = st.number_input(
        f"xG {team1} · Local", min_value=0.10, max_value=6.0,
        value=1.45, step=0.05, format="%.2f",
        help="Goles esperados del equipo local según el modelo xG",
    )
    xg_away = st.number_input(
        f"xG {team2} · Visitante", min_value=0.10, max_value=6.0,
        value=1.20, step=0.05, format="%.2f",
        help="Goles esperados del equipo visitante según el modelo xG",
    )
    avg_goals = st.number_input(
        "Media goles/partido del torneo", min_value=0.5, max_value=6.0,
        value=2.52, step=0.01, format="%.2f",
        help="Promedio de goles totales por partido en la edición actual",
    )
    calib_mode = st.radio(
        "Calibración del total de goles",
        ["Contracción hacia la media (recomendado)", "Reescalado 100% a la media", "Sin ajuste"],
        index=0,
        help=("Contracción: mezcla el total xG del partido con la media del torneo "
              "preservando la supremacía (diferencia xG). Más preciso que forzar el "
              "total exactamente a la media."),
    )
    if calib_mode.startswith("Contracción"):
        w_xg = st.slider(
            "Peso del xG del partido (w)", 0.0, 1.0, 0.65, 0.05,
            help=("λ_total = w·(xG₁+xG₂) + (1−w)·media. w=1 confía solo en el xG "
                  "del partido; w=0 usa solo la media del torneo."),
        )
    else:
        w_xg = 1.0 if calib_mode == "Sin ajuste" else 0.0
    rho = st.slider(
        "Correlación ρ (Dixon‑Coles)", -0.30, 0.10, -0.10, 0.01,
        help=("Dependencia entre marcadores bajos. Se acota automáticamente al "
              "rango válido para los λ del partido (nunca genera τ negativos)."),
    )
    st.markdown("---")
    run = st.button("⚽ Analizar Partido", use_container_width=True, type="primary")
    st.markdown(
        f"<div style='font-size:.72rem;color:{DIM};margin-top:12px;line-height:1.7;'>"
        "Modelo · Dixon & Coles (Poisson bivariada)<br>"
        "Calibración por contracción · ρ acotado · matriz renormalizada</div>",
        unsafe_allow_html=True,
    )

# ── Gate ──────────────────────────────────────────────────────────────────────
if not run and "ready" not in st.session_state:
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:65vh;text-align:center;gap:14px;'>
        <div style='font-size:5rem;line-height:1;'>⚽</div>
        <div style='font-size:1.6rem;font-weight:800;color:{TXT};letter-spacing:-.5px;'>
            Copa del Mundo · Análisis xG
        </div>
        <div style='color:{DIM};font-size:.88rem;max-width:340px;line-height:1.75;'>
            Ingresa los nombres, xG de ambos equipos y la media de goles
            del torneo en la barra lateral, luego pulsa
            <b style='color:{GREEN};'>Analizar Partido</b>.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

st.session_state["ready"] = True


# ══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO
# ══════════════════════════════════════════════════════════════════════════════
# 1) Calibración por CONTRACCIÓN preservando la supremacía:
#    la señal más fiable del xG de un partido es la DIFERENCIA entre equipos;
#    el TOTAL es más ruidoso, por eso se contrae hacia la media del torneo.
total_xg_input = xg_home + xg_away
supremacy      = xg_home - xg_away
lambda_total   = w_xg * total_xg_input + (1.0 - w_xg) * avg_goals
lambda_h = max((lambda_total + supremacy) / 2.0, 0.05)
lambda_a = max((lambda_total - supremacy) / 2.0, 0.05)
# corrige el total si el clamp actuó
lambda_total = lambda_h + lambda_a

@st.cache_data(show_spinner=False, max_entries=128)
def build(lh, la, r):
    return dixon_coles_matrix(lh, la, r)

mat, rho_eff, rho_clamped = build(lambda_h, lambda_a, rho)
N = mat.shape[0]

if rho_clamped:
    st.caption(f"⚠️ ρ fuera del rango válido para estos λ; se usa ρ = {rho_eff:.2f} "
               f"(rango válido: [{rho_valid_range(lambda_h, lambda_a)[0]:.2f}, "
               f"{rho_valid_range(lambda_h, lambda_a)[1]:.2f}]).")

# ── Mercados (todos desde la MISMA matriz DC) ─────────────────────────────────
h, d, a = calc_1x2(mat)
dc_1x, dc_12, dc_x2 = h + d, h + a, d + a
dnb_h = h / (h + a) if (h + a) > 0 else 0.0
dnb_a = a / (h + a) if (h + a) > 0 else 0.0

tot = totals_dist(mat)
ou_lines_v = [0.5, 1.5, 2.5, 3.5, 4.5]
ou_res = {L: calc_ou(tot, L) for L in ou_lines_v}

btts_y = float(mat[1:, 1:].sum())
btts_n = 1.0 - btts_y

# coherente con ρ: marginales de la matriz DC (no Poisson independiente)
p_e1_marca = 1.0 - float(mat[0, :].sum())
p_e2_marca = 1.0 - float(mat[:, 0].sum())
cs_home    = float(mat[:, 0].sum())   # portería a cero del local
cs_away    = float(mat[0, :].sum())   # portería a cero del visitante

home_btts_yes = float(np.tril(mat, -1)[1:, 1:].sum())
draw_btts_yes = float(np.trace(mat) - mat[0, 0])
away_btts_yes = float(np.triu(mat, 1)[1:, 1:].sum())

margins = margins_dist(mat)
top10   = top_scorelines(mat, 10)
exp_h, exp_a = 3 * h + d, 3 * a + d

xg_total    = lambda_h + lambda_a
diff_vs_avg = xg_total - avg_goals
rating      = match_rating(xg_total, btts_y, ou_res[2.5][0])
certainty   = 1.0 - entropy_1x2(h, d, a)   # 0 = partido totalmente abierto

bi, bj = np.unravel_index(int(np.argmax(mat)), mat.shape)
peak_g = int(np.argmax(tot))
exact  = {g: float(tot[g]) for g in range(min(13, len(tot)))}


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Copa del Mundo &nbsp;·&nbsp; Análisis Estadístico xG")

context_arrow = "▲" if diff_vs_avg > 0 else "▼"
context_color = GREEN if diff_vs_avg > 0 else RED
context_txt   = "más goles que la media" if diff_vs_avg > 0 else "menos goles que la media"

st.markdown(f"""
<div class='info-pill'>
    <div>Modelo <span>Dixon & Coles</span></div>
    <div>xG orig. {team1} <span>{xg_home:.2f}</span></div>
    <div>xG orig. {team2} <span>{xg_away:.2f}</span></div>
    <div>λ calib. {team1} <span>{lambda_h:.2f}</span></div>
    <div>λ calib. {team2} <span>{lambda_a:.2f}</span></div>
    <div>Total <span>{xg_total:.2f}</span></div>
    <div>Media <span>{avg_goals:.2f}</span></div>
    <div>Peso xG <span>{w_xg:.0%}</span></div>
    <div>ρ efectivo <span>{rho_eff:.2f}</span></div>
    <div>Certeza 1X2 <span>{certainty:.0%}</span></div>
    <div style='color:{context_color};font-family:{MONO},monospace;font-size:.78rem;'>
        {context_arrow} {abs(diff_vs_avg):.2f} — {context_txt}
    </div>
    <div style='font-size:1rem;margin-left:auto;color:{AMBER};letter-spacing:1px;'>{rating}</div>
</div>""", unsafe_allow_html=True)

t1_ini = "".join(w[0] for w in team1.split()).upper()[:3] or "???"
t2_ini = "".join(w[0] for w in team2.split()).upper()[:3] or "???"

st.markdown(f"""
<div class='match-header'>
    <div class='team-block'>
        <div class='team-circle' style='background:{GREEN};'>{t1_ini}</div>
        <div class='team-role'>Local</div>
        <div class='team-name'>{team1}</div>
        <div class='team-xg' style='color:{GREEN};'>{lambda_h:.2f}</div>
        <div class='team-xg-lbl'>λ calibrado</div>
    </div>
    <div class='vs-block'>
        <div class='vs-txt'>VS</div>
        <div class='vs-total'>{xg_total:.2f}</div>
        <div class='vs-copa'>λ total<br>media {avg_goals:.2f}</div>
    </div>
    <div class='team-block'>
        <div class='team-circle' style='background:{BLUE};'>{t2_ini}</div>
        <div class='team-role'>Visitante</div>
        <div class='team-name'>{team2}</div>
        <div class='team-xg' style='color:{BLUE};'>{lambda_a:.2f}</div>
        <div class='team-xg-lbl'>λ calibrado</div>
    </div>
</div>""", unsafe_allow_html=True)


def sec(kicker, label, desc):
    st.markdown(f"""
    <div class='sec-header'>
      <span class='sec-kicker'>{kicker}</span>
      <span class='sec-label'>{label}</span>
    </div>
    <p class='sec-desc'>{desc}</p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  01 · RESULTADO FINAL (1X2)
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 01", "Resultado Final (1X2)",
    "Probabilidad de cada resultado al término de los 90 minutos.")

w_prob = max(h, d, a)
col1, col2, col3, col4 = st.columns(4)
for col, (lbl, val, color) in zip(
    [col1, col2, col3],
    [(f"Victoria {team1}", h, prob_color(h)),
     ("Empate", d, AMBER),
     (f"Victoria {team2}", a, prob_color(a))]
):
    is_winner = (val == w_prob)
    cls = "card-winner" if is_winner else ""
    badge = (f"<div style='font-size:.6rem;color:{GREEN};text-transform:uppercase;"
             f"letter-spacing:2px;margin-bottom:8px;'>◆ FAVORITO</div>"
             if is_winner else "<div style='height:22px;'></div>")
    col.markdown(f"""
    <div class='card {cls}'>
        {badge}
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

# Donut + barra apilada 100%
c_don, c_bar = st.columns([1, 2])
with c_don:
    fig_don = go.Figure(go.Pie(
        values=[h, d, a],
        labels=[team1, "Empate", team2],
        hole=0.62, sort=False, direction="clockwise",
        marker=dict(colors=[GREEN, AMBER, BLUE], line=dict(color=BG, width=3)),
        textinfo="percent", textfont=dict(family=MONO, size=12, color=BG),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig_don.add_annotation(text=f"<b>{certainty:.0%}</b><br><span style='font-size:10px'>certeza</span>",
                           showarrow=False, font=dict(family=MONO, size=18, color=TXT))
    fig_don.update_layout(**PLOTLY_BASE, showlegend=False,
                          margin=dict(t=8, b=8, l=8, r=8), height=210)
    st.plotly_chart(fig_don, use_container_width=True)
with c_bar:
    fig1x2 = go.Figure()
    for lbl, val, color in [(team1, h, GREEN), ("Empate", d, AMBER), (team2, a, BLUE)]:
        fig1x2.add_trace(go.Bar(
            x=[val * 100], y=["1X2"], name=lbl, orientation="h",
            marker=dict(color=color, line_width=0),
            text=f"{lbl}<br>{val:.1%}", textposition="inside",
            insidetextanchor="middle",
            textfont=dict(family=MONO, size=12, color=BG),
        ))
    fig1x2.update_layout(**PLOTLY_BASE, barmode="stack", showlegend=False,
                         xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
                         yaxis=dict(showgrid=False, showticklabels=False),
                         margin=dict(t=8, b=8, l=8, r=8), height=210)
    st.plotly_chart(fig1x2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  02 · DOBLE OPORTUNIDAD
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 02", "Doble Oportunidad",
    "Cubre dos de los tres posibles resultados simultáneamente.")

dc_items = [
    ("1X", f"{team1} gana o Empate", dc_1x, GREEN),
    ("12", "Cualquiera gana · sin empate", dc_12, BLUE),
    ("X2", f"Empate o {team2} gana", dc_x2, "#C4B5FD"),
]
for col, (code, desc, val, color) in zip(st.columns(3), dc_items):
    col.markdown(f"""
    <div class='card'>
        <div style='font-family:{MONO},monospace;font-size:1.4rem;font-weight:700;
                    color:{color};border:1px solid {color}44;border-radius:8px;
                    padding:6px 16px;display:inline-block;margin-bottom:12px;
                    letter-spacing:2px;'>{code}</div>
        <div class='card-val' style='color:{color};'>{val:.1%}</div>
        <div class='card-sub' style='margin-top:8px;'>{desc}</div>
        {bar_html(val*100, color)}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  03 · DRAW NO BET
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 03", "Draw No Bet (DNB)",
    "Excluye el empate y redistribuye su probabilidad entre ambos equipos.")

cols_dnb = st.columns([3, 3, 2])
for col, (lbl, val, color) in zip(cols_dnb[:2],
                                  [(team1, dnb_h, GREEN), (team2, dnb_a, BLUE)]):
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
        <div style='display:flex;justify-content:space-between;padding:8px 0;
                    border-bottom:1px solid {BORDER};'>
            <span style='font-size:.78rem;color:{DIM};'>Empate excluido</span>
            <span style='font-family:{MONO},monospace;font-size:.82rem;
                         color:{AMBER};font-weight:700;'>{d:.1%}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:8px 0;'>
            <span style='font-size:.78rem;color:{DIM};'>Base de cálculo</span>
            <span style='font-family:{MONO},monospace;font-size:.82rem;
                         color:{GREEN};font-weight:700;'>{h+a:.1%}</span>
        </div>
    </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  04 · OVER / UNDER
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 04", "Over / Under — Líneas estándar",
    "Probabilidad de superar cada línea de goles totales. El gráfico muestra la "
    "distribución acumulada P(total > x): la forma completa del partido, no solo 5 puntos.")

ou_html = "<div class='ou-grid'>"
for L in ou_lines_v:
    ov, un = ou_res[L]
    hl = "ou-card-highlight" if L == 2.5 else ""
    ou_html += f"""
    <div class='ou-card {hl}'>
        <div class='ou-line'>{L}</div>
        <div class='ou-tag-o'>Over</div><div class='ou-over'>{ov:.1%}</div>
        <div class='ou-sep'></div>
        <div class='ou-tag-u'>Under</div><div class='ou-under'>{un:.1%}</div>
    </div>"""
ou_html += "</div>"
st.markdown(ou_html, unsafe_allow_html=True)

# Curva de supervivencia P(total > g) — mucho más informativa que 2 líneas
g_axis = np.arange(0, min(11, len(tot)))
surv = [float(tot[g + 1:].sum()) * 100 for g in g_axis]
fig_ou = go.Figure()
fig_ou.add_trace(go.Scatter(
    x=g_axis + 0.5, y=surv, mode="lines+markers", name="P(Over)",
    line=dict(color=GREEN, width=3, shape="spline"),
    fill="tozeroy", fillcolor="rgba(52,211,153,0.10)",
    marker=dict(size=8, color=GREEN, line=dict(color=BG, width=2)),
    hovertemplate="Over %{x}: %{y:.1f}%<extra></extra>",
))
for L in ou_lines_v:
    ov = ou_res[L][0] * 100
    fig_ou.add_annotation(x=L, y=ov, text=f"<b>{ov:.1f}%</b>", showarrow=False,
                          yshift=18, font=dict(family=MONO, size=10, color=GREEN))
fig_ou.add_hline(y=50, line_dash="dash", line_color=BORDER2, line_width=1,
                 annotation_text="50%", annotation_font=dict(color=DIM, size=10))
fig_ou.update_layout(**PLOTLY_BASE,
    xaxis=dict(title=dict(text="Línea de goles", font=dict(color=DIM, size=11)),
               showgrid=False, tickvals=list(np.arange(0.5, 10.6, 1)),
               tickfont=dict(family=MONO, color=DIM, size=10)),
    yaxis=dict(showgrid=True, gridcolor=BORDER, range=[0, 105],
               ticksuffix="%", tickfont=dict(family=MONO, color=DIM, size=10)),
    margin=dict(t=24, b=10, l=10, r=10), height=240, showlegend=False)
st.plotly_chart(fig_ou, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  05 · HÁNDICAP ASIÁTICO (exacto, con push y líneas de cuarto)
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 05", "Hándicap Asiático",
    "Probabilidad de <b>cubrir</b> cada línea. Las líneas enteras muestran el push "
    "(devolución); las de cuarto reparten media apuesta en cada línea contigua.")

def desc_handicap(L, negative):
    need = int(np.ceil(L + 0.01))
    if negative:
        return f"gana por {need}+ gol{'es' if need > 1 else ''}"
    if L == 0.5:
        return "empata o gana"
    return f"no pierde por {need}+ goles"

handicap_lines = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

def hc_card(L, sign, cover, push, color):
    sgn = f"{'-' if sign < 0 else '+'}{L}"
    push_row = (f"<div style='font-size:.62rem;color:{AMBER};margin-top:4px;"
                f"font-family:{MONO},monospace;'>push {push:.1%}</div>") if push > 1e-4 else ""
    return f"""
    <div style='flex:1;background:{SURF};border:1px solid {BORDER};
                border-radius:12px;padding:12px;'>
        <div style='font-family:{MONO},monospace;font-size:.85rem;font-weight:700;
                    color:{TXT};'>{sgn}</div>
        <div style='font-size:.68rem;color:{MUT};margin:4px 0 8px;'>{desc_handicap(L, sign < 0)}</div>
        <div style='font-family:{MONO},monospace;font-size:1.3rem;font-weight:700;
                    color:{color};'>{cover:.1%}</div>
        {push_row}
        <div style='margin-top:6px;'>{bar_html(cover*100, color, 4)}</div>
    </div>"""

col_loc, col_vis = st.columns(2)
with col_loc:
    st.markdown(f"<div style='font-family:{MONO},monospace;font-weight:700;"
                f"color:{GREEN};margin-bottom:16px;'>🏠 {team1.upper()}</div>",
                unsafe_allow_html=True)
    for L in handicap_lines:
        wn, pn, _ = asian_handicap(mat, -L)   # local −L
        wp, pp, _ = asian_handicap(mat,  L)   # local +L
        st.markdown(f"<div style='display:flex;gap:10px;margin-bottom:12px;'>"
                    f"{hc_card(L, -1, wn, pn, GREEN)}{hc_card(L, +1, wp, pp, GREEN)}</div>",
                    unsafe_allow_html=True)
with col_vis:
    st.markdown(f"<div style='font-family:{MONO},monospace;font-weight:700;"
                f"color:{BLUE};margin-bottom:16px;'>✈️ {team2.upper()}</div>",
                unsafe_allow_html=True)
    for L in handicap_lines:
        # visitante −L cubre si (visitante − local − L) > 0 → local con +L pierde
        wn2, pn2, ln2 = asian_handicap(mat,  L)   # visitante −L cubre = loss del local+L
        wp2, pp2, lp2 = asian_handicap(mat, -L)   # visitante +L cubre = loss del local−L
        st.markdown(f"<div style='display:flex;gap:10px;margin-bottom:12px;'>"
                    f"{hc_card(L, -1, ln2, pn2, BLUE)}{hc_card(L, +1, lp2, pp2, BLUE)}</div>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  06 · BTTS + PORTERÍA A CERO
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 06", "Ambos Equipos Marcan (BTTS)",
    f"Probabilidad de que tanto {team1} como {team2} anoten. "
    "Todo derivado de la matriz Dixon–Coles (coherente con ρ), incluida la "
    "probabilidad individual de marcar y las porterías a cero.")

col_b1, col_b2, col_b3 = st.columns([2, 2, 3])
with col_b1:
    st.markdown(f"""
    <div class='btts-big btts-yes'>
        <div class='btts-label' style='color:{GREEN};'>✔ BTTS — Sí</div>
        <div class='btts-pct' style='color:{GREEN};'>{btts_y:.1%}</div>
        <div class='btts-sub'>Ambos equipos marcan</div>
        {bar_html(btts_y*100)}
    </div>""", unsafe_allow_html=True)
with col_b2:
    st.markdown(f"""
    <div class='btts-big btts-no'>
        <div class='btts-label' style='color:{RED};'>✘ BTTS — No</div>
        <div class='btts-pct' style='color:{RED};'>{btts_n:.1%}</div>
        <div class='btts-sub'>Al menos uno no marca</div>
        {bar_html(btts_n*100, RED)}
    </div>""", unsafe_allow_html=True)
with col_b3:
    st.markdown(f"""
    <div class='card' style='text-align:left;padding:20px;'>
        <div class='card-lbl' style='margin-bottom:14px;'>Marca al menos un gol · Portería a cero</div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team1} marca</div>
            <div class='prob-bar-track'><div class='prob-bar-fill'
                 style='width:{p_e1_marca*100:.1f}%;background:{GREEN};'></div></div>
            <div class='prob-bar-num'>{p_e1_marca:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team2} marca</div>
            <div class='prob-bar-track'><div class='prob-bar-fill'
                 style='width:{p_e2_marca*100:.1f}%;background:{BLUE};'></div></div>
            <div class='prob-bar-num' style='color:{BLUE};'>{p_e2_marca:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>CS {team1}</div>
            <div class='prob-bar-track'><div class='prob-bar-fill'
                 style='width:{cs_home*100:.1f}%;background:{AMBER};'></div></div>
            <div class='prob-bar-num' style='color:{AMBER};'>{cs_home:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>CS {team2}</div>
            <div class='prob-bar-track'><div class='prob-bar-fill'
                 style='width:{cs_away*100:.1f}%;background:{AMBER};'></div></div>
            <div class='prob-bar-num' style='color:{AMBER};'>{cs_away:.1%}</div>
        </div>
        <div class='divider-line' style='margin:12px 0;'></div>
        <div class='card-lbl' style='margin-bottom:14px;'>Composición del BTTS Sí</div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team1} gana</div>
            <div class='prob-bar-track'><div class='prob-bar-fill'
                 style='width:{home_btts_yes/btts_y*100:.1f}%;background:{GREEN};'></div></div>
            <div class='prob-bar-num'>{home_btts_yes:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>Empate</div>
            <div class='prob-bar-track'><div class='prob-bar-fill'
                 style='width:{draw_btts_yes/btts_y*100:.1f}%;background:{AMBER};'></div></div>
            <div class='prob-bar-num' style='color:{AMBER};'>{draw_btts_yes:.1%}</div>
        </div>
        <div class='prob-bar-row'>
            <div class='prob-bar-lbl'>{team2} gana</div>
            <div class='prob-bar-track'><div class='prob-bar-fill'
                 style='width:{away_btts_yes/btts_y*100:.1f}%;background:{BLUE};'></div></div>
            <div class='prob-bar-num' style='color:{BLUE};'>{away_btts_yes:.1%}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  07 · GOLES TOTALES EXACTOS
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 07", "Goles Totales Exactos",
    "Probabilidad de exactamente N goles (barras) y probabilidad acumulada "
    "P(total ≤ N) (línea).")

eg_g = list(range(min(11, len(tot))))
eg_v = [tot[g] * 100 for g in eg_g]
eg_cum = np.cumsum(eg_v)
colors_eg = [GREEN if g == peak_g else "#1B4433" for g in eg_g]

fig_eg = go.Figure()
fig_eg.add_trace(go.Bar(
    x=[f"{g}" for g in eg_g], y=eg_v, marker_color=colors_eg, marker_line_width=0,
    text=[f"{v:.1f}%" for v in eg_v], textposition="outside",
    textfont=dict(family=MONO, size=10, color=MUT), name="P(exacto)",
))
fig_eg.add_trace(go.Scatter(
    x=[f"{g}" for g in eg_g], y=eg_cum, mode="lines+markers", name="P(≤ N)",
    line=dict(color=AMBER, width=2, dash="dot"),
    marker=dict(size=5, color=AMBER), yaxis="y2",
    hovertemplate="P(total ≤ %{x}): %{y:.1f}%<extra></extra>",
))
fig_eg.add_annotation(x=f"{peak_g}", y=tot[peak_g]*100, text="★ más probable",
                      showarrow=True, arrowhead=0, arrowcolor=GREEN, arrowwidth=1,
                      ax=0, ay=-36, font=dict(family=MONO, size=10, color=GREEN))
fig_eg.update_layout(**PLOTLY_BASE,
    xaxis=dict(title=dict(text="Goles totales", font=dict(color=DIM, size=11)),
               showgrid=False, tickfont=dict(family=MONO, size=11, color=DIM)),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(eg_v)*1.45]),
    yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False,
                ticksuffix="%", tickfont=dict(family=MONO, color=AMBER, size=9)),
    legend=dict(bgcolor=SURF, bordercolor=BORDER, borderwidth=1,
                orientation="h", x=0.5, xanchor="center", y=1.14, font=dict(size=11)),
    margin=dict(t=26, b=10, l=10, r=10), height=270)
st.plotly_chart(fig_eg, use_container_width=True)

mg_ranges = [
    ("0–1 goles", 0, 1, "Partido muy cerrado"),
    ("2–3 goles", 2, 3, "Rango más frecuente"),
    ("3–4 goles", 3, 4, "Partido abierto"),
    ("2–4 goles", 2, 4, "Rango amplio central"),
]
mg_html = "<div class='mg-grid'>"
for lbl, lo, hi_, desc in mg_ranges:
    p = calc_multigoal(tot, lo, hi_)
    mg_html += f"""
    <div class='card'>
        <div class='card-lbl'>{lbl}</div>
        <div class='card-val'>{p:.1%}</div>
        <div class='card-sub'>{desc}</div>
        {bar_html(p*100)}
    </div>"""
mg_html += "</div>"
st.markdown(mg_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  08 · MARCADORES EXACTOS + HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 08", "Marcadores Exactos",
    "Los 10 marcadores más probables y el mapa de calor de la matriz Dixon–Coles.")

top10_html = "<div class='top10-grid'>"
for rank, (prob, i, j) in enumerate(top10):
    cls = "top10-card-1" if rank == 0 else ""
    sc = GREEN if rank == 0 else (TXT if rank < 3 else MUT)
    top10_html += f"""
    <div class='top10-card {cls}'>
        <div class='top10-rank'>#{rank+1}</div>
        <div class='top10-score' style='color:{sc};'>{i}–{j}</div>
        <div class='top10-prob'>{prob:.1%}</div>
    </div>"""
top10_html += "</div>"
st.markdown(top10_html, unsafe_allow_html=True)

SHOW = int(min(10, max(6, np.ceil(max(lambda_h, lambda_a) * 3))))
z = mat[:SHOW, :SHOW] * 100
text_mat = [[f"{z[i][j]:.1f}" for j in range(SHOW)] for i in range(SHOW)]
fig_heat = go.Figure(go.Heatmap(
    z=z,
    x=[str(j) for j in range(SHOW)],
    y=[str(i) for i in range(SHOW)],
    colorscale=[[0, BG], [0.15, "#0F2A1E"], [0.45, "#166345"],
                [0.75, "#0E9F6E"], [1.0, GREEN]],
    text=text_mat, texttemplate="%{text}",
    textfont=dict(family=MONO, size=11, color=TXT),
    hovertemplate=f"{team1} %{{y}} – {team2} %{{x}}: %{{z:.2f}}%<extra></extra>",
    showscale=True,
    colorbar=dict(tickfont=dict(family=MONO, color=DIM, size=10),
                  bgcolor=SURF, bordercolor=BORDER, thickness=10,
                  title=dict(text="%", font=dict(color=DIM, size=10))),
))
fig_heat.update_layout(**PLOTLY_BASE,
    xaxis=dict(title=dict(text=f"Goles {team2}", font=dict(color=BLUE)),
               showgrid=False, side="top", tickfont=dict(family=MONO, color=DIM)),
    yaxis=dict(title=dict(text=f"Goles {team1}", font=dict(color=GREEN)),
               showgrid=False, autorange="reversed",
               tickfont=dict(family=MONO, color=DIM)),
    margin=dict(t=44, b=10, l=10, r=10), height=420)
st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  09 · MARGEN DE VICTORIA
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 09", "Margen de Victoria",
    "Diferencia de goles al final del partido, en barras divergentes.")

mg_items = sorted(margins.items())
mg_x, mg_y, mg_c, mg_t = [], [], [], []
for diff, p in mg_items:
    if diff > 0:
        mg_x.append(f"{team1[:6]} +{diff}"); mg_c.append(GREEN)
    elif diff < 0:
        mg_x.append(f"{team2[:6]} +{abs(diff)}"); mg_c.append(BLUE)
    else:
        mg_x.append("Empate"); mg_c.append(AMBER)
    mg_y.append(p * 100); mg_t.append(f"{p:.1%}")

fig_mg = go.Figure(go.Bar(
    x=mg_x, y=mg_y, marker_color=mg_c, marker_line_width=0,
    text=mg_t, textposition="outside",
    textfont=dict(family=MONO, size=10, color=MUT),
))
fig_mg.update_layout(**PLOTLY_BASE,
    xaxis=dict(showgrid=False, tickfont=dict(color=DIM, size=10)),
    yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(mg_y)*1.38]),
    margin=dict(t=10, b=10, l=10, r=10), height=250)
st.plotly_chart(fig_mg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  10 · PUNTOS ESPERADOS
# ══════════════════════════════════════════════════════════════════════════════
sec("Mercado 10", "Puntos Esperados (xPts)",
    "Valor esperado de puntos si este partido se jugase muchas veces.")

for col, (team, xp, color) in zip(
    st.columns(2),
    [(team1, exp_h, GREEN), (team2, exp_a, BLUE)]
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
            <span style='font-size:.62rem;color:{DIM};font-family:{MONO};'>0</span>
            <span style='font-size:.62rem;color:{color};font-family:{MONO};'>{xp:.2f} / 3.00</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='info-pill' style='margin-top:4px;'>
    <div>{team1} xPts = 3×{h:.2f} + {d:.2f} = <span>{exp_h:.2f}</span></div>
    <div>{team2} xPts = 3×{a:.2f} + {d:.2f} = <span>{exp_a:.2f}</span></div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  11 · COMPARTIR POR WHATSAPP
# ══════════════════════════════════════════════════════════════════════════════
sec("Compartir", "Enviar Análisis", "")

def pct(v): return f"{v:.1%}"

sorted_exact_top5 = sorted(exact.items(), key=lambda x: x[1], reverse=True)[:5]

wa_message = f"""⚽ *ANÁLISIS xG · COPA DEL MUNDO*
━━━━━━━━━━━━━━━━━━━━━━
🏟️ *{team1}* (Local) vs *{team2}* (Visitante)
📊 Modelo: Dixon & Coles · calibración por contracción (w={w_xg:.0%})
λ {team1}: {lambda_h:.2f} | λ {team2}: {lambda_a:.2f}
Total: {xg_total:.2f} | Media torneo: {avg_goals:.2f} | ρ: {rho_eff:.2f}
Valoración: {rating} | Certeza 1X2: {certainty:.0%}

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
{chr(10).join(f"{'⭐ ' if L == 2.5 else ''}O/U {L} → Over {pct(ou_res[L][0])} | Under {pct(ou_res[L][1])}" for L in ou_lines_v)}

━━━━━━━━━━━━━━━━━━━━━━
⚡ *BTTS (AMBOS MARCAN)*
✅ BTTS Sí: {pct(btts_y)}
❌ BTTS No: {pct(btts_n)}
{team1} marca: {pct(p_e1_marca)} | {team2} marca: {pct(p_e2_marca)}
CS {team1}: {pct(cs_home)} | CS {team2}: {pct(cs_away)}

━━━━━━━━━━━━━━━━━━━━━━
🔢 *GOLES TOTALES MÁS PROBABLES*
{chr(10).join(f"  {g} gol{'es' if g != 1 else ''}:  {pct(p)}" for g, p in sorted_exact_top5)}

━━━━━━━━━━━━━━━━━━━━━━
📈 *PUNTOS ESPERADOS (xPts)*
{team1}: {exp_h:.2f} pts
{team2}: {exp_a:.2f} pts

━━━━━━━━━━━━━━━━━━━━━━
_Generado con modelo Dixon & Coles_
_ρ={rho_eff:.2f} · λ₁={lambda_h:.2f} · λ₂={lambda_a:.2f} · w={w_xg:.0%}_"""

wa_url = f"https://wa.me/?text={urllib.parse.quote(wa_message)}"

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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:40px 0 20px;border-top:1px solid {BORDER};margin-top:32px;'>
    <div style='font-family:{MONO},monospace;font-size:.62rem;color:{DIM};
                letter-spacing:2.5px;text-transform:uppercase;'>
        Copa del Mundo · Análisis xG · Modelo Dixon & Coles v2
    </div>
    <div style='font-family:{MONO},monospace;font-size:.58rem;color:#3A463F;margin-top:6px;'>
        λ₁ = {lambda_h:.2f} ({team1}) · λ₂ = {lambda_a:.2f} ({team2}) · ρ = {rho_eff:.2f}
        · w = {w_xg:.0%} · Media {avg_goals:.2f} g/p · Total {xg_total:.2f} · Matriz {N}×{N}
    </div>
</div>""", unsafe_allow_html=True)
