import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import contextmanager

import streamlit as st
st.set_option('client.showErrorDetails', True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

from chatbot_engine import get_response
from predict_aqi    import (predict_aqi, get_aqi_category, get_aqi_risk,
                             get_default_features, FEATURE_NAMES)
from api            import get_real_aqi
from forecast       import forecast_next_7_days

# Optional: Plotly for the real 3D Earth on the Real-Time AQI page.
# If unavailable in the user's env, the page falls back gracefully.
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AQI Smart Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "aqi_india_work.csv")

CITY_COLS   = ['ahmedabad','bengaluru','chandigarh','chennai','delhi','guwahati',
               'hyderabad','jaipur','kolkata','lucknow','mumbai','raipur',
               'ranchi','shimla','visakhapatnam']
SEASON_MAP  = {0:"Post-Monsoon", 1:"Summer", 2:"Monsoon", 3:"Winter"}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ─────────────────────────────────────────
# SVG ICON LIBRARY  (clean outline style, Lucide-inspired)
# ─────────────────────────────────────────
@contextmanager
def card(title=None, subtitle=None, icon_name=None):
    """Unified card container — heading + body live in ONE bordered box.

    Built on st.container(border=True), then re-skinned via global CSS that
    targets [data-testid="stVerticalBlockBorderWrapper"]. This guarantees the
    chart/columns/widgets all sit visually inside the same card — fixing the
    problem where opening/closing <div class="card"> via st.markdown failed
    to wrap Streamlit-rendered elements like st.plotly_chart."""
    with st.container(border=True):
        if title:
            ico_html = (f'<span class="ico">{icon(icon_name, 18, PRIMARY_HOV, 2)}</span>'
                        if icon_name else "")
            st.markdown(
                f'<div class="uni-card-head">{ico_html}{title}</div>',
                unsafe_allow_html=True,
            )
        if subtitle:
            st.markdown(
                f'<div class="uni-card-sub">{subtitle}</div>',
                unsafe_allow_html=True,
            )
        yield


def icon(name, size=18, stroke="currentColor", stroke_width=1.8):
    paths = {
        "home":      '<path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7h-6v7H4a1 1 0 0 1-1-1z"/>',
        "bot":       '<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M9 14h.01M15 14h.01M12 4v4M8 21v-1M16 21v-1"/>',
        "chart":     '<path d="M3 3v18h18"/><path d="M7 14l4-4 3 3 5-6"/>',
        "trending":  '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
        "lab":       '<path d="M9 3v6L4 19a2 2 0 0 0 1.7 3h12.6A2 2 0 0 0 20 19l-5-10V3"/><path d="M8 3h8"/>',
        "trophy":    '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M17 4h3v3a3 3 0 0 1-3 3M7 4H4v3a3 3 0 0 0 3 3"/>',
        "globe":     '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
        "sparkles":  '<path d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
        "search":    '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
        "play":      '<path d="M5 3l14 9-14 9z"/>',
        "refresh":   '<path d="M3 12a9 9 0 0 1 15.5-6.3L21 8M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.3L3 16M3 21v-5h5"/>',
        "trash":     '<path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14"/>',
        "send":      '<path d="M22 2L11 13M22 2l-7 20-4-9-9-4z"/>',
        "moon":      '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
        "sun":       '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
        "wind":      '<path d="M3 8h12a3 3 0 1 0-3-3M3 12h18M3 16h10a3 3 0 1 1-3 3"/>',
        "shield":    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        "alert":     '<path d="M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
        "check":     '<path d="M20 6L9 17l-5-5"/>',
        "info":      '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/>',
        "pin":       '<path d="M12 22s7-7.5 7-13a7 7 0 1 0-14 0c0 5.5 7 13 7 13z"/><circle cx="12" cy="9" r="2.5"/>',
        "menu":      '<path d="M3 6h18M3 12h18M3 18h18"/>',
        "activity":  '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "layers":    '<path d="M12 2l10 6-10 6L2 8l10-6z"/><path d="M2 16l10 6 10-6M2 12l10 6 10-6"/>',
        "scale":     '<path d="M12 3v18M5 21h14M6 8l-4 7a4 4 0 0 0 8 0L6 8zm12 0l-4 7a4 4 0 0 0 8 0l-4-7z"/>',
        "pie":       '<path d="M21 15.5A9 9 0 1 1 8.5 3v9H21z"/>',
        "list":      '<path d="M3 6h18M3 12h18M3 18h18"/>',
        "lightning": '<path d="M13 2L3 14h7l-1 8 10-12h-7z"/>',
        "calendar":  '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 9h18M8 3v4M16 3v4"/>',
        "settings":  '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
        "user":      '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
        "spark":     '<path d="M12 2v6M12 16v6M2 12h6M16 12h6M5 5l4 4M15 15l4 4M5 19l4-4M15 9l4-4"/>',
        "database":  '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v6c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 11v6c0 1.7 4 3 9 3s9-1.3 9-3v-6"/>',
        "map-pin":   '<path d="M12 22s7-7.5 7-13a7 7 0 1 0-14 0c0 5.5 7 13 7 13z"/><circle cx="12" cy="9" r="2.5"/>',
        "bar":       '<path d="M3 3v18h18"/><rect x="7" y="13" width="3" height="6" rx="0.5"/><rect x="12" y="9" width="3" height="10" rx="0.5"/><rect x="17" y="5" width="3" height="14" rx="0.5"/>',
        "sliders":   '<path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h14M18 18h2"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="16" cy="18" r="2"/>',
        "chat-bubble": '<path d="M21 12a8 8 0 0 1-11.5 7.2L3 21l1.8-6.5A8 8 0 1 1 21 12z"/><circle cx="9" cy="12" r="0.8" fill="currentColor"/><circle cx="12" cy="12" r="0.8" fill="currentColor"/><circle cx="15" cy="12" r="0.8" fill="currentColor"/>',
    }
    p = paths.get(name, '<circle cx="12" cy="12" r="3"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="display:inline-block;vertical-align:middle;flex-shrink:0;">{p}</svg>')


# ─────────────────────────────────────────
# SESSION STATE  (initialised exactly once)
# ─────────────────────────────────────────
SESSION_DEFAULTS = {
    "messages":          [],
    "ctx_city":          None,
    "ctx_aqi":           None,
    "dark":              True,                    # premium dark by default
    "section":           "Home",
    "features":          None,
    "predicted_aqi":     None,
    "fc_city":           "",
    "fc_live_aqi":       None,
    "fc_live_name":      None,
    "fc_result":         None,
    "fc_anchor_source":  None,
}
for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

DARK = st.session_state.dark


# ─────────────────────────────────────────
# THEME TOKENS — premium dark blue SaaS system
# ─────────────────────────────────────────
if DARK:
    BG          = "#0B1F3A"
    BG_GRAD     = "radial-gradient(1200px 600px at 10% -10%, rgba(59,130,246,0.18), transparent 60%), radial-gradient(900px 500px at 95% 0%, rgba(29,78,216,0.22), transparent 55%), linear-gradient(180deg, #0B1F3A 0%, #08172B 100%)"
    SURFACE     = "rgba(20, 43, 77, 0.72)"
    SURFACE_SOLID = "#142B4D"
    SURFACE_2   = "rgba(11, 31, 58, 0.6)"
    TEXT        = "#E6EEFB"
    MUTED       = "#8AA3C7"
    HEADING     = "#F4F8FF"
    BORDER      = "rgba(148, 178, 230, 0.14)"
    BORDER_HOVER = "rgba(148, 178, 230, 0.28)"
    PRIMARY     = "#2563EB"
    PRIMARY_HOV = "#3B82F6"
    ACCENT_FROM = "#1D4ED8"
    ACCENT_TO   = "#3B82F6"
    GLOW        = "0 10px 30px -10px rgba(37,99,235,0.45), 0 0 0 1px rgba(148,178,230,0.06)"
    SHADOW      = "0 1px 0 rgba(255,255,255,0.04) inset, 0 10px 30px -12px rgba(0,0,0,0.55)"
    PLT_BG, PLT_TEXT, PLT_GRID = "#0F2747", "#C9D8F2", (148/255, 178/255, 230/255, 0.18)
    # Theme-aware row + bar tokens (used by lists, forecast rows, progress tracks)
    ROW_BG      = "rgba(20, 43, 77, 0.55)"
    ROW_BORDER  = "rgba(148, 178, 230, 0.12)"
    BAR_TRACK   = "rgba(148, 178, 230, 0.14)"
    CHIP_BG     = "rgba(59, 130, 246, 0.10)"
    # Chat input (blue-themed, high contrast)
    CHAT_INPUT_BG     = "rgba(20, 43, 77, 0.92)"
    CHAT_INPUT_BORDER = "rgba(59, 130, 246, 0.55)"
    CHAT_INPUT_TEXT   = "#F4F8FF"
    CHAT_INPUT_PH     = "rgba(202, 218, 244, 0.65)"
else:
    BG          = "#F4F7FB"
    BG_GRAD     = "radial-gradient(1200px 600px at 10% -10%, rgba(59,130,246,0.10), transparent 60%), linear-gradient(180deg, #F4F7FB 0%, #EAF1FB 100%)"
    SURFACE     = "rgba(255,255,255,0.92)"
    SURFACE_SOLID = "#FFFFFF"
    SURFACE_2   = "#F2F6FC"
    TEXT        = "#0F1F3A"
    MUTED       = "#475569"
    HEADING     = "#0B1F3A"
    BORDER      = "rgba(15, 31, 58, 0.10)"
    BORDER_HOVER = "rgba(15, 31, 58, 0.22)"
    PRIMARY     = "#1D4ED8"
    PRIMARY_HOV = "#2563EB"
    ACCENT_FROM = "#1D4ED8"
    ACCENT_TO   = "#3B82F6"
    GLOW        = "0 10px 30px -12px rgba(29,78,216,0.30), 0 0 0 1px rgba(15,31,58,0.04)"
    SHADOW      = "0 1px 2px rgba(15,31,58,0.04), 0 8px 24px -10px rgba(15,31,58,0.10)"
    PLT_BG, PLT_TEXT, PLT_GRID = "#FFFFFF", "#0F1F3A", (15/255, 31/255, 58/255, 0.10)
    # Theme-aware row + bar tokens
    ROW_BG      = "#F8FAFD"
    ROW_BORDER  = "rgba(15, 31, 58, 0.10)"
    BAR_TRACK   = "rgba(15, 31, 58, 0.10)"
    CHIP_BG     = "rgba(29, 78, 216, 0.08)"
    # Chat input — keep the blue identity even in light mode
    CHAT_INPUT_BG     = "#EAF1FB"
    CHAT_INPUT_BORDER = "rgba(29, 78, 216, 0.45)"
    CHAT_INPUT_TEXT   = "#0B1F3A"
    CHAT_INPUT_PH     = "rgba(15, 31, 58, 0.55)"

# Sidebar — always-dark premium rail
SB_BG       = "linear-gradient(180deg, #08172B 0%, #0B1F3A 100%)"
SB_SURFACE  = "rgba(255,255,255,0.04)"
SB_TEXT     = "#E6EEFB"
SB_MUTED    = "#8AA3C7"
SB_BORDER   = "rgba(255,255,255,0.06)"
SB_ACTIVE_GRAD = "linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%)"

# Functional AQI palette — only used on data, per brief (Green / Yellow / Red focus)
AQI_PALETTE = {
    "Good":      {"bg":"rgba(34,197,94,0.14)","text":"#86efac","border":"rgba(34,197,94,0.45)","hex":"#22c55e"},
    "Moderate":  {"bg":"rgba(234,179,8,0.14)", "text":"#fde68a","border":"rgba(234,179,8,0.45)", "hex":"#eab308"},
    "Unhealthy": {"bg":"rgba(239,68,68,0.14)", "text":"#fca5a5","border":"rgba(239,68,68,0.45)", "hex":"#ef4444"},
    "Poor":      {"bg":"rgba(239,68,68,0.18)", "text":"#fecaca","border":"rgba(239,68,68,0.55)", "hex":"#dc2626"},
    "Hazardous": {"bg":"rgba(168,85,247,0.16)","text":"#e9d5ff","border":"rgba(168,85,247,0.50)","hex":"#a855f7"},
}
if not DARK:
    AQI_PALETTE = {
        "Good":      {"bg":"#dcfce7","text":"#14532d","border":"#22c55e","hex":"#22c55e"},
        "Moderate":  {"bg":"#fef9c3","text":"#713f12","border":"#eab308","hex":"#eab308"},
        "Unhealthy": {"bg":"#fee2e2","text":"#7f1d1d","border":"#ef4444","hex":"#ef4444"},
        "Poor":      {"bg":"#fecaca","text":"#7f1d1d","border":"#dc2626","hex":"#dc2626"},
        "Hazardous": {"bg":"#f3e8ff","text":"#4c1d95","border":"#a855f7","hex":"#a855f7"},
    }


# ─────────────────────────────────────────
# Sidebar nav icon CSS — injects an SVG (mask-image) before each radio label
# so each nav item shows a clean outline icon that follows the text color.
# ─────────────────────────────────────────
import urllib.parse as _ulp

def _svg_uri_for_mask(name):
    paths = {
        "home":     '<path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7h-6v7H4a1 1 0 0 1-1-1z"/>',
        "bot":      '<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M9 14h.01M15 14h.01M12 4v4M8 21v-1M16 21v-1"/>',
        "chat-bubble":'<path d="M21 12a8 8 0 0 1-11.5 7.2L3 21l1.8-6.5A8 8 0 1 1 21 12z"/>',
        "sliders":  '<path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h14M18 18h2"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="16" cy="18" r="2"/>',
        "trending": '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
        "bar":      '<path d="M3 3v18h18"/><rect x="7" y="13" width="3" height="6" rx="0.5"/><rect x="12" y="9" width="3" height="10" rx="0.5"/><rect x="17" y="5" width="3" height="14" rx="0.5"/>',
        "trophy":   '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0z"/><path d="M17 4h3v3a3 3 0 0 1-3 3M7 4H4v3a3 3 0 0 0 3 3"/>',
        "globe":    '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    }
    p = paths.get(name, paths["home"])
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
           'fill="none" stroke="black" stroke-width="2" '
           'stroke-linecap="round" stroke-linejoin="round">' + p + '</svg>')
    return _ulp.quote(svg, safe='')

# Order matches NAV_ITEMS — kept in sync below
_NAV_ICONS_FOR_CSS = ["home", "chat-bubble", "sliders", "trending", "bar", "trophy", "globe"]

NAV_ICON_CSS = ""
for _i, _ic in enumerate(_NAV_ICONS_FOR_CSS, start=1):
    _uri = _svg_uri_for_mask(_ic)
    NAV_ICON_CSS += (
        '[data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type(' + str(_i) +
        ') > div:nth-of-type(2)::before {'
        'content:""; display:inline-block; width:18px; height:18px;'
        'vertical-align:-4px; margin-right:12px;'
        'background-color:currentColor;'
        '-webkit-mask:url("data:image/svg+xml;utf8,' + _uri + '") center/contain no-repeat;'
        'mask:url("data:image/svg+xml;utf8,' + _uri + '") center/contain no-repeat;'
        'flex-shrink:0;}'
    )
NAV_ICON_CSS += (
    '[data-testid="stSidebar"] [role="radiogroup"] > label > div:nth-of-type(2) {'
    'display:flex; align-items:center;}'
)


# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ───────────────────────────────────────────────────────────── */
*, body, .stApp, [class*="css"] {{
    font-family:'Inter','Segoe UI',system-ui,sans-serif !important;
    -webkit-font-smoothing:antialiased;
}}
.stApp {{
    background:{BG} !important;
    background-image:{BG_GRAD} !important;
    background-attachment:fixed !important;
    color:{TEXT};
}}
.block-container {{
    padding-top:1.1rem !important;
    padding-bottom:1.2rem !important;
    max-width:1440px;
}}
/* ── Tighter vertical rhythm — kill Streamlit's default oversized gaps ── */
[data-testid="stVerticalBlock"] {{ gap:0.7rem !important; }}
[data-testid="stHorizontalBlock"] {{
    gap:1rem !important;
    align-items:stretch !important;            /* equalize column heights */
}}
[data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    display:flex !important;
    flex-direction:column !important;
}}
/* Each bordered card stretches to fill its column → top & bottom borders line up */
[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"] {{
    flex:1 1 auto !important;
    min-height:0 !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] {{
    height:auto;
}}
/* Drop the empty-element bottom margin Streamlit injects */
[data-testid="element-container"]:empty {{ display:none !important; margin:0 !important; }}
[data-testid="stVerticalBlock"] > div:empty {{ display:none !important; }}
h1,h2,h3,h4,h5 {{ color:{HEADING} !important; letter-spacing:-0.02em; font-weight:700; }}
hr {{ border-color:{BORDER} !important; }}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility:hidden; height:0; }}

/* ── Page enter animation ──────────────────────────────────────────── */
@keyframes fadeUp  {{ from {{opacity:0; transform:translateY(8px);}} to {{opacity:1; transform:translateY(0);}} }}
@keyframes fadeIn  {{ from {{opacity:0;}} to {{opacity:1;}} }}
@keyframes pulseDot{{ 0%{{box-shadow:0 0 0 0 rgba(34,197,94,0.6);}} 70%{{box-shadow:0 0 0 8px rgba(34,197,94,0);}} 100%{{box-shadow:0 0 0 0 rgba(34,197,94,0);}} }}
@keyframes spin     {{ from {{transform:rotate(0deg);}} to {{transform:rotate(360deg);}} }}
@keyframes shimmer  {{ 0%{{background-position:-200% 0;}} 100%{{background-position:200% 0;}} }}
@keyframes typing   {{ 0%,60%,100%{{transform:translateY(0); opacity:0.4;}} 30%{{transform:translateY(-4px); opacity:1;}} }}

.block-container > div {{ animation:fadeUp 0.45s ease-out both; }}

/* ── Sidebar (always-dark, premium) ────────────────────────────────── */
[data-testid="stSidebar"] {{
    background:{SB_BG} !important;
    border-right:1px solid {SB_BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top:0; }}
[data-testid="stSidebar"] * {{ color:{SB_TEXT}; }}
[data-testid="stSidebar"] hr {{
    border-color:{SB_BORDER} !important; margin:14px 0 !important; opacity:0.9;
}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
    font-size:0.66rem !important; font-weight:700 !important;
    text-transform:uppercase; letter-spacing:0.14em;
    color:{SB_MUTED} !important; margin-bottom:8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="checkbox"] {{ margin-top:4px; }}
[data-testid="stSidebar"] [data-baseweb="checkbox"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] * {{
    font-size:0.85rem !important; font-weight:500 !important;
    text-transform:none !important; letter-spacing:0 !important;
    color:{SB_TEXT} !important; margin-bottom:0 !important;
}}

/* Nav radio → premium pill buttons with active glow */
[data-testid="stSidebar"] [role="radiogroup"] {{
    display:flex; flex-direction:column; gap:4px;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
    padding:11px 14px !important;
    border-radius:12px !important;
    cursor:pointer;
    transition:all 0.2s cubic-bezier(0.4,0,0.2,1);
    font-size:0.92rem !important; font-weight:500 !important;
    background:transparent !important;
    color:{SB_TEXT} !important;
    border:1px solid transparent !important;
    position:relative; overflow:hidden;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
    background:rgba(255,255,255,0.04) !important;
    border-color:rgba(255,255,255,0.06) !important;
    transform:translateX(2px);
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
    background:{SB_ACTIVE_GRAD} !important;
    color:#fff !important;
    font-weight:600 !important;
    box-shadow:0 8px 24px -10px rgba(59,130,246,0.7), 0 0 0 1px rgba(255,255,255,0.06) inset;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{ color:#fff !important; }}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{ display:none !important; }}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:nth-child(2) p {{
    font-size:0.92rem !important; margin:0 !important;
}}

/* Nav icons (mask-image, follows currentColor) */
{NAV_ICON_CSS}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {{
    background:{SB_SURFACE} !important; color:{SB_TEXT} !important;
    border:1px solid {SB_BORDER} !important; font-weight:500 !important;
    border-radius:10px !important;
    transition:all 0.2s ease;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background:rgba(255,255,255,0.08) !important;
    border-color:rgba(255,255,255,0.18) !important;
    transform:translateY(-1px);
}}

/* ── Topbar — premium glass card ───────────────────────────────────── */
.topbar {{
    background:{SURFACE};
    backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border:1px solid {BORDER};
    border-radius:18px;
    padding:22px 28px;
    margin-bottom:22px;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:{SHADOW};
    position:relative; overflow:hidden;
}}
.topbar::before {{
    content:""; position:absolute; inset:0; pointer-events:none;
    background:linear-gradient(120deg, rgba(59,130,246,0.0), rgba(59,130,246,0.08), rgba(29,78,216,0.0));
    background-size:200% 100%; animation:shimmer 8s linear infinite; opacity:0.5;
}}
.topbar-left {{ display:flex; align-items:center; gap:16px; position:relative; z-index:1; }}
.topbar-icon {{
    width:46px; height:46px; border-radius:13px;
    background:linear-gradient(135deg, {ACCENT_FROM}, {ACCENT_TO});
    display:flex; align-items:center; justify-content:center;
    color:#fff;
    box-shadow:0 8px 22px -8px rgba(59,130,246,0.65);
}}
.topbar h1 {{
    font-size:1.45rem !important; font-weight:800 !important;
    color:{HEADING} !important; margin:0 !important; letter-spacing:-0.02em;
}}
.topbar p  {{ font-size:0.86rem; color:{MUTED}; margin:3px 0 0 0; }}
.status-pill {{
    background:rgba(34,197,94,0.12);
    border:1px solid rgba(34,197,94,0.35);
    border-radius:999px; padding:7px 14px;
    display:flex; align-items:center; gap:8px;
    position:relative; z-index:1;
}}
.status-pill .dot {{
    width:7px; height:7px; border-radius:50%; background:#22c55e;
    animation:pulseDot 1.8s ease-out infinite;
}}
.status-pill .label {{
    font-size:0.7rem; font-weight:700; color:#4ade80;
    letter-spacing:0.1em; text-transform:uppercase;
}}

/* ── Cards — glass with subtle inner highlight ─────────────────────── */
.card {{
    background:{SURFACE};
    backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border:1px solid {BORDER};
    border-radius:16px;
    padding:22px 24px;
    box-shadow:{SHADOW};
    margin-bottom:16px;
    transition:transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}}
.card:hover {{
    border-color:{BORDER_HOVER};
    transform:translateY(-2px);
    box-shadow:{GLOW};
}}
.card-title {{
    font-size:0.95rem; font-weight:700; color:{HEADING};
    margin-bottom:16px; padding-bottom:12px;
    border-bottom:1px solid {BORDER}; letter-spacing:-0.01em;
    display:flex; align-items:center; gap:10px;
}}
.card-title .ico {{ color:{PRIMARY_HOV}; }}

/* ── KPI tiles — premium gradient header ───────────────────────────── */
.kpi {{
    background:{SURFACE};
    backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border:1px solid {BORDER};
    border-radius:16px;
    padding:18px 18px;
    box-shadow:{SHADOW};
    transition:transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position:relative; overflow:hidden;
}}
.kpi::before {{
    content:""; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg, {ACCENT_FROM}, {ACCENT_TO});
    opacity:0.85;
}}
.kpi:hover {{ transform:translateY(-3px); box-shadow:{GLOW}; border-color:{BORDER_HOVER}; }}
.kpi-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; color:{MUTED}; }}
.kpi-icon {{
    width:34px; height:34px; border-radius:10px;
    background:linear-gradient(135deg, rgba(29,78,216,0.18), rgba(59,130,246,0.18));
    border:1px solid rgba(59,130,246,0.25);
    display:flex; align-items:center; justify-content:center;
    color:{PRIMARY_HOV};
}}
.kpi-val   {{
    font-size:1.95rem; font-weight:800; color:{TEXT};
    line-height:1.05; letter-spacing:-0.025em;
}}
.kpi-label {{
    font-size:0.7rem; color:{MUTED}; margin-top:6px;
    font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
}}

/* ── AQI big result ───────────────────────────────────────────────── */
.aqi-result {{
    border-radius:18px; padding:30px 24px; text-align:center;
    border:1px solid transparent;
    box-shadow:{SHADOW};
    position:relative; overflow:hidden;
}}
.aqi-result::after {{
    content:""; position:absolute; inset:0; pointer-events:none;
    background:radial-gradient(400px 200px at 50% 0%, rgba(255,255,255,0.06), transparent 60%);
}}
.aqi-big  {{
    font-size:4rem; font-weight:900; line-height:1; letter-spacing:-0.04em;
    text-shadow:0 0 40px rgba(255,255,255,0.05);
}}
.aqi-cat  {{ font-size:1.15rem; font-weight:700; margin-top:8px; }}
.aqi-risk {{
    font-size:0.7rem; font-weight:700; margin-top:10px;
    letter-spacing:0.16em; text-transform:uppercase;
}}

/* ── Chat — modern ChatGPT-style ──────────────────────────────────── */
.chat-shell {{
    background:{SURFACE};
    backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border:1px solid {BORDER};
    border-radius:18px;
    box-shadow:{SHADOW};
    overflow:hidden;
    margin-bottom:16px;
}}
.chat-header {{
    padding:14px 20px; border-bottom:1px solid {BORDER};
    display:flex; align-items:center; gap:12px;
    background:linear-gradient(180deg, rgba(59,130,246,0.06), transparent);
}}
.chat-avatar {{
    width:40px; height:40px; border-radius:12px;
    background:linear-gradient(135deg, {ACCENT_FROM}, {ACCENT_TO});
    display:flex; align-items:center; justify-content:center;
    color:#fff;
    box-shadow:0 6px 18px -6px rgba(59,130,246,0.6);
}}
.chat-name  {{ font-weight:700; color:{HEADING}; font-size:0.95rem; }}
.chat-stat  {{ font-size:0.72rem; color:#22c55e; display:flex; align-items:center; gap:6px; }}
.chat-stat::before {{
    content:""; width:6px; height:6px; border-radius:50%; background:#22c55e;
    box-shadow:0 0 0 3px rgba(34,197,94,0.18);
}}
.chat-body {{
    padding:18px 20px;
    height:520px;
    overflow-y:auto; overflow-x:hidden;
    scroll-behavior:smooth;
    overscroll-behavior:contain;
}}
.chat-body::-webkit-scrollbar {{ width:6px; }}
.chat-body::-webkit-scrollbar-track {{ background:transparent; }}
.chat-body::-webkit-scrollbar-thumb {{
    background:rgba(148,178,230,0.20); border-radius:3px;
}}
.chat-body::-webkit-scrollbar-thumb:hover {{ background:rgba(148,178,230,0.35); }}

.row-user {{ display:flex; justify-content:flex-end; margin:10px 0; animation:fadeUp 0.25s ease both; }}
.row-bot  {{ display:flex; justify-content:flex-start; gap:10px; margin:10px 0; animation:fadeUp 0.25s ease both; }}
.bubble-user {{
    background:linear-gradient(135deg, {ACCENT_FROM}, {ACCENT_TO});
    color:#fff;
    border-radius:16px 16px 4px 16px;
    padding:10px 14px; max-width:75%; font-size:0.92rem; line-height:1.5;
    box-shadow:0 8px 22px -10px rgba(59,130,246,0.55);
}}
.bubble-bot {{
    background:{SURFACE_2};
    color:{TEXT};
    border-radius:4px 16px 16px 16px;
    padding:10px 14px; max-width:80%; font-size:0.92rem; line-height:1.5;
    border:1px solid {BORDER};
}}
.bot-avatar {{
    width:30px; height:30px; border-radius:50%;
    background:linear-gradient(135deg, {ACCENT_FROM}, {ACCENT_TO});
    flex:0 0 30px; display:flex; align-items:center; justify-content:center; color:#fff;
}}
.meta {{ font-size:0.66rem; opacity:0.7; margin-bottom:4px; font-weight:500; letter-spacing:0.04em; text-transform:uppercase; }}
.typing-dots {{ display:inline-flex; gap:4px; padding:6px 2px; }}
.typing-dots span {{
    width:6px; height:6px; border-radius:50%;
    background:{MUTED}; animation:typing 1.2s infinite ease-in-out;
}}
.typing-dots span:nth-child(2) {{ animation-delay:0.15s; }}
.typing-dots span:nth-child(3) {{ animation-delay:0.3s; }}
.empty-chat {{
    text-align:center; padding:50px 20px; color:{MUTED};
    display:flex; flex-direction:column; align-items:center; gap:10px;
}}
.empty-chat .glow {{
    width:64px; height:64px; border-radius:18px;
    background:linear-gradient(135deg, {ACCENT_FROM}, {ACCENT_TO});
    display:flex; align-items:center; justify-content:center; color:#fff;
    box-shadow:0 12px 36px -10px rgba(59,130,246,0.7);
}}

/* Suggestion pills */
.pill-btn {{
    display:inline-block; padding:7px 12px; border-radius:999px;
    background:rgba(59,130,246,0.08);
    border:1px solid rgba(59,130,246,0.22);
    color:{TEXT}; font-size:0.8rem; cursor:pointer; margin:3px;
    transition:all 0.2s ease;
}}
.pill-btn:hover {{ background:rgba(59,130,246,0.18); transform:translateY(-1px); }}

/* ── Section subtitle ─────────────────────────────────────────────── */
.sec-sub {{ font-size:0.9rem; color:{MUTED}; margin:-6px 0 22px 0; line-height:1.6; }}

/* ── Buttons (modern, gradient primary) ───────────────────────────── */
.stButton > button {{
    border-radius:12px !important;
    font-weight:600 !important;
    font-size:0.88rem !important;
    padding:0.6rem 1.15rem !important;
    border:1px solid {BORDER} !important;
    background:{SURFACE} !important;
    color:{TEXT} !important;
    transition:all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow:none !important;
    backdrop-filter:blur(10px);
}}
.stButton > button:hover {{
    border-color:{PRIMARY_HOV} !important;
    color:{PRIMARY_HOV} !important;
    transform:translateY(-1px);
    box-shadow:0 6px 16px -8px rgba(59,130,246,0.4) !important;
}}
.stButton > button[kind="primary"] {{
    background:linear-gradient(135deg, {ACCENT_FROM}, {ACCENT_TO}) !important;
    color:#fff !important;
    border-color:transparent !important;
    box-shadow:0 8px 22px -10px rgba(59,130,246,0.6) !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform:translateY(-2px);
    box-shadow:0 14px 30px -10px rgba(59,130,246,0.7) !important;
    filter:brightness(1.08);
}}

/* ── Inputs — strongly visible in BOTH modes ──────────────────────── */
/* The wrapper that BaseWeb places around the input — give it the visible chrome */
.stTextInput > div > div,
.stNumberInput > div > div,
.stTextArea > div > div,
[data-baseweb="input"],
[data-baseweb="textarea"] {{
    background:{CHAT_INPUT_BG} !important;
    border:1.5px solid {CHAT_INPUT_BORDER} !important;
    border-radius:12px !important;
    box-shadow:0 4px 14px -8px rgba(29,78,216,0.30),
               0 0 0 1px rgba(59,130,246,0.04) inset !important;
    transition:border-color 0.2s, box-shadow 0.2s !important;
}}
.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextArea > div > div:focus-within,
[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within {{
    border-color:{PRIMARY_HOV} !important;
    box-shadow:0 0 0 4px rgba(59,130,246,0.20),
               0 8px 22px -10px rgba(29,78,216,0.45) !important;
}}
/* The actual input element rides on the wrapper — keep it transparent */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {{
    background:transparent !important;
    border:none !important;
    color:{CHAT_INPUT_TEXT} !important;
    font-size:0.95rem !important;
    font-weight:500 !important;
    padding:0.7rem 0.95rem !important;
    caret-color:{PRIMARY_HOV} !important;
}}
.stTextInput > div > div > input::placeholder,
.stTextArea textarea::placeholder,
[data-baseweb="input"] input::placeholder {{
    color:{CHAT_INPUT_PH} !important;
    opacity:1 !important;
    font-weight:500 !important;
}}
.stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label, .stTextArea label {{
    font-size:0.78rem !important; font-weight:700 !important;
    color:{TEXT} !important; margin-bottom:6px !important;
    text-transform:uppercase; letter-spacing:0.08em;
}}

/* ── Chat input — blue-themed, high contrast in BOTH modes ───────── */
[data-testid="stChatInput"] {{
    background:transparent !important;
    padding-top:8px !important;
}}
/* The container that wraps the textarea + send button */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
.stChatInput > div {{
    background:{CHAT_INPUT_BG} !important;
    border:1.5px solid {CHAT_INPUT_BORDER} !important;
    border-radius:16px !important;
    box-shadow:0 8px 24px -12px rgba(29,78,216,0.35),
               0 0 0 1px rgba(59,130,246,0.06) inset !important;
    transition:border-color 0.2s ease, box-shadow 0.2s ease !important;
}}
[data-testid="stChatInput"] > div:focus-within,
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {{
    border-color:{PRIMARY_HOV} !important;
    box-shadow:0 0 0 4px rgba(59,130,246,0.22),
               0 10px 28px -10px rgba(29,78,216,0.45) !important;
}}
[data-testid="stChatInput"] textarea,
.stChatInput textarea {{
    background:transparent !important;
    border:none !important;
    color:{CHAT_INPUT_TEXT} !important;
    font-size:0.95rem !important;
    font-weight:500 !important;
    padding:14px 16px !important;
    caret-color:{PRIMARY_HOV} !important;
}}
[data-testid="stChatInput"] textarea::placeholder,
.stChatInput textarea::placeholder {{
    color:{CHAT_INPUT_PH} !important;
    font-weight:500 !important;
    opacity:1 !important;
}}
/* Send button inside the chat input */
[data-testid="stChatInput"] button {{
    background:linear-gradient(135deg, {ACCENT_FROM}, {ACCENT_TO}) !important;
    border:none !important;
    border-radius:12px !important;
    color:#fff !important;
    box-shadow:0 6px 16px -6px rgba(59,130,246,0.55) !important;
    transition:transform 0.15s ease, box-shadow 0.2s ease !important;
}}
[data-testid="stChatInput"] button:hover {{
    transform:translateY(-1px);
    box-shadow:0 10px 22px -8px rgba(59,130,246,0.7) !important;
    filter:brightness(1.08);
}}
[data-testid="stChatInput"] button svg {{
    color:#fff !important; fill:#fff !important;
}}

/* ── Unified card via st.container(border=True) ──────────────────── */
/* Streamlit wraps bordered containers in a known wrapper — restyle it */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background:{SURFACE} !important;
    backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border:1px solid {BORDER} !important;
    border-radius:16px !important;
    padding:18px 20px !important;
    box-shadow:{SHADOW} !important;
    transition:border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    display:flex !important;
    flex-direction:column !important;
}}
/* Inner stack inside a bordered card stretches so children stay top-aligned */
[data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {{
    flex:1 1 auto !important;
    min-height:0 !important;
    justify-content:flex-start !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color:{BORDER_HOVER} !important;
    box-shadow:{GLOW} !important;
}}
/* Heading lockup that lives INSIDE the bordered container */
.uni-card-head {{
    display:flex; align-items:center; gap:10px;
    font-size:0.98rem; font-weight:700; color:{HEADING};
    letter-spacing:-0.01em;
    padding-bottom:12px; margin-bottom:14px;
    border-bottom:1px solid {BORDER};
}}
.uni-card-head .ico {{ color:{PRIMARY_HOV}; display:inline-flex; }}
.uni-card-sub {{
    color:{MUTED}; font-size:0.85rem; line-height:1.5;
    margin:-8px 0 14px 0;
}}

/* ── Sliders ──────────────────────────────────────────────────────── */
.stSlider > div > div > div > div {{
    background:linear-gradient(90deg, {ACCENT_FROM}, {ACCENT_TO}) !important;
}}
.stSlider [role="slider"] {{
    background:#fff !important;
    border:2px solid {PRIMARY_HOV} !important;
    box-shadow:0 4px 12px -2px rgba(59,130,246,0.5) !important;
}}

/* ── Alerts & misc ────────────────────────────────────────────────── */
.stAlert {{
    border-radius:14px !important;
    border:1px solid {BORDER} !important;
    backdrop-filter:blur(10px);
}}
[data-testid="stMetric"] {{
    background:{SURFACE};
    backdrop-filter:blur(14px);
    border:1px solid {BORDER};
    border-radius:14px; padding:16px 18px;
    transition:transform 0.2s ease;
}}
[data-testid="stMetric"]:hover {{ transform:translateY(-2px); border-color:{BORDER_HOVER}; }}
[data-testid="stMetricLabel"] {{ color:{MUTED} !important; font-weight:600 !important; font-size:0.72rem !important; text-transform:uppercase; letter-spacing:0.08em; }}
[data-testid="stMetricValue"] {{ color:{TEXT} !important; font-weight:800 !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap:4px; background:transparent; border-bottom:1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{MUTED} !important;
    border-radius:10px 10px 0 0 !important; padding:11px 18px !important;
    font-weight:600 !important; font-size:0.88rem !important;
    border:none !important;
    transition:color 0.2s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ color:{TEXT} !important; }}
.stTabs [aria-selected="true"] {{
    color:{PRIMARY_HOV} !important;
    box-shadow:inset 0 -2px 0 {PRIMARY_HOV} !important;
}}

/* ── Expander — kill any leaked Material-icon text, add custom chevron ── */
.streamlit-expanderHeader, [data-testid="stExpander"] details > summary {{
    background:{SURFACE} !important;
    border:1px solid {BORDER} !important;
    border-radius:12px !important;
    font-weight:700 !important;
    color:{TEXT} !important;
    padding:12px 16px !important;
    list-style:none !important;
    display:flex !important; align-items:center !important;
    gap:10px !important;
    cursor:pointer;
    transition:border-color 0.2s ease, background 0.2s ease;
}}
[data-testid="stExpander"] details > summary:hover {{
    border-color:{BORDER_HOVER} !important;
}}
/* Hide the default disclosure marker AND any leaked material icon text */
[data-testid="stExpander"] details > summary::-webkit-details-marker {{ display:none !important; }}
[data-testid="stExpander"] details > summary::marker {{ display:none !important; content:"" !important; }}
[data-testid="stExpander"] details > summary [class*="material"],
[data-testid="stExpander"] details > summary [data-testid*="icon"] + span:not([data-testid]),
[data-testid="stExpander"] summary span.material-symbols-rounded,
[data-testid="stExpander"] summary span.material-symbols-outlined,
[data-testid="stExpander"] summary span.material-icons {{
    font-size:0 !important; width:0 !important; height:0 !important;
    overflow:hidden !important; opacity:0 !important;
}}
/* Also catch the bare-text fallback when the icon font fails to load */
[data-testid="stExpander"] details > summary > span:first-child:not([data-testid]) {{
    font-size:0 !important; opacity:0 !important;
}}
/* Our own chevron — always visible, animates on open */
[data-testid="stExpander"] details > summary::after {{
    content:"";
    margin-left:auto;
    width:8px; height:8px;
    border-right:2px solid {MUTED};
    border-bottom:2px solid {MUTED};
    transform:rotate(-45deg);
    transition:transform 0.2s ease;
    flex-shrink:0;
}}
[data-testid="stExpander"] details[open] > summary::after {{
    transform:rotate(45deg);
    border-color:{PRIMARY_HOV};
}}

/* Dataframe */
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}

/* Progress bar */
.stProgress > div > div > div {{
    background:linear-gradient(90deg, {ACCENT_FROM}, {ACCENT_TO}) !important;
    border-radius:999px;
}}

/* Toggle */
.stCheckbox, [data-testid="stToggle"] label {{ color:{SB_TEXT} !important; }}

/* AQI scale rows */
.scale-row {{
    display:flex; align-items:center; justify-content:space-between;
    padding:9px 12px; border-radius:10px; margin:5px 0;
    border:1px solid {BORDER};
    transition:transform 0.15s ease;
}}
.scale-row:hover {{ transform:translateX(3px); }}

/* Feature row in Home — each row is its OWN symmetric mini-card */
.feat-row {{
    display:flex; align-items:flex-start; gap:14px;
    padding:14px 16px;
    margin:8px 0;
    border-radius:14px;
    background:{SURFACE};
    border:1px solid {BORDER};
    box-shadow:0 4px 14px -8px rgba(15, 32, 64, 0.18);
    transition:transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}}
.feat-row:hover {{
    transform:translateY(-2px);
    border-color:{BORDER_HOVER};
    box-shadow:0 10px 24px -10px rgba(29, 78, 216, 0.28);
}}
.feat-icon {{
    width:38px; height:38px; border-radius:11px; flex:0 0 38px;
    background:linear-gradient(135deg, rgba(29,78,216,0.18), rgba(59,130,246,0.18));
    border:1px solid rgba(59,130,246,0.22);
    display:flex; align-items:center; justify-content:center; color:{PRIMARY_HOV};
}}
.feat-title {{ font-weight:700; color:{TEXT}; font-size:0.95rem; }}
.feat-desc  {{ color:{MUTED}; font-size:0.82rem; margin-top:2px; line-height:1.5; }}

/* Globe shell for Real-Time AQI */
.globe-wrap {{
    position:relative; width:100%; aspect-ratio:1/1; max-width:360px; margin:0 auto;
    border-radius:50%;
    background:radial-gradient(circle at 30% 30%, #3B82F6 0%, #1D4ED8 35%, #0B1F3A 80%);
    box-shadow:
        inset -20px -30px 60px rgba(0,0,0,0.5),
        inset 20px 30px 60px rgba(59,130,246,0.18),
        0 30px 80px -20px rgba(59,130,246,0.45);
    overflow:hidden;
    animation:fadeIn 0.6s ease both;
}}
.globe-wrap::before {{
    content:""; position:absolute; inset:0; border-radius:50%;
    background:
      radial-gradient(circle at 25% 25%, rgba(255,255,255,0.16), transparent 35%),
      repeating-linear-gradient(0deg, transparent, transparent 28px, rgba(255,255,255,0.06) 28px, rgba(255,255,255,0.06) 29px),
      repeating-linear-gradient(90deg, transparent, transparent 28px, rgba(255,255,255,0.06) 28px, rgba(255,255,255,0.06) 29px);
}}
.globe-wrap::after {{
    content:""; position:absolute; inset:-10%; border-radius:50%;
    background:conic-gradient(from 180deg, transparent 0deg, rgba(59,130,246,0.18) 60deg, transparent 120deg);
    animation:spin 14s linear infinite;
    mix-blend-mode:screen; pointer-events:none;
}}
.globe-pin {{
    position:absolute; top:50%; left:50%; transform:translate(-50%, -50%);
    z-index:2; text-align:center;
}}
.globe-pin .badge {{
    background:rgba(11,31,58,0.78);
    border:1px solid rgba(255,255,255,0.18);
    backdrop-filter:blur(8px);
    border-radius:14px; padding:14px 18px; color:#fff;
    box-shadow:0 18px 40px -10px rgba(0,0,0,0.6);
}}
.globe-pin .city {{ font-size:0.78rem; opacity:0.75; letter-spacing:0.04em; }}
.globe-pin .num  {{ font-size:2.2rem; font-weight:900; line-height:1.05; }}
.globe-pin .cat  {{ font-size:0.78rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-top:4px; }}
.orbit-ring {{
    position:absolute; inset:-6%;
    border-radius:50%;
    border:1px dashed rgba(255,255,255,0.12);
    pointer-events:none;
}}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def aqi_color(aqi):
    if aqi <= 50:   return "#22c55e"
    if aqi <= 100:  return "#eab308"
    if aqi <= 200:  return "#f97316"
    if aqi <= 300:  return "#ef4444"
    return "#a855f7"

def aqi_palette(cat):
    return AQI_PALETTE.get(cat, {"bg":SURFACE_2,"text":TEXT,"border":PRIMARY,"hex":PRIMARY})

def risk_color(risk):
    return {"LOW":"#22c55e","MEDIUM":"#eab308","HIGH":"#ef4444"}.get(risk, PRIMARY)

def plt_style():
    plt.rcParams.update({
        "figure.facecolor":PLT_BG,"axes.facecolor":PLT_BG,"axes.edgecolor":PLT_GRID,
        "axes.labelcolor":PLT_TEXT,"xtick.color":PLT_TEXT,"ytick.color":PLT_TEXT,
        "text.color":PLT_TEXT,"grid.color":PLT_GRID,"grid.alpha":0.7,
        "font.family":"DejaVu Sans",
    })


# ─────────────────────────────────────────
# GEOCODING — converts a city name to (lat, lon, pretty_name)
# Uses a tiny built-in dictionary first (instant, no network)
# then falls back to Open-Meteo's free geocoding API (no key required).
# ─────────────────────────────────────────
CITY_COORDS = {
    "delhi": (28.6139, 77.2090, "Delhi, India"),
    "new delhi": (28.6139, 77.2090, "New Delhi, India"),
    "mumbai": (19.0760, 72.8777, "Mumbai, India"),
    "kolkata": (22.5726, 88.3639, "Kolkata, India"),
    "chennai": (13.0827, 80.2707, "Chennai, India"),
    "bengaluru": (12.9716, 77.5946, "Bengaluru, India"),
    "bangalore": (12.9716, 77.5946, "Bengaluru, India"),
    "hyderabad": (17.3850, 78.4867, "Hyderabad, India"),
    "ahmedabad": (23.0225, 72.5714, "Ahmedabad, India"),
    "jaipur": (26.9124, 75.7873, "Jaipur, India"),
    "lucknow": (26.8467, 80.9462, "Lucknow, India"),
    "chandigarh": (30.7333, 76.7794, "Chandigarh, India"),
    "guwahati": (26.1445, 91.7362, "Guwahati, India"),
    "raipur": (21.2514, 81.6296, "Raipur, India"),
    "ranchi": (23.3441, 85.3096, "Ranchi, India"),
    "shimla": (31.1048, 77.1734, "Shimla, India"),
    "visakhapatnam": (17.6868, 83.2185, "Visakhapatnam, India"),
    "london": (51.5074, -0.1278, "London, UK"),
    "paris": (48.8566, 2.3522, "Paris, France"),
    "berlin": (52.5200, 13.4050, "Berlin, Germany"),
    "madrid": (40.4168, -3.7038, "Madrid, Spain"),
    "rome": (41.9028, 12.4964, "Rome, Italy"),
    "moscow": (55.7558, 37.6173, "Moscow, Russia"),
    "beijing": (39.9042, 116.4074, "Beijing, China"),
    "shanghai": (31.2304, 121.4737, "Shanghai, China"),
    "tokyo": (35.6762, 139.6503, "Tokyo, Japan"),
    "seoul": (37.5665, 126.9780, "Seoul, South Korea"),
    "bangkok": (13.7563, 100.5018, "Bangkok, Thailand"),
    "singapore": (1.3521, 103.8198, "Singapore"),
    "jakarta": (-6.2088, 106.8456, "Jakarta, Indonesia"),
    "dubai": (25.2048, 55.2708, "Dubai, UAE"),
    "karachi": (24.8607, 67.0011, "Karachi, Pakistan"),
    "lahore": (31.5204, 74.3587, "Lahore, Pakistan"),
    "dhaka": (23.8103, 90.4125, "Dhaka, Bangladesh"),
    "kathmandu": (27.7172, 85.3240, "Kathmandu, Nepal"),
    "new york": (40.7128, -74.0060, "New York, USA"),
    "los angeles": (34.0522, -118.2437, "Los Angeles, USA"),
    "chicago": (41.8781, -87.6298, "Chicago, USA"),
    "san francisco": (37.7749, -122.4194, "San Francisco, USA"),
    "toronto": (43.6532, -79.3832, "Toronto, Canada"),
    "mexico city": (19.4326, -99.1332, "Mexico City, Mexico"),
    "sao paulo": (-23.5505, -46.6333, "São Paulo, Brazil"),
    "buenos aires": (-34.6037, -58.3816, "Buenos Aires, Argentina"),
    "cairo": (30.0444, 31.2357, "Cairo, Egypt"),
    "lagos": (6.5244, 3.3792, "Lagos, Nigeria"),
    "johannesburg": (-26.2041, 28.0473, "Johannesburg, South Africa"),
    "sydney": (-33.8688, 151.2093, "Sydney, Australia"),
    "melbourne": (-37.8136, 144.9631, "Melbourne, Australia"),
    "auckland": (-36.8485, 174.7633, "Auckland, New Zealand"),
}

@st.cache_data(show_spinner=False, ttl=86400)
def geocode_city(name: str):
    """Return (lat, lon, pretty_name) for a city name. None on failure."""
    if not name:
        return None
    key = name.strip().lower()
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    if not REQUESTS_OK:
        return None
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name.strip(), "count": 1, "language": "en", "format": "json"},
            timeout=4,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("results"):
                hit = data["results"][0]
                pretty = hit.get("name", name.title())
                if hit.get("country"):
                    pretty += f", {hit['country']}"
                return (float(hit["latitude"]), float(hit["longitude"]), pretty)
    except Exception:
        pass
    return None


def build_earth_globe(lat=None, lon=None, label=None, aqi_v=None, color="#3B82F6"):
    """Real 3D earth (orthographic projection) with optional city marker."""
    if not PLOTLY_OK:
        return None

    fig = go.Figure()

    # Add a glow halo around the marker (larger semi-transparent ring)
    if lat is not None and lon is not None:
        for r, alpha in [(36, 0.18), (24, 0.32), (16, 0.55)]:
            fig.add_trace(go.Scattergeo(
                lon=[lon], lat=[lat],
                mode="markers",
                marker=dict(size=r, color=color, opacity=alpha,
                            line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ))
        hover = f"<b>{label or 'City'}</b>"
        if aqi_v is not None:
            hover += f"<br>AQI: <b>{aqi_v}</b>"
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat],
            mode="markers+text",
            marker=dict(size=10, color="#ffffff",
                        line=dict(color=color, width=3),
                        symbol="circle"),
            text=[label or ""], textposition="top center",
            textfont=dict(size=12, color="#ffffff", family="Inter"),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

    # Center the orthographic view on the city (or 0,0 default)
    center_lat = lat if lat is not None else 20
    center_lon = lon if lon is not None else 0

    # Theme-aware globe colors so the Earth has presence in BOTH dark & light modes.
    if DARK:
        # Brighten the planet enough to pop off the deep-navy page background
        _land   = "#3D6BAA"           # mid-blue land mass
        _ocean  = "#234B86"           # lighter than page bg (#0B1F3A) so globe is visible
        _country= "rgba(220,235,255,0.55)"
        _coast  = "rgba(235,245,255,0.85)"
        _grid   = "rgba(180,210,255,0.20)"
    else:
        _land   = "#7FA8DC"           # soft blue land
        _ocean  = "#3B82F6"           # vivid ocean for contrast on white
        _country= "rgba(255,255,255,0.65)"
        _coast  = "rgba(255,255,255,0.90)"
        _grid   = "rgba(255,255,255,0.25)"

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=center_lon, lat=center_lat, roll=0),
        showland=True,    landcolor=_land,
        showocean=True,   oceancolor=_ocean,
        showcountries=True, countrycolor=_country,
        showcoastlines=True, coastlinecolor=_coast,
        coastlinewidth=0.8,
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
        lataxis=dict(showgrid=True, gridcolor=_grid, gridwidth=0.5),
        lonaxis=dict(showgrid=True, gridcolor=_grid, gridwidth=0.5),
    )
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="orbit",
    )
    return fig


def features_for_target_aqi(target_aqi: float):
    base = get_default_features()
    base_pred = predict_aqi(base) or 100.0
    if base_pred <= 0:
        return base
    scale = float(target_aqi) / float(base_pred)
    scaled = list(base)
    for i in range(8):
        scaled[i] = max(0.0, base[i] * scale)
    return scaled


# ─────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(n=80000):
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    return df.sample(min(n, len(df)), random_state=42)

@st.cache_data(show_spinner=False)
def city_avg(_df):
    return pd.Series({c: _df[_df[c]==1]["us_aqi"].mean() if (_df[c]==1).sum()>0 else 0
                      for c in CITY_COLS}).sort_values(ascending=False)

@st.cache_data(show_spinner=False)
def season_avg(_df):
    s = _df.groupby("season")["us_aqi"].mean()
    s.index = [SEASON_MAP.get(i, str(i)) for i in s.index]
    return s

@st.cache_data(show_spinner=False)
def corr_matrix(_df):
    cols = ['pm2_5_ugm3','pm10_ugm3','co_ugm3','no2_ugm3','so2_ugm3','o3_ugm3','dust_ugm3','us_aqi']
    return _df[[c for c in cols if c in _df.columns]].corr()

@st.cache_data(show_spinner=False)
def cat_counts(_df):
    return _df["us_aqi"].apply(get_aqi_category).value_counts()

@st.cache_data(show_spinner=False)
def model_perf(_df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler as SK
    from sklearn.linear_model    import LinearRegression
    from sklearn.tree            import DecisionTreeRegressor
    from sklearn.ensemble        import RandomForestRegressor
    from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
    feat = [c for c in FEATURE_NAMES if c in _df.columns]
    sub  = _df[feat+["us_aqi"]].dropna().sample(min(30000,len(_df)), random_state=42)
    X, y = sub[feat], sub["us_aqi"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    sc = SK(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    out = {}
    for name, mdl in [("Linear Regression",LinearRegression()),
                      ("Decision Tree",DecisionTreeRegressor(max_depth=10,random_state=42)),
                      ("Random Forest",RandomForestRegressor(n_estimators=50,max_depth=10,n_jobs=-1,random_state=42))]:
        mdl.fit(Xtr_s,ytr); p=mdl.predict(Xte_s)
        out[name] = {"R²":round(r2_score(yte,p),4),
                     "RMSE":round(np.sqrt(mean_squared_error(yte,p)),2),
                     "MAE":round(mean_absolute_error(yte,p),2)}
    return out


# ─────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────
NAV_ITEMS = ["Home","Chatbot","AQI Prediction","Forecast",
             "Data Analysis","Model Performance","Real-Time AQI"]
NAV_ICONS = {
    "Home":"home", "Chatbot":"chat-bubble", "AQI Prediction":"sliders",
    "Forecast":"trending", "Data Analysis":"bar",
    "Model Performance":"trophy", "Real-Time AQI":"globe",
}
NAV_DISPLAY = {
    "Chatbot":  "AI Chatbot",
    "Forecast": "7-Day Forecast",
}

with st.sidebar:
    st.markdown(f"""
    <div style='padding:22px 14px 8px 14px;'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <div style='width:42px;height:42px;border-radius:12px;
                        background:linear-gradient(135deg,{ACCENT_FROM},{ACCENT_TO});
                        display:flex;align-items:center;justify-content:center;
                        color:#fff; box-shadow:0 8px 22px -8px rgba(59,130,246,0.7);'>
                {icon("wind", 22, "#fff", 2)}
            </div>
            <div>
                <div style='font-size:1rem;font-weight:800;color:{SB_TEXT};line-height:1.1;letter-spacing:-0.01em;'>AQI Dashboard</div>
                <div style='font-size:0.7rem;color:{SB_MUTED};margin-top:3px;letter-spacing:0.04em;'>Smart Air Quality System</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    section = st.radio("NAVIGATE", NAV_ITEMS,
                       format_func=lambda x: NAV_DISPLAY.get(x, x),
                       index=NAV_ITEMS.index(st.session_state.section)
                       if st.session_state.section in NAV_ITEMS else 0)
    st.session_state.section = section

    st.markdown("<hr>", unsafe_allow_html=True)
    toggle = st.toggle("Dark Mode", value=st.session_state.dark)
    if toggle != st.session_state.dark:
        st.session_state.dark = toggle
        st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)

    # Persistent pills
    if st.session_state.predicted_aqi is not None:
        cat_p = get_aqi_category(st.session_state.predicted_aqi)
        p_pal = aqi_palette(cat_p)
        st.markdown(f"""
        <div style='background:{SB_SURFACE};border-radius:12px;padding:14px;
                    border:1px solid {SB_BORDER};margin-bottom:10px;
                    background-image:linear-gradient(135deg, rgba(29,78,216,0.10), rgba(59,130,246,0.04));'>
            <div style='font-size:0.62rem;color:{SB_MUTED};text-transform:uppercase;
                        letter-spacing:0.14em;font-weight:700;'>Last Prediction</div>
            <div style='display:flex;align-items:baseline;gap:8px;margin-top:8px;'>
                <div style='font-weight:800;font-size:1.5rem;color:{p_pal["hex"]};line-height:1;'>
                    {st.session_state.predicted_aqi:.0f}
                </div>
                <div style='font-size:0.78rem;color:{SB_TEXT};font-weight:600;'>{cat_p}</div>
            </div>
            <div style='font-size:0.66rem;color:{SB_MUTED};margin-top:6px;'>Manual inputs</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.fc_live_aqi is not None and st.session_state.fc_live_name:
        cat_c = get_aqi_category(st.session_state.fc_live_aqi)
        c_pal = aqi_palette(cat_c)
        st.markdown(f"""
        <div style='background:{SB_SURFACE};border-radius:12px;padding:14px;
                    border:1px solid {SB_BORDER};margin-bottom:10px;'>
            <div style='font-size:0.62rem;color:{SB_MUTED};text-transform:uppercase;
                        letter-spacing:0.14em;font-weight:700;'>Last City</div>
            <div style='font-size:0.95rem;font-weight:700;color:{SB_TEXT};margin-top:6px;line-height:1.2;'>
                {st.session_state.fc_live_name}
            </div>
            <div style='display:flex;align-items:baseline;gap:6px;margin-top:4px;'>
                <span style='font-weight:800;color:{c_pal["hex"]};font-size:0.95rem;'>
                    AQI {st.session_state.fc_live_aqi:.0f}
                </span>
                <span style='font-size:0.72rem;color:{SB_MUTED};'>· {cat_c}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.predicted_aqi is not None or st.session_state.fc_live_aqi is not None:
        st.markdown("")

    if st.button("Reset all inputs", use_container_width=True):
        for k in SESSION_DEFAULTS:
            st.session_state[k] = SESSION_DEFAULTS[k]
        st.rerun()

    if section == "Chatbot" and st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ctx_city = None
        st.session_state.ctx_aqi  = None
        st.rerun()


# ─────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────
title_map = {
    "Home":              ("AQI Smart Assistant", "AI-powered Air Quality Intelligence Platform", "wind"),
    "Chatbot":           ("AI Chatbot", "Ask anything about air quality, health, and forecasts", "bot"),
    "AQI Prediction":    ("AQI Prediction", "Enter pollutant levels manually to get the predicted AQI", "chart"),
    "Forecast":          ("7-Day Forecast", "AI-generated air quality forecast for the next week", "trending"),
    "Data Analysis":     ("Data Analysis", "Insights from the AQI dataset", "lab"),
    "Model Performance": ("Model Performance", "Comparing Linear Regression · Decision Tree · Random Forest", "trophy"),
    "Real-Time AQI":     ("Real-Time AQI", "Live data from WAQI API — any city, worldwide", "globe"),
}
t_title, t_sub, t_icon = title_map.get(section, ("AQI Dashboard","","wind"))
st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-icon">{icon(t_icon, 22, "#fff", 2)}</div>
        <div>
            <h1>{t_title}</h1><p>{t_sub}</p>
        </div>
    </div>
    <div class="status-pill">
        <span class="dot"></span>
        <span class="label">Live</span>
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════
if section == "Home":
    feats = st.session_state.features or get_default_features()
    pred  = st.session_state.predicted_aqi or (predict_aqi(feats) or 120.0)
    cat   = get_aqi_category(pred)
    risk  = get_aqi_risk(pred)
    p     = aqi_palette(cat)

    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        (c1, f"{round(pred,0):.0f}", "Current AQI",  aqi_color(pred), "activity"),
        (c2, cat,                    "Category",      p["hex"],        "layers"),
        (c3, risk,                   "Risk Level",     risk_color(risk),"shield"),
        (c4, "LIVE",                 "API Status",    "#22c55e",        "globe"),
    ]
    for col, val, label, color, ic in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-head">
                    <div class="kpi-icon" style="color:{color};border-color:{color}55;background:linear-gradient(135deg, {color}22, {color}11);">{icon(ic, 18, color, 2)}</div>
                    <div style="font-size:0.65rem;color:{MUTED};font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">Now</div>
                </div>
                <div class="kpi-val" style="color:{color};">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.4, 1])

    with col_l:
        with card("What can I do for you?", icon_name="sparkles"):
            feats_list = [
                ("bot",      "AI Chatbot",         "Ask anything — health advice, city AQI, causes, precautions"),
                ("chart",    "AQI Prediction",     "Enter pollutant levels OR a city to get a predicted AQI"),
                ("trending", "7-Day Forecast",     "Day 1 = your last prediction; trend chart for the rest of the week"),
                ("lab",      "Data Analysis",      "City rankings, seasonal trends, correlation maps"),
                ("trophy",   "Model Performance",  "Compare Linear Regression · Decision Tree · Random Forest"),
                ("globe",    "Real-Time AQI",      "Fetch live AQI for any city worldwide"),
            ]
            for ic, title, desc in feats_list:
                st.markdown(f"""
                <div class="feat-row">
                    <div class="feat-icon">{icon(ic, 18, PRIMARY_HOV, 2)}</div>
                    <div>
                        <div class="feat-title">{title}</div>
                        <div class="feat-desc">{desc}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with col_r:
        with card("Quick City Check", icon_name="pin"):
            quick_city = st.text_input("City name", placeholder="Delhi, Mumbai, London…", key="home_city", label_visibility="collapsed")
            if st.button("Check AQI", use_container_width=True, type="primary", key="home_check"):
                if quick_city.strip():
                    with st.spinner("Fetching…"):
                        qa, qn = get_real_aqi(quick_city.strip().lower())
                    if qa:
                        qcat = get_aqi_category(qa); qp = aqi_palette(qcat)
                        st.markdown(f"""
                        <div class="aqi-result" style="background:{qp['bg']};border-color:{qp['border']};margin-top:14px;">
                            <div class="aqi-big" style="color:{qp['hex']};">{qa}</div>
                            <div class="aqi-cat"  style="color:{qp['hex']};">{qcat}</div>
                            <div class="aqi-risk" style="color:{qp['hex']};opacity:0.85;">{get_aqi_risk(qa)} RISK</div>
                            <div style='font-size:0.8rem;color:{MUTED};margin-top:10px;'>{qn}</div>
                        </div>""", unsafe_allow_html=True)
                        st.session_state.ctx_city = quick_city.strip().lower()
                        st.session_state.ctx_aqi  = qa
                    else:
                        st.error("City not found. Try again.")

        with card("AQI Scale", icon_name="scale"):
            for cat_name, rng in [("Good","0–50"),("Moderate","51–100"),("Unhealthy","101–200"),
                                  ("Poor","201–300"),("Hazardous","300+")]:
                p2 = aqi_palette(cat_name)
                st.markdown(f"""
                <div class="scale-row" style="background:{p2['bg']};border-color:{p2['border']};">
                    <span style='display:flex;align-items:center;gap:8px;color:{p2["hex"]};font-weight:700;font-size:0.85rem;'>
                        <span style="width:8px;height:8px;border-radius:50%;background:{p2['hex']};box-shadow:0 0 0 3px {p2['hex']}33;"></span>
                        {cat_name}
                    </span>
                    <span style='color:{p2["hex"]};font-size:0.8rem;font-weight:600;opacity:0.9;'>{rng}</span>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# CHATBOT
# ═══════════════════════════════════════════════════════
elif section == "Chatbot":
    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">
                <span class="ico">{icon("chat-bubble",18,PRIMARY_HOV,2)}</span> Quick Questions
            </div>
            <div style="color:{MUTED};font-size:0.82rem;margin:-6px 0 12px 0;">
                Try one of these to get started
            </div>
        </div>
        """, unsafe_allow_html=True)
        examples = [
            "AQI in Delhi?", "AQI in Mumbai?", "7-day forecast?",
            "AQI after 3 days?", "Should I wear a mask?",
            "Safe for kids outside?", "Can I exercise today?",
            "What causes high AQI?", "Health effects of AQI?",
            "Precautions for AQI 250?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"q_{ex}"):
                st.session_state["_qi"] = ex

    with col_main:
        st.markdown(f'<div class="chat-shell">', unsafe_allow_html=True)

        # Header row with Clear button on the right
        h_l, h_r = st.columns([5, 1])
        with h_l:
            st.markdown(f"""
            <div class="chat-header" style="border-bottom:none;">
                <div class="chat-avatar">{icon("bot", 20, "#fff", 2)}</div>
                <div>
                    <div class="chat-name">AQI Assistant</div>
                    <div class="chat-stat">Online</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with h_r:
            st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
            if st.button("Clear", key="chat_clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown(f'<div style="border-top:1px solid {BORDER};margin:0 0 6px 0;"></div>',
                    unsafe_allow_html=True)

        msgs_html = ""
        if not st.session_state.messages:
            msgs_html = f"""
            <div class="empty-chat">
                <div class="glow">{icon("bot", 28, "#fff", 2)}</div>
                <div style='font-size:1.05rem;font-weight:700;color:{TEXT};margin-top:6px;'>Hello! I'm your AQI Assistant.</div>
                <div style='font-size:0.88rem;color:{MUTED};max-width:360px;line-height:1.55;'>
                    Ask me about air quality, health effects, forecasts, or real-time AQI for any city.
                </div>
            </div>"""
        else:
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"].replace("\n","<br>")
                if role == "user":
                    msgs_html += f"""
                    <div class="row-user">
                        <div style="display:flex;flex-direction:column;align-items:flex-end;max-width:80%;">
                            <div class="meta" style="margin-right:4px;">YOU</div>
                            <div class="bubble-user">{content}</div>
                        </div>
                    </div>"""
                else:
                    msgs_html += f"""
                    <div class="row-bot">
                        <div style="flex:1;min-width:0;">
                            <div class="meta" style="margin-left:4px;">AQI ASSISTANT</div>
                            <div class="bubble-bot">{content}</div>
                        </div>
                    </div>"""

        st.markdown(f'<div class="chat-body">{msgs_html}</div></div>', unsafe_allow_html=True)

        pending = st.session_state.pop("_qi", None)
        user_in = st.chat_input("Ask about air quality, health, forecasts…")
        final   = pending or user_in

        if final:
            st.session_state.messages.append({"role":"user","content":final})
            with st.spinner("Thinking…"):
                resp, nc, na = get_response(
                    final,
                    context_city        = st.session_state.ctx_city,
                    context_aqi         = st.session_state.ctx_aqi,
                    saved_features      = st.session_state.features,
                    saved_predicted_aqi = st.session_state.predicted_aqi,
                )
            if nc: st.session_state.ctx_city = nc
            if na: st.session_state.ctx_aqi  = na
            st.session_state.messages.append({"role":"assistant","content":resp})
            st.rerun()


# ═══════════════════════════════════════════════════════
# AQI PREDICTION  (manual pollutant sliders only)
# ═══════════════════════════════════════════════════════
elif section == "AQI Prediction":
    st.markdown(f'<p class="sec-sub">Enter pollutant readings and click Predict AQI.</p>',
                unsafe_allow_html=True)

    dflt = st.session_state.features or get_default_features()

    # Slider config — order MUST match FEATURE_NAMES expected by predict_aqi.
    slider_cfg = [
        ("PM2.5 (µg/m³)",     "pm2_5_ugm3",          0.,  500.,  float(dflt[0])),
        ("PM10 (µg/m³)",      "pm10_ugm3",           0.,  600.,  float(dflt[1])),
        ("CO (µg/m³)",        "co_ugm3",             0.,  5000., float(dflt[2])),
        ("NO₂ (µg/m³)",       "no2_ugm3",            0.,  300.,  float(dflt[3])),
        ("SO₂ (µg/m³)",       "so2_ugm3",            0.,  200.,  float(dflt[4])),
        ("O₃ (µg/m³)",        "o3_ugm3",             0.,  300.,  float(dflt[5])),
        ("Dust (µg/m³)",      "dust_ugm3",           0.,  100.,  float(dflt[6])),
        ("AOD",               "aod",                 0.,  3.,    float(dflt[7])),
        ("Humidity (%)",      "humidity_percent",    0.,  100.,  float(dflt[8])),
        ("Dew Point (°C)",    "dew_point_c",        -30., 40.,   float(dflt[9])),
        ("Wind Gusts (km/h)", "wind_gusts_kmh",      0.,  150.,  float(dflt[10])),
        ("Pressure (hPa)",    "pressure_msl_hpa",   900., 1100., float(dflt[11])),
        ("Cloud Cover (%)",   "cloud_cover_percent", 0.,  100.,  float(dflt[12])),
    ]

    with card("Pollutant & Weather Inputs", icon_name="settings"):
        vals = []
        cols3 = st.columns(3)
        for i, (label, key, mn, mx, dft) in enumerate(slider_cfg):
            with cols3[i % 3]:
                vals.append(st.slider(label, float(mn), float(mx), float(dft), key=f"s_{key}"))

    cpa, cpb = st.columns([1, 4])
    with cpa:
        do_predict = st.button("Predict AQI", type="primary", use_container_width=True)
    with cpb:
        st.caption("Adjust the sliders above to model a scenario, then click Predict AQI.")

    if do_predict:
        with st.spinner("Predicting…"):
            res = predict_aqi(vals)
        if res is None:
            st.error("Prediction failed — check model files.")
        else:
            st.session_state.features      = vals
            st.session_state.predicted_aqi = round(float(res), 2)
            st.session_state.fc_result     = None

    if st.session_state.predicted_aqi is not None:
        av   = round(st.session_state.predicted_aqi, 1)
        cat  = get_aqi_category(av)
        risk = get_aqi_risk(av)
        p    = aqi_palette(cat)
        rc   = risk_color(risk)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        ra, rb, rc2, rd = st.columns(4)
        for col, val, lbl, clr, ic in [
            (ra, str(av),  "Predicted AQI",   p["hex"], "activity"),
            (rb, cat,      "Category",         p["hex"], "layers"),
            (rc2, risk,    "Risk Level",        rc,      "shield"),
            (rd, f"{min(av/500*100,100):.0f}%", "Pollution %", p["hex"], "spark"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi">
                    <div class="kpi-head">
                        <div class="kpi-icon" style="color:{clr};border-color:{clr}55;background:linear-gradient(135deg, {clr}22, {clr}11);">{icon(ic, 18, clr, 2)}</div>
                    </div>
                    <div class="kpi-val" style="color:{clr};font-size:{'1.6rem' if len(val)>5 else '2rem'};">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="aqi-result" style="background:{p['bg']};border-color:{p['border']};">
            <div class="aqi-big" style="color:{p['hex']};">{av}</div>
            <div class="aqi-cat"  style="color:{p['hex']};">{cat}</div>
            <div class="aqi-risk" style="color:{p['hex']};opacity:0.85;">{risk} RISK</div>
            <div style='font-size:0.82rem;color:{MUTED};margin-top:12px;'>
                Source: manual pollutant inputs
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"**AQI Progress — {av} / 500**")
        st.progress(min(av/500, 1.0))

        if av > 200:
            st.error(f"ALERT: AQI {av} is **{cat}**. Stay indoors, use N95 mask.")
        elif av > 100:
            st.warning(f"CAUTION: AQI {av} (**{cat}**). Sensitive groups should limit outdoor exposure.")
        else:
            st.success(f"SAFE: AQI {av} (**{cat}**). Air quality is acceptable for most people.")

        st.info("Open **Forecast** — leave the city blank to forecast from this prediction, "
                "or type a city to anchor Day 1 to live AQI.")


# ═══════════════════════════════════════════════════════
# FORECAST
# ═══════════════════════════════════════════════════════
elif section == "Forecast":
    # ── Forecast Source card (heading + inputs in ONE container) ─────
    with card("Forecast Source",
              subtitle="Leave city empty to use your latest prediction as Day 1",
              icon_name="globe"):
        fc1, fc2 = st.columns([3, 1])
        with fc1:
            city_in = st.text_input("City (optional)",
                                    value=st.session_state.fc_city,
                                    placeholder="Delhi, Mumbai, London…  (leave blank to use prediction)",
                                    key="fc_city_in",
                                    label_visibility="collapsed")
        with fc2:
            run = st.button("Generate Forecast", type="primary", use_container_width=True)

    if run:
        st.session_state.fc_city = city_in.strip()

        if city_in.strip():
            with st.spinner(f"Fetching live AQI for {city_in}…"):
                aqi_v, name_v = get_real_aqi(city_in.strip().lower())
            if aqi_v is None:
                st.error(f"Could not fetch live AQI for **{city_in}**. "
                         "Try Delhi, Mumbai, London, Beijing, etc.")
                st.session_state.fc_result = None
            else:
                st.session_state.fc_live_aqi      = float(aqi_v)
                st.session_state.fc_live_name     = name_v
                st.session_state.fc_anchor_source = "city"
                features = list(st.session_state.features) \
                            if st.session_state.features \
                            else features_for_target_aqi(aqi_v)
                fc = forecast_next_7_days(features, base_aqi=float(aqi_v))
                fc[0] = round(float(aqi_v), 2)
                st.session_state.fc_result = fc
        else:
            if st.session_state.predicted_aqi is None:
                st.warning("No manual prediction yet. Either type a city above, "
                           "or open **AQI Prediction** first to predict an AQI.")
                st.session_state.fc_result = None
            else:
                st.session_state.fc_live_aqi      = None
                st.session_state.fc_live_name     = None
                st.session_state.fc_anchor_source = "prediction"
                features = list(st.session_state.features) \
                            if st.session_state.features \
                            else get_default_features()
                anchor   = float(st.session_state.predicted_aqi)
                fc = forecast_next_7_days(features, base_aqi=anchor)
                fc[0] = round(anchor, 2)
                st.session_state.fc_result = fc

    fc = st.session_state.fc_result
    if not fc:
        st.info("Enter a city for a live baseline, or leave it blank to forecast "
                "from your last manual prediction. Then click **Run Forecast**.")
    else:
        if st.session_state.fc_anchor_source == "city":
            anchor_aqi   = float(st.session_state.fc_live_aqi)
            anchor_label = f"live AQI for **{st.session_state.fc_live_name}**"
        else:
            anchor_aqi   = float(st.session_state.predicted_aqi or fc[0])
            anchor_label = "your last manual prediction"

        anchor_cat = get_aqi_category(anchor_aqi)
        p_anchor   = aqi_palette(anchor_cat)

        assert abs(fc[0] - anchor_aqi) < 1e-6, "Day 1 must equal anchor AQI"

        days = [f"Day {i+1}" for i in range(len(fc))]
        anchor_text = (f"Anchored to live AQI for {st.session_state.fc_live_name}"
                       if st.session_state.fc_anchor_source == "city"
                       else f"Anchored to your last manual prediction ({anchor_aqi:.0f})")

        # ── Two-column layout: Trend Chart (left) + Daily Forecast (right) ──
        left, right = st.columns([1.7, 1])

        with left:
            with card("Trend Chart", subtitle=anchor_text, icon_name="trending"):
                if PLOTLY_OK:
                    x = list(range(1, len(fc) + 1))
                    fig = go.Figure()

                    # Soft area fill (layered for gradient feel)
                    fig.add_trace(go.Scatter(
                        x=x, y=fc,
                        mode="lines",
                        line=dict(shape="spline", smoothing=1.1,
                                  color="rgba(59,130,246,0)", width=0),
                        fill="tozeroy",
                        fillcolor="rgba(59,130,246,0.22)",
                        hoverinfo="skip", showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=x, y=fc,
                        mode="lines",
                        line=dict(shape="spline", smoothing=1.1,
                                  color="rgba(59,130,246,0)", width=0),
                        fill="tozeroy",
                        fillcolor="rgba(29,78,216,0.10)",
                        hoverinfo="skip", showlegend=False,
                    ))
                    # Smooth main line
                    fig.add_trace(go.Scatter(
                        x=x, y=fc,
                        mode="lines+markers",
                        line=dict(shape="spline", smoothing=1.1,
                                  color=PRIMARY_HOV, width=3),
                        marker=dict(size=8, color=PRIMARY_HOV,
                                    line=dict(color="#0B1F3A", width=2)),
                        hovertemplate="<b>Day %{x}</b><br>AQI = %{y:.0f}<extra></extra>",
                        showlegend=False,
                    ))

                    ymax = max(max(fc) * 1.25, 160)
                    fig.update_layout(
                        height=380, margin=dict(l=10, r=10, t=10, b=30),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=PLT_TEXT, family="Inter", size=11),
                        xaxis=dict(
                            tickmode="array", tickvals=x,
                            ticktext=days,
                            showgrid=True, gridcolor="rgba(148,178,230,0.08)",
                            zeroline=False, showline=False,
                        ),
                        yaxis=dict(
                            range=[0, ymax],
                            showgrid=True, gridcolor="rgba(148,178,230,0.10)",
                            zeroline=False, showline=False,
                        ),
                        hoverlabel=dict(bgcolor="#0B1F3A", bordercolor=PRIMARY_HOV,
                                        font=dict(color="#fff", family="Inter")),
                        transition=dict(duration=500, easing="cubic-in-out"),
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

        with right:
            with card("Daily Forecast",
                      subtitle="Day-by-day breakdown",
                      icon_name="calendar"):
                for i, val in enumerate(fc, start=1):
                    cat = get_aqi_category(val)
                    clr = aqi_color(val)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;justify-content:space-between;
                                padding:9px 14px;margin-bottom:7px;border-radius:12px;
                                background:{ROW_BG};
                                border:1px solid {ROW_BORDER};
                                box-shadow:0 1px 2px rgba(15,31,58,0.04);
                                transition:transform 0.15s ease, border-color 0.2s ease;">
                        <div style="min-width:0;display:flex;align-items:baseline;gap:10px;">
                            <div style="font-size:0.65rem;color:{MUTED};font-weight:700;
                                        letter-spacing:0.1em;text-transform:uppercase;">Day {i}</div>
                            <div style="font-size:1.35rem;font-weight:800;color:{clr};
                                        line-height:1;letter-spacing:-0.025em;">
                                {val:.0f}
                            </div>
                            <div style="font-size:0.62rem;color:{MUTED};
                                        font-weight:600;letter-spacing:0.08em;">AQI</div>
                        </div>
                        <div style="display:inline-flex;align-items:center;gap:6px;
                                    padding:5px 10px;border-radius:999px;
                                    background:{clr}1F;border:1px solid {clr}66;
                                    color:{clr};font-size:0.72rem;font-weight:700;
                                    white-space:nowrap;flex-shrink:0;">
                            <span style="width:6px;height:6px;border-radius:50%;
                                         background:{clr};box-shadow:0 0 0 3px {clr}33;"></span>
                            {cat}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Bottom KPI strip ──────────────────────────────────────────
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        trend = "Worsening" if fc[-1] > fc[0] else "Improving"
        trend_clr = "#ef4444" if trend == "Worsening" else "#22c55e"
        avg   = round(sum(fc)/len(fc), 1)
        avg_c = get_aqi_category(avg)
        avg_clr = aqi_color(avg)
        peak  = max(fc); peak_day = fc.index(peak) + 1
        peak_clr = aqi_color(peak)

        b1, b2, b3 = st.columns(3)
        for col, ic, val, lbl, sub, clr in [
            (b1, "activity", f"{avg}",        "AVERAGE AQI", avg_c,           avg_clr),
            (b2, "trending", trend,           "TREND",        f"Day 1 → Day {len(fc)}", trend_clr),
            (b3, "shield",   f"{peak:.0f}",   "PEAK AQI",     f"Day {peak_day}",         peak_clr),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi">
                    <div class="kpi-head">
                        <div class="kpi-icon" style="color:{clr};border-color:{clr}55;
                                    background:linear-gradient(135deg, {clr}22, {clr}11);">
                            {icon(ic, 18, clr, 2)}
                        </div>
                        <div style="font-size:0.65rem;color:{MUTED};font-weight:700;
                                    letter-spacing:0.08em;text-transform:uppercase;">{sub}</div>
                    </div>
                    <div class="kpi-val" style="color:{clr};">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        with st.expander("Raw Forecast Table"):
            st.dataframe(pd.DataFrame({
                "Day":      days,
                "AQI":      fc,
                "Category": [get_aqi_category(a) for a in fc],
                "Risk":     [get_aqi_risk(a)     for a in fc],
            }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# DATA ANALYSIS
# ═══════════════════════════════════════════════════════
elif section == "Data Analysis":
    st.markdown(f'<p class="sec-sub">Explore patterns across cities, seasons, and pollutants</p>',
                unsafe_allow_html=True)
    with st.spinner("Loading dataset…"):
        df = load_df()

    if df is None:
        st.warning("Dataset file `aqi_india_work.csv` is not present in the project. "
                   "Drop the CSV alongside `app.py` to enable this page.")
    else:
        # ── KPI tiles ──────────────────────────────────────────────────
        kpi_specs = [
            ("database", f"{len(df):,}",        "TOTAL RECORDS",  "#3B82F6"),
            ("map-pin",  f"{len(CITY_COLS)}",   "CITIES TRACKED", "#22c55e"),
            ("layers",   f"{df.shape[1]}",      "FEATURES",        "#f59e0b"),
            ("bar",      "5",                   "AQI CATEGORIES", "#a78bfa"),
        ]
        k1, k2, k3, k4 = st.columns(4)
        for col, (ic, val, lbl, clr) in zip([k1,k2,k3,k4], kpi_specs):
            with col:
                st.markdown(f"""
                <div class="kpi" style="border-color:rgba(148,178,230,0.10);">
                    <div style="display:flex;align-items:center;gap:14px;">
                        <div style="width:44px;height:44px;border-radius:12px;
                                    display:flex;align-items:center;justify-content:center;
                                    background:{clr}1f;border:1px solid {clr}55;">
                            {icon(ic, 22, clr, 2)}
                        </div>
                        <div>
                            <div class="kpi-val" style="color:{TEXT};font-size:1.9rem;line-height:1;">{val}</div>
                            <div class="kpi-label" style="margin-top:4px;">{lbl}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        # ── City AQI Ranking ──────────────────────────────────────────
        ca = city_avg(df).sort_values(ascending=False)
        ca_top = ca.head(15)

        with card("City AQI Ranking",
                  subtitle="Average AQI across major cities — sorted worst first",
                  icon_name="map-pin"):
            # Chart now spans the FULL width of the card (no more cramped 1.6:1 split)
            if PLOTLY_OK:
                clrs = [aqi_color(v) for v in ca_top.values]
                fig = go.Figure(go.Bar(
                    x=ca_top.values[::-1],
                    y=[c.title() for c in ca_top.index[::-1]],
                    orientation="h",
                    marker=dict(color=clrs[::-1],
                                line=dict(color="rgba(255,255,255,0.06)", width=1)),
                    text=[f"{v:.0f}" for v in ca_top.values[::-1]],
                    textposition="outside",
                    textfont=dict(color=PLT_TEXT, size=12, family="Inter"),
                    hovertemplate="<b>%{y}</b><br>AQI: %{x:.1f}<extra></extra>",
                ))
                fig.update_layout(
                    height=480, margin=dict(l=10, r=40, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=PLT_TEXT, family="Inter", size=12),
                    xaxis=dict(showgrid=True, gridcolor="rgba(148,178,230,0.10)",
                               zeroline=False, showline=False),
                    yaxis=dict(showgrid=False, zeroline=False, showline=False,
                               tickfont=dict(size=12)),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.bar_chart(ca_top)

            # Rankings now BELOW chart in a 3-column responsive grid (much more readable)
            st.markdown(
                f'<div style="margin:18px 0 8px 0;font-size:0.78rem;font-weight:700;'
                f'color:{MUTED};letter-spacing:0.1em;text-transform:uppercase;">'
                f'Detailed Rankings</div>',
                unsafe_allow_html=True,
            )
            rank_cols = st.columns(3)
            cities_list = list(ca_top.items())
            per_col = (len(cities_list) + 2) // 3
            for col_idx, col in enumerate(rank_cols):
                with col:
                    chunk = cities_list[col_idx * per_col:(col_idx + 1) * per_col]
                    for i, (city, val) in enumerate(chunk, start=col_idx * per_col + 1):
                        cat = get_aqi_category(val)
                        cp  = aqi_palette(cat)
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;justify-content:space-between;
                                    padding:11px 14px;margin-bottom:8px;border-radius:12px;
                                    background:{ROW_BG};
                                    border:1px solid {ROW_BORDER};
                                    box-shadow:0 1px 2px rgba(15,31,58,0.04);">
                            <div style="display:flex;align-items:center;gap:11px;min-width:0;">
                                <span style="display:inline-flex;align-items:center;justify-content:center;
                                             width:24px;height:24px;border-radius:8px;flex-shrink:0;
                                             background:{CHIP_BG};
                                             color:{PRIMARY_HOV};font-size:0.72rem;font-weight:800;">
                                    {i}
                                </span>
                                <span style="color:{TEXT};font-weight:600;font-size:0.88rem;
                                             white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                                    {city.title()}
                                </span>
                            </div>
                            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
                                <span style="color:{cp['hex']};font-weight:800;font-size:0.95rem;
                                             letter-spacing:-0.01em;">{val:.0f}</span>
                                <span style="font-size:0.66rem;font-weight:700;padding:3px 8px;
                                             border-radius:999px;background:{cp['bg']};
                                             color:{cp['hex']};border:1px solid {cp['border']};
                                             white-space:nowrap;">{cat}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

        # ── Seasonal + Monthly row ────────────────────────────────────
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
        s_l, s_r = st.columns(2)
        sa = season_avg(df)
        monthly = df.groupby("month")["us_aqi"].mean().sort_index()

        with s_l:
            with card("Seasonal Pattern",
                      subtitle="Average AQI by season",
                      icon_name="wind"):
                if PLOTLY_OK:
                    sclrs = [aqi_color(v) for v in sa.values]
                    fig = go.Figure(go.Bar(
                        x=sa.index, y=sa.values, marker=dict(color=sclrs),
                        text=[f"{v:.0f}" for v in sa.values], textposition="outside",
                        textfont=dict(color=PLT_TEXT, size=11),
                        hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>",
                    ))
                    fig.update_layout(
                        height=300, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=PLT_TEXT, family="Inter", size=11),
                        xaxis=dict(showgrid=False, zeroline=False, showline=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(148,178,230,0.10)",
                                   zeroline=False, showline=False),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

        with s_r:
            with card("Monthly Trend",
                      subtitle="Average AQI across the year",
                      icon_name="calendar"):
                if PLOTLY_OK:
                    mx = list(monthly.index.astype(int))
                    my = list(monthly.values)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=mx, y=my, mode="lines+markers",
                        line=dict(color=ACCENT_TO, width=3, shape="spline", smoothing=1.3),
                        marker=dict(size=7, color=ACCENT_TO,
                                    line=dict(color="#ffffff", width=2)),
                        fill="tozeroy",
                        fillcolor="rgba(59,130,246,0.12)",
                        hovertemplate="%{x}<br>AQI: %{y:.1f}<extra></extra>",
                    ))
                    fig.update_layout(
                        height=300, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=PLT_TEXT, family="Inter", size=11),
                        xaxis=dict(tickmode="array", tickvals=mx,
                                   ticktext=[MONTH_NAMES[m-1] for m in mx],
                                   showgrid=False, zeroline=False, showline=False),
                        yaxis=dict(showgrid=True, gridcolor="rgba(148,178,230,0.10)",
                                   zeroline=False, showline=False),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

        # ── Category Distribution + Risk Distribution ─────────────────
        c_l, c_r = st.columns(2)
        cc = cat_counts(df)
        order = [c for c in ["Good","Moderate","Unhealthy","Poor","Hazardous"] if c in cc.index]
        cc = cc.reindex(order)
        total_cc = float(cc.sum()) or 1.0

        with c_l:
            with card("Category Distribution",
                      subtitle="Share of records in each AQI band",
                      icon_name="pie"):
                d_chart, d_legend = st.columns([1, 1])
                with d_chart:
                    if PLOTLY_OK:
                        bclrs = [aqi_palette(c)["hex"] for c in cc.index]
                        fig = go.Figure(go.Pie(
                            labels=list(cc.index), values=list(cc.values),
                            hole=0.62,
                            marker=dict(colors=bclrs,
                                        line=dict(color=SURFACE_SOLID, width=3)),
                            textinfo="none",
                            hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
                        ))
                        fig.update_layout(
                            height=240, margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True,
                                        config={"displayModeBar": False})
                with d_legend:
                    for cat_name in cc.index:
                        pct = (cc[cat_name] / total_cc) * 100
                        cp  = aqi_palette(cat_name)
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;justify-content:space-between;
                                    padding:10px 13px;margin-bottom:7px;border-radius:11px;
                                    background:{ROW_BG};
                                    border:1px solid {ROW_BORDER};">
                            <div style="display:flex;align-items:center;gap:10px;">
                                <span style="width:10px;height:10px;border-radius:50%;
                                             background:{cp['hex']};display:inline-block;
                                             box-shadow:0 0 0 3px {cp['hex']}22;"></span>
                                <span style="color:{TEXT};font-size:0.86rem;font-weight:600;">{cat_name}</span>
                            </div>
                            <span style="color:{cp['hex']};font-weight:800;font-size:0.9rem;">{pct:.1f}%</span>
                        </div>""", unsafe_allow_html=True)

        with c_r:
            with card("Risk Distribution",
                      subtitle="Aggregate health risk across all records",
                      icon_name="shield"):
                risk_s = df["us_aqi"].apply(get_aqi_risk).value_counts()
                risk_total = float(risk_s.sum()) or 1.0
                risk_specs = [
                    ("LOW RISK",    "LOW",    "#22c55e"),
                    ("MEDIUM RISK", "MEDIUM", "#eab308"),
                    ("HIGH RISK",   "HIGH",   "#ef4444"),
                ]
                for lbl, key, clr in risk_specs:
                    cnt = int(risk_s.get(key, 0))
                    pct = (cnt / risk_total) * 100
                    st.markdown(f"""
                    <div style="margin-bottom:20px;">
                        <div style="display:flex;justify-content:space-between;align-items:baseline;
                                    margin-bottom:9px;">
                            <span style="color:{clr};font-weight:800;font-size:0.86rem;
                                         letter-spacing:0.08em;">{lbl}</span>
                            <span style="color:{TEXT};font-weight:700;font-size:0.92rem;">
                                {cnt:,} <span style="color:{MUTED};font-weight:500;">({pct:.1f}%)</span>
                            </span>
                        </div>
                        <div style="height:10px;border-radius:999px;
                                    background:{BAR_TRACK};overflow:hidden;
                                    border:1px solid {ROW_BORDER};">
                            <div style="height:100%;width:{pct:.1f}%;border-radius:999px;
                                        background:linear-gradient(90deg,{clr}cc,{clr});
                                        box-shadow:0 0 14px {clr}66;
                                        transition:width 0.6s ease;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        # ── Pollutant Correlation (pill heatmap) ──────────────────────
        cm = corr_matrix(df)
        # Friendly labels
        nice = {
            "pm2_5_ugm3":"PM2.5", "pm10_ugm3":"PM10", "co_ugm3":"CO",
            "no2_ugm3":"NO\u2082", "so2_ugm3":"SO\u2082", "o3_ugm3":"O\u2083",
            "dust_ugm3":"DUST", "us_aqi":"AQI",
        }
        cols = list(cm.columns)
        labels = [nice.get(c, c.upper()) for c in cols]

        def corr_color(v):
            # Diverging palette: strong blue (positive) ↔ neutral ↔ warm red (negative).
            # White text on saturated tints, dark text on the neutral middle band so the
            # heatmap stays readable in BOTH light and dark mode.
            if v >=  0.85: return ("#1D4ED8", "#FFFFFF")  # very strong +
            if v >=  0.70: return ("#2563EB", "#FFFFFF")
            if v >=  0.55: return ("#3B82F6", "#FFFFFF")
            if v >=  0.40: return ("#60A5FA", "#0B1F3A")
            if v >=  0.25: return ("#93C5FD", "#0B1F3A")
            if v >=  0.10: return ("#BFDBFE", "#0B1F3A")
            if v >= -0.10: return ("#E5E7EB" if not DARK else "#334155",
                                   "#0B1F3A" if not DARK else "#E6EEFB")  # neutral
            if v >= -0.25: return ("#FECACA", "#0B1F3A")
            if v >= -0.40: return ("#FCA5A5", "#0B1F3A")
            if v >= -0.70: return ("#F87171", "#FFFFFF")
            return            ("#DC2626", "#FFFFFF")

        # Header row
        head_cells = "<th style='padding:10px 8px;'></th>" + "".join(
            f"<th style='color:{MUTED};font-weight:700;font-size:0.78rem;"
            f"padding:10px 8px;text-align:center;letter-spacing:0.04em;'>{lab}</th>"
            for lab in labels)

        body_rows = ""
        for i, row_lbl in enumerate(labels):
            cells = (f"<td style='color:{TEXT};font-weight:700;font-size:0.82rem;"
                     f"padding:8px 14px 8px 4px;text-align:right;'>{row_lbl}</td>")
            for j in range(len(labels)):
                v = float(cm.iat[i, j])
                bg, fg = corr_color(v)
                # Diagonal cells (self-correlation) get a subtle outline
                outline = (f"box-shadow:inset 0 0 0 2px {PRIMARY_HOV}55;" if i == j else "")
                cells += (f"<td style='padding:5px 6px;text-align:center;'>"
                          f"<div style='display:inline-block;min-width:60px;padding:8px 12px;"
                          f"border-radius:10px;background:{bg};color:{fg};{outline}"
                          f"font-weight:800;font-size:0.82rem;font-family:Inter;"
                          f"letter-spacing:-0.01em;'>"
                          f"{v:+.2f}</div></td>")
            body_rows += f"<tr>{cells}</tr>"

        # Build a clear legend strip so users can decode the colors at a glance
        legend_steps = [
            ("#DC2626", "−1.0"),
            ("#F87171", "−0.6"),
            ("#FCA5A5", "−0.2"),
            ("#E5E7EB" if not DARK else "#334155", "0"),
            ("#BFDBFE", "+0.2"),
            ("#60A5FA", "+0.6"),
            ("#1D4ED8", "+1.0"),
        ]
        legend_html = (
            f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;'
            f'margin-top:18px;padding:12px 14px;border-radius:12px;'
            f'background:{ROW_BG};border:1px solid {ROW_BORDER};">'
            f'<span style="color:{MUTED};font-size:0.74rem;font-weight:700;'
            f'letter-spacing:0.1em;text-transform:uppercase;">Correlation</span>'
            f'<div style="display:flex;align-items:center;gap:0;flex:1;min-width:240px;">'
        )
        for clr, _ in legend_steps:
            legend_html += (
                f'<div style="flex:1;height:14px;background:{clr};'
                f'border-top:1px solid {ROW_BORDER};border-bottom:1px solid {ROW_BORDER};"></div>'
            )
        legend_html += '</div><div style="display:flex;justify-content:space-between;width:100%;margin-top:-4px;">'
        for _, lbl in legend_steps:
            legend_html += (f'<span style="flex:1;text-align:center;color:{MUTED};'
                            f'font-size:0.7rem;font-weight:600;">{lbl}</span>')
        legend_html += '</div></div>'

        with card("Pollutant Correlation",
                  subtitle="Pearson correlation between pollutants and AQI — "
                           "blue = positive, red = negative, neutral = no relationship",
                  icon_name="chart"):
            st.markdown(
                f'<div style="overflow-x:auto;">'
                f'<table style="border-collapse:separate;border-spacing:0;width:100%;">'
                f'<thead><tr>{head_cells}</tr></thead>'
                f'<tbody>{body_rows}</tbody>'
                f'</table>'
                f'</div>'
                f'{legend_html}',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
elif section == "Model Performance":
    st.markdown(f'<p class="sec-sub">Compare regression models trained on the AQI dataset</p>',
                unsafe_allow_html=True)
    df = load_df()
    if df is None:
        st.warning("Dataset file `aqi_india_work.csv` is not present in the project.")
    else:
        with st.spinner("Training models…"):
            perf = model_perf(df)
        micons = {"Linear Regression":"activity",
                  "Decision Tree":"activity",
                  "Random Forest":"activity"}
        mclrs  = {"Linear Regression":"#3B82F6",
                  "Decision Tree":"#a78bfa",
                  "Random Forest":"#22c55e"}

        best = max(perf, key=lambda k: perf[k]["R²"])
        best_clr = mclrs[best]

        # ── Hero: Best Model card (theme-aware gradient) ──────────────
        st.markdown(f"""
        <div class="card" style="background:linear-gradient(135deg,
                    {best_clr}1A 0%, {SURFACE} 65%);
                    border:1px solid {best_clr}55;">
            <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
                <div style="width:54px;height:54px;border-radius:14px;
                            background:linear-gradient(135deg,{best_clr},{best_clr}aa);
                            display:flex;align-items:center;justify-content:center;
                            box-shadow:0 8px 24px -8px {best_clr}88;">
                    {icon("trophy", 26, "#ffffff", 2.2)}
                </div>
                <div style="flex:1;min-width:240px;">
                    <div style="color:{best_clr};font-size:0.72rem;font-weight:800;
                                letter-spacing:0.14em;">BEST MODEL</div>
                    <div style="color:{TEXT};font-size:1.6rem;font-weight:800;
                                margin-top:2px;">{best}</div>
                    <div style="color:{MUTED};font-size:0.86rem;margin-top:2px;">
                        Highest R² and lowest error across all evaluated models
                    </div>
                </div>
                <div style="display:flex;gap:30px;align-items:center;">
                    <div style="text-align:right;">
                        <div style="color:{MUTED};font-size:0.7rem;font-weight:700;
                                    letter-spacing:0.12em;">R² SCORE</div>
                        <div style="color:{best_clr};font-size:1.9rem;font-weight:800;
                                    line-height:1;margin-top:2px;">{perf[best]['R²']:.3f}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:{MUTED};font-size:0.7rem;font-weight:700;
                                    letter-spacing:0.12em;">RMSE</div>
                        <div style="color:{TEXT};font-size:1.9rem;font-weight:800;
                                    line-height:1;margin-top:2px;">{perf[best]['RMSE']}</div>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # ── Three model cards ────────────────────────────────────────
        mcols = st.columns(3)
        for col, (name, metrics) in zip(mcols, perf.items()):
            clr = mclrs.get(name, PRIMARY)
            ic  = micons.get(name, "chart")
            r2  = float(metrics["R²"])
            r2_pct = max(0.0, min(100.0, r2 * 100.0))
            is_best = (name == best)
            best_badge = (f'<span style="font-size:0.65rem;font-weight:800;letter-spacing:0.1em;'
                          f'padding:4px 10px;border-radius:999px;background:{clr}22;'
                          f'color:{clr};border:1px solid {clr}55;">BEST</span>'
                          if is_best else "")
            with col:
                card_html = (
                    f'<div class="card" style="border:1px solid {clr}55;'
                    f'background:linear-gradient(180deg, {clr}14 0%, {SURFACE} 100%);">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">'
                    f'<div style="display:flex;align-items:center;gap:10px;">'
                    f'<div style="width:34px;height:34px;border-radius:10px;background:{clr}22;'
                    f'border:1px solid {clr}55;display:flex;align-items:center;justify-content:center;">'
                    f'{icon(ic, 18, clr, 2)}</div>'
                    f'<div style="color:{TEXT};font-weight:700;font-size:1rem;">{name}</div>'
                    f'</div>{best_badge}</div>'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
                    f'<div style="display:flex;align-items:center;gap:6px;color:{MUTED};'
                    f'font-size:0.78rem;font-weight:600;">'
                    f'<span style="width:8px;height:8px;border-radius:50%;background:{clr};display:inline-block;"></span>'
                    f'R\u00b2 Score</div>'
                    f'<div style="color:{clr};font-weight:800;font-size:1.05rem;">{r2:.3f}</div>'
                    f'</div>'
                    f'<div style="height:9px;border-radius:999px;background:{BAR_TRACK};'
                    f'overflow:hidden;margin-bottom:18px;border:1px solid {ROW_BORDER};">'
                    f'<div style="height:100%;width:{r2_pct:.1f}%;border-radius:999px;'
                    f'background:linear-gradient(90deg,{clr}aa,{clr});box-shadow:0 0 12px {clr}88;'
                    f'transition:width 0.6s ease;"></div>'
                    f'</div>'
                    f'<div style="display:flex;justify-content:space-between;gap:14px;">'
                    f'<div><div style="display:flex;align-items:center;gap:6px;color:{MUTED};'
                    f'font-size:0.72rem;font-weight:600;letter-spacing:0.04em;">'
                    f'{icon("trending", 13, MUTED, 2)} RMSE</div>'
                    f'<div style="color:{TEXT};font-weight:800;font-size:1.25rem;margin-top:4px;">{metrics["RMSE"]}</div>'
                    f'</div>'
                    f'<div><div style="display:flex;align-items:center;gap:6px;color:{MUTED};'
                    f'font-size:0.72rem;font-weight:600;letter-spacing:0.04em;">'
                    f'{icon("scale", 13, MUTED, 2)} MAE</div>'
                    f'<div style="color:{TEXT};font-weight:800;font-size:1.25rem;margin-top:4px;">{metrics["MAE"]}</div>'
                    f'</div></div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        # ── R² Comparison + Error Comparison ─────────────────────────
        names = list(perf.keys())
        cclrs = [mclrs[n] for n in names]
        r2s   = [float(perf[n]["R²"])  for n in names]
        rmses = [float(perf[n]["RMSE"]) for n in names]
        maes  = [float(perf[n]["MAE"])  for n in names]

        cmp_l, cmp_r = st.columns(2)

        with cmp_l:
            with card("R² Score Comparison",
                      subtitle="Higher is better — fits between 0 and 1",
                      icon_name="chart"):
                if PLOTLY_OK:
                    fig = go.Figure(go.Bar(
                        x=names, y=r2s, marker=dict(color=cclrs),
                        text=[f"{v:.3f}" for v in r2s], textposition="outside",
                        textfont=dict(color=PLT_TEXT, size=11),
                        hovertemplate="<b>%{x}</b><br>R² = %{y:.3f}<extra></extra>",
                    ))
                    fig.update_layout(
                        height=320, margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=PLT_TEXT, family="Inter", size=11),
                        xaxis=dict(showgrid=False, zeroline=False, showline=False),
                        yaxis=dict(range=[0,1.05], showgrid=True,
                                   gridcolor="rgba(148,178,230,0.10)",
                                   zeroline=False, showline=False),
                        showlegend=False, bargap=0.45,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

        with cmp_r:
            with card("Error Comparison",
                      subtitle="Lower is better — RMSE penalizes large errors",
                      icon_name="trending"):
                if PLOTLY_OK:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="RMSE", x=names, y=rmses,
                        marker=dict(color=cclrs),
                        hovertemplate="<b>%{x}</b><br>RMSE = %{y:.2f}<extra></extra>",
                    ))
                    def _hex_to_rgba(h, a=0.6):
                        h = h.lstrip("#")
                        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
                    fig.add_trace(go.Bar(
                        name="MAE", x=names, y=maes,
                        marker=dict(color=[_hex_to_rgba(c, 0.6) for c in cclrs],
                                    line=dict(color=cclrs, width=0)),
                        hovertemplate="<b>%{x}</b><br>MAE = %{y:.2f}<extra></extra>",
                    ))
                    fig.update_layout(
                        height=320, margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=PLT_TEXT, family="Inter", size=11),
                        xaxis=dict(showgrid=False, zeroline=False, showline=False),
                        yaxis=dict(showgrid=True,
                                   gridcolor="rgba(148,178,230,0.10)",
                                   zeroline=False, showline=False),
                        barmode="group", bargap=0.30, bargroupgap=0.10,
                        legend=dict(orientation="h", x=0.5, y=-0.18,
                                    xanchor="center", bgcolor="rgba(0,0,0,0)",
                                    font=dict(color=PLT_TEXT, size=11)),
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════
# REAL-TIME AQI  (with globe-style visualization)
# ═══════════════════════════════════════════════════════
elif section == "Real-Time AQI":
    st.markdown(f'<p class="sec-sub">Live AQI data from WAQI API — any city worldwide.</p>',
                unsafe_allow_html=True)

    col_in, col_btn = st.columns([3,1])
    with col_in:
        rt_city = st.text_input("City name",
                                placeholder="Delhi · Mumbai · London · Beijing · New York",
                                key="rt_city")
    with col_btn:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        fetch = st.button("Fetch Live AQI", type="primary", use_container_width=True)

    def render_globe_card(fig, title="3D Earth"):
        with card(title, icon_name="globe"):
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False, "scrollZoom": True})
                st.caption("Drag to rotate the Earth · Scroll to zoom")
            else:
                st.info("Install `plotly` to enable the 3D Earth view: `pip install plotly`")

    # Default globe (no fetch yet) — show the Earth with no marker
    if not fetch:
        fig = build_earth_globe()
        render_globe_card(fig, "3D Earth · Awaiting Input")
        st.caption("Enter any city above to fetch a live AQI reading and "
                   "spin the globe to that location.")

    elif fetch and rt_city.strip():
        with st.spinner(f"Fetching live data for **{rt_city}**…"):
            aqi_v, city_n = get_real_aqi(rt_city.strip().lower())

        if aqi_v:
            cat  = get_aqi_category(aqi_v)
            risk = get_aqi_risk(aqi_v)
            p    = aqi_palette(cat)
            clr  = p["hex"]

            # Geocode → lat/lon for the marker
            geo = geocode_city(city_n) or geocode_city(rt_city)
            if geo:
                lat, lon, pretty = geo
                pretty_name = city_n or pretty
            else:
                lat, lon, pretty_name = None, None, city_n

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            left, right = st.columns([1.4, 1])
            with left:
                fig = build_earth_globe(lat=lat, lon=lon, label=pretty_name,
                                         aqi_v=aqi_v, color=clr)
                render_globe_card(fig, f"3D Earth · {pretty_name}")
                st.session_state.ctx_city = rt_city.strip().lower()
                st.session_state.ctx_aqi  = aqi_v

            with right:
                advice = {
                    "Good":      ("Excellent air quality.", "Enjoy outdoor activities freely."),
                    "Moderate":  ("Acceptable air quality.", "Sensitive individuals should limit outdoor exertion."),
                    "Unhealthy": ("Unhealthy for all groups.", "Wear a mask outdoors. Limit time outside."),
                    "Poor":      ("Serious health risk.", "Stay indoors, keep windows closed, use N95 mask."),
                    "Hazardous": ("Health emergency.", "Do not go outside. Seek medical help if symptoms appear."),
                }
                headline, detail = advice.get(cat, ("—","—"))

                st.markdown(f"""
                <div class="aqi-result" style="background:{p['bg']};border-color:{p['border']};margin-bottom:14px;">
                    <div style='font-size:0.75rem;color:{clr};opacity:0.85;letter-spacing:0.06em;'>{pretty_name}</div>
                    <div class="aqi-big" style="color:{clr};">{aqi_v}</div>
                    <div class="aqi-cat"  style="color:{clr};">{cat}</div>
                    <div class="aqi-risk" style="color:{clr};opacity:0.85;">{risk} RISK</div>
                </div>
                <div class="card">
                    <div class="card-title"><span class="ico">{icon("shield",18,PRIMARY_HOV,2)}</span> Health Advisory</div>
                    <div style='font-size:1.1rem;font-weight:700;color:{clr};margin-bottom:8px;'>{headline}</div>
                    <div style='color:{MUTED};font-size:0.92rem;line-height:1.6;'>{detail}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.error(f"Could not fetch AQI for **{rt_city}**. Try delhi, mumbai, london, beijing.")
    elif fetch:
        st.warning("Please enter a city name.")
