"""Small CSS layer for what the theme config cannot set.

Follows Apple's interface guidance as translated for the web: size-specific
tracking and leading, instant press feedback, a translucent top bar that
content scrolls under, and fallbacks for reduced motion and reduced
transparency. Kept to stable data-testid selectors. Fills use Apple's
translucent grey (rgba 120, 120, 128) and the CSS system color Canvas, so the
same rules work in light and dark (Streamlit sets color-scheme on .stApp).
"""

import streamlit as st

CSS = """
<style>
/* Type: tighter tracking and leading as size grows, body left alone */
h1 {
    font-size: clamp(2.75rem, 1.6rem + 2.4vw, 4.5rem) !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em;
    line-height: 1.02;
}
h2 {
    font-size: clamp(1.5rem, 0.95rem + 0.6vw, 2rem) !important;
    text-wrap: balance;
    letter-spacing: -0.022em;
    line-height: 1.12;
}
h3, h4 { letter-spacing: -0.012em; line-height: 1.25; }
[data-testid="stCaptionContainer"] { letter-spacing: 0.003em; }

/* KPI tiles: a filled grey surface instead of an outline */
[data-testid="stMetric"] {
    background: rgba(120, 120, 128, 0.08);
    border: none !important;
    border-radius: 18px;
    padding: 0.85rem 1.1rem 0.95rem;
}
[data-testid="stMetricValue"] { letter-spacing: -0.02em; line-height: 1.1; }

/* Rounded map and tables */
[data-testid="stDeckGlJsonChart"],
[data-testid="stDeckGlJsonChart"] > div { border-radius: 16px; overflow: hidden; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

/* Top bar as a translucent material: content scrolls under it */
[data-testid="stHeader"] {
    background: color-mix(in srgb, Canvas 72%, transparent) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    backdrop-filter: blur(20px) saturate(180%);
}

/* Press feedback: instant, subtle, on press rather than release */
@media (prefers-reduced-motion: no-preference) {
    button, [data-testid="stTopNavLink"] {
        transition: transform 120ms cubic-bezier(0.23, 1, 0.32, 1);
    }
    button:active, [data-testid="stTopNavLink"]:active { transform: scale(0.97); }
}

@media (prefers-reduced-transparency: reduce) {
    [data-testid="stHeader"] {
        background: Canvas !important;
        -webkit-backdrop-filter: none;
        backdrop-filter: none;
    }
}
</style>
"""


def apply() -> None:
    st.html(CSS)
