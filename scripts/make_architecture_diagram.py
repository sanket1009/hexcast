"""Generate docs/images/architecture.svg, the Hexcast architecture diagram.

Run from the repo root:
    python scripts/make_architecture_diagram.py

Top to bottom so it stays readable at GitHub's README width (about 1000 px).
Style: white background, IBM Plex Sans, earth palette, sections as soft
tints of their color with white cards inside. GitHub shows SVGs as images,
which cannot load web fonts, so the two Plex weights in scripts/assets/fonts/
are embedded in the file (SIL Open Font License).
"""

import base64
from pathlib import Path
from xml.sax.saxutils import escape

OUT = "docs/images/architecture.svg"
FONT_DIR = Path(__file__).parent / "assets" / "fonts"
W, H = 1200, 1830
TITLE, BODY, LINE = 17, 15, 20
FONT = "'IBM Plex Sans', Helvetica, Arial, sans-serif"


def shade(color, f):
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{max(0, min(255, round(v * f))):02X}" for v in (r, g, b))


def mix(c1, c2, t):
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b, strict=True)
    )


BG = "#FFFFFF"  # white
RULE, INK, MUTED = shade(BG, 0.84), "#1A1A1A", "#5C5C58"
ARROW = "#8A8A85"
C = {  # section color; cards and tints derive from it
    "src": INK,  # data sources
    "flow": "#A63D2A",  # weekly pipeline and spatial: terracotta
    "ml": "#33577A",  # forecasting: slate blue
    "llm": "#5B7031",  # RAG and LLM: olive
    "eval": "#A87A2A",  # evals and tracking: ochre
    "store": INK,  # storage
    "cloud": INK,  # serving
    "ci": MUTED,  # CI/CD
}
out = []


def font_css():
    faces = []
    for weight in (400, 700):
        path = FONT_DIR / f"ibm-plex-sans-latin-{weight}-normal.woff2"
        data = base64.b64encode(path.read_bytes()).decode()
        faces.append(
            "@font-face{font-family:'IBM Plex Sans';font-style:normal;"
            f"font-weight:{weight};src:url(data:font/woff2;base64,{data}) format('woff2');}}"
        )
    return f"<style>{''.join(faces)}</style>"


def text(x, y, s, size=BODY, weight="normal", color=INK, anchor="start", letter=0):
    ls = f' letter-spacing="{letter}"' if letter else ""
    out.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{ls}>{escape(s)}</text>'
    )


def panel(x, y, w, h, title, key, note=None):
    """A section: soft tint of its color, no border, label in the color."""
    color = C[key]
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{mix(BG, color, 0.07)}"/>'
    )
    text(x + 16, y + 28, title.upper(), 13.5, "bold", color, letter=1.2)
    if note:
        text(x + w - 16, y + 28, note, 13.5, "normal", MUTED, "end")


def card(x, y, w, h, title, lines, key):
    stroke = C[key]
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#FFFFFF" '
        f'stroke="{RULE}" stroke-width="1"/>'
    )
    text(x + 16, y + 27, title, TITLE, "bold", stroke)
    for i, line in enumerate(lines):
        text(x + 16, y + 50 + i * LINE, line, BODY, "normal", MUTED)


def cylinder(x, y, w, h, title, lines, key):
    ry = 16
    out.append(
        f'<path d="M{x},{y + ry} a{w / 2},{ry} 0 0 0 {w},0 v{h - 2 * ry} '
        f'a{w / 2},{ry} 0 0 1 -{w},0 z" fill="#FFFFFF" stroke="{INK}" stroke-width="1.2"/>'
    )
    out.append(
        f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="#FFFFFF" '
        f'stroke="{INK}" stroke-width="1.2"/>'
    )
    block = 24 + len(lines) * LINE  # title line + body lines
    top = y + 2 * ry + (h - 2 * ry - block) / 2 + 16  # first baseline
    text(x + w / 2, top, title, 18, "bold", INK, "middle")
    for i, line in enumerate(lines):
        text(x + w / 2, top + 26 + i * LINE, line, BODY, "normal", MUTED, "middle")


def arrow(points, label=None, at=None, dashed=False):
    d = "M" + " L".join(f"{px},{py}" for px, py in points)
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    out.append(
        f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="2"{dash} '
        f'marker-end="url(#arr)"/>'
    )
    if label:
        lx, ly = at
        wl = len(label) * 7.2 + 14
        out.append(
            f'<rect x="{lx - wl / 2}" y="{ly - 14}" width="{wl}" height="21" rx="4" '
            f'fill="{BG}" stroke="{RULE}"/>'
        )
        text(lx, ly + 2, label, 13, "normal", MUTED, "middle")


# ------------------------------------------------------------------ canvas
out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
)
out.append(
    f"<defs>{font_css()}"
    '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{ARROW}"/></marker></defs>'
)
out.append(
    f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6" fill="{BG}" stroke="{RULE}"/>'
)
text(40, 58, "Hexcast architecture", 32, "bold")
text(
    40,
    90,
    "Weekly pipeline for 2,498 Los Angeles County census tracts, 105 weeks (June 2024 to June 2026)",
    16,
    "normal",
    MUTED,
)

# ------------------------------------------------------------------ 1. sources
panel(40, 115, 1120, 270, "Data sources", "src")
sources = [
    ("EPA AQS", ["Daily PM2.5 from ~21 monitors", "LA and counties within 50 km"]),
    ("Landsat 8 and 9", ["Land surface temperature", "32-day composite, Earth Engine"]),
    ("OpenStreetMap", ["Geofabrik extract, pyosmium", "336,813-node drive network"]),
    ("CDPH", ["Emergency departments and beds", "LA County plus a 30 km buffer"]),
    ("Census Bureau", ["TIGER tracts, ACS 5-year,", "2020 blocks (91,626)"]),
    ("Benchmarks", ["CalEnviroScreen (index check)", "CDC PLACES (briefing context)"]),
]
for i, (title, lines) in enumerate(sources):
    card(56 + (i % 3) * 368, 161 + (i // 3) * 108, 352, 96, title, lines, "src")

arrow([(600, 365), (600, 423)], "weekly", (660, 400))

# ------------------------------------------------------------------ 2. Airflow
panel(40, 425, 1120, 540, "Apache Airflow 3.3 in Docker", "flow", "one week at a time")
card(
    56,
    473,
    300,
    130,
    "ingest_weekly",
    [
        "pull_pm25 from AQS",
        "pull_lst from Earth Engine",
        "reference refresh on demand",
        "triggers spatial_transform",
    ],
    "flow",
)
card(
    56,
    619,
    300,
    110,
    "backfill",
    [
        "105 weeks, oldest first",
        "waits for each week to finish",
        "guard: full 4-week PM2.5 window",
    ],
    "flow",
)
card(
    56,
    745,
    300,
    110,
    "Reference tables",
    [
        "ACS vulnerability, 7 shares",
        "(empirical Bayes shrinkage)",
        "H3 res 9 hexes, tract outlines",
    ],
    "flow",
)

panel(376, 465, 768, 350, "spatial_transform", "flow")
chips = [
    (
        "Kriging PM2.5",
        ["Ordinary Kriging, ~21 monitors", "LOO error weekly; 4-week mean"],
    ),
    (
        "Land surface temperature",
        ["Water masked per tract; cloudy", "weeks reuse the last good week"],
    ),
    (
        "E2SFCA emergency care access",
        ["30-minute Gaussian catchment", "30 km cross-county buffer"],
    ),
    (
        "Compound score and classes",
        ["Signed z: heat, PM2.5, vulnerability", "raise burden; access lowers it"],
    ),
    (
        "Gi* hot spots, chronic burden",
        ["Two-tailed 95% and 99%", "weeks CRITICAL in the last 52"],
    ),
    (
        "Confidence",
        ["200 draws of correlated Kriging", "error plus ±10% on the weights"],
    ),
]
for i, (title, lines) in enumerate(chips):
    card(392 + (i % 2) * 376, 511 + (i // 2) * 98, 360, 86, title, lines, "flow")

arrow([(356, 538), (390, 538)])
arrow([(356, 660), (390, 660)])
arrow([(356, 770), (390, 770)])
arrow([(572, 793), (572, 841)])
arrow([(948, 793), (948, 841)])
card(
    376,
    843,
    376,
    106,
    "alert_dispatch",
    ["Weekly email digest of CRITICAL", "tracts and their drivers", "No model calls"],
    "flow",
)
card(
    768,
    843,
    376,
    106,
    "model_retrain",
    [
        "Calibrated P(CRITICAL within 4 weeks)",
        "per-feature reasons, 8 drift checks",
        "promoted only if it beats 2 rules",
    ],
    "ml",
)

# ------------------------------------------------------------------ 3. storage, RAG, evals
arrow([(196, 855), (196, 1003)], "writes", (250, 984))
cylinder(
    56,
    1005,
    280,
    220,
    "DuckDB warehouse",
    ["25 tables", "per week and city", "reruns replace a week", "in one transaction"],
    "store",
)

panel(416, 1005, 430, 360, "RAG policy briefings", "llm", "run by hand")
rag = [
    ("Corpus: 12 policy PDFs", ["county and state heat, smoke, air, transport"]),
    ("Retrieve: ChromaDB + Qwen3 embeddings", ["3 excerpts per driver, cutoff 0.48"]),
    ("Generate: Claude Haiku 5.5", ["typed output: summary, risks, actions"]),
    ("Ground: citation checks", ["uncited actions dropped; urgency from class"]),
]
for i, (title, lines) in enumerate(rag):
    card(432, 1051 + i * 78, 398, 66, title, lines, "llm")
    if i < 3:
        arrow([(631, 1117 + i * 78), (631, 1127 + i * 78)])

arrow([(336, 1085), (430, 1085)], "inputs", (383, 1064))
arrow([(432, 1165), (338, 1165)], "briefings", (383, 1144))

arrow([(1040, 949), (1040, 1003)], dashed=True)
card(
    886,
    1005,
    274,
    130,
    "MLflow",
    ["model comparisons", "retrain and eval runs", "git SHA on every run"],
    "eval",
)
card(
    886,
    1155,
    274,
    210,
    "Eval suite A1 to A9",
    [
        "retrieval gold set,",
        "faithfulness, rules, specificity,",
        "LLM judge vs human labels,",
        "robustness, CI gate",
    ],
    "eval",
)
arrow([(1040, 1155), (1040, 1137)], dashed=True)

# ------------------------------------------------------------------ 4. Google Cloud
panel(40, 1405, 1120, 150, "Google Cloud", "cloud", "us-central1")
serving = [
    ("BigQuery", ["copy of every table,", "partitioned by week"]),
    (
        "FastAPI, private",
        ["13 endpoints on Cloud Run", "only the dashboard may call it"],
    ),
    (
        "Streamlit + pydeck, public",
        ["H3 map, watch list, validation", "light and dark themes"],
    ),
]
for i, (title, lines) in enumerate(serving):
    card(56 + i * 376, 1449, 336, 92, title, lines, "cloud")
arrow([(196, 1225), (196, 1447)], "sync after each run", (196, 1320))
arrow([(392, 1495), (430, 1495)])
arrow([(768, 1495), (806, 1495)])

# ------------------------------------------------------------------ 5. GitHub Actions
panel(40, 1595, 1120, 150, "GitHub Actions", "ci")
ci = [
    ("CI on every push and PR", ["ruff, 281 tests, eval regression gate"]),
    ("Deploy on merge to main", ["Workload Identity Federation, no keys"]),
    ("Visitors", ["public dashboard, private API behind it"]),
]
for i, (title, lines) in enumerate(ci):
    card(56 + i * 368, 1641, 352, 88, title, lines, "ci")
arrow([(600, 1639), (600, 1543)], "deploy", (660, 1578))

# ------------------------------------------------------------------ legend
legend = [
    ("flow", "Weekly pipeline and spatial"),
    ("ml", "Forecasting"),
    ("llm", "RAG and LLM"),
    ("eval", "Evals and tracking"),
    ("store", "Sources, storage, serving"),
]
lx = 40
for key, label in legend:
    out.append(
        f'<rect x="{lx}" y="1775" width="15" height="15" rx="2" fill="{C[key]}"/>'
    )
    text(lx + 22, 1788, label, 14, "normal", MUTED)
    lx += 44 + len(label) * 7.3
text(
    40,
    1814,
    "Dashed arrows: logged to MLflow, not stored in the warehouse.",
    13,
    "normal",
    MUTED,
)

out.append("</svg>")
with open(OUT, "w") as f:
    f.write("\n".join(out))
print("written")
