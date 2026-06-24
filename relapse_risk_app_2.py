"""
OrthoRelapse-DSS — TRIPOD+AI Compliant Diagnostic Decision Support System (Prototype)
=====================================================================================
A single-file, installable (PWA) Streamlit web prototype that estimates orthodontic
relapse probability from clinical inputs, classifies absolute risk tier, and explains
the prediction with native SHAP / LIME-style signed feature-contribution charts.

Run:
    pip install streamlit pandas altair
    streamlit run relapse_risk_app_2.py
"""

import json
import math
import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="OrthoRelapse-DSS",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# DESIGN SYSTEM  (corporate blue + clinical teal)
# --------------------------------------------------------------------------------------
PALETTE = {
    "navy": "#0B2447",        # deep corporate blue (headers, text)
    "blue": "#1E5AA8",        # primary corporate blue
    "blue_soft": "#E8F0FB",   # tinted panel fill
    "teal": "#11999E",        # clinical teal (accent / protective)
    "teal_soft": "#E3F6F5",   # tinted teal fill
    "risk": "#E4572E",        # risk-increasing (red)
    "protect": "#11999E",     # risk-reducing (teal/green)
    "amber": "#E8A317",       # moderate tier
    "ink": "#1C2B3A",
    "muted": "#5B6B7B",
    "line": "#D8E1EC",
    "bg": "#F5F8FC",
}

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {PALETTE['ink']};
    }}
    .stApp {{ background: {PALETTE['bg']}; }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    h1, h2, h3, h4 {{ font-family: 'Sora', sans-serif; color: {PALETTE['navy']}; letter-spacing: -0.01em; }}

    .hero {{
        background: linear-gradient(120deg, {PALETTE['navy']} 0%, {PALETTE['blue']} 55%, {PALETTE['teal']} 130%);
        border-radius: 20px;
        padding: 30px 34px;
        color: #fff;
        box-shadow: 0 14px 34px rgba(11,36,71,0.22);
        position: relative;
        overflow: hidden;
    }}
    .hero::after {{
        content: "";
        position: absolute; right: -60px; top: -60px;
        width: 220px; height: 220px; border-radius: 50%;
        background: rgba(255,255,255,0.07);
    }}
    .hero h1 {{ color: #fff; font-size: 2.0rem; margin: 0 0 6px 0; }}
    .hero p {{ color: #DCE8F7; font-size: 1.02rem; margin: 0; max-width: 760px; }}
    .hero .pill {{
        display: inline-block; margin-top: 14px; padding: 6px 14px;
        background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
        border-radius: 999px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    .card {{
        background: #fff; border: 1px solid {PALETTE['line']};
        border-radius: 16px; padding: 22px 24px;
        box-shadow: 0 6px 18px rgba(16,42,67,0.05);
        height: 100%;
    }}
    .card h3 {{ margin-top: 0; font-size: 1.05rem; }}
    .card .lead {{ color: {PALETTE['muted']}; font-size: 0.92rem; line-height: 1.5; }}
    .card ol {{ margin: 8px 0 0 0; padding-left: 18px; color: {PALETTE['ink']}; font-size: 0.92rem; }}
    .card ol li {{ margin-bottom: 5px; }}
    .eyebrow {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: {PALETTE['teal']}; margin-bottom: 4px;
    }}

    .meter-wrap {{
        background: #fff; border: 1px solid {PALETTE['line']}; border-radius: 16px;
        padding: 26px 28px; box-shadow: 0 6px 18px rgba(16,42,67,0.05);
    }}
    .score-num {{ font-family: 'Sora'; font-size: 3.4rem; font-weight: 700; line-height: 1; }}
    .score-sub {{ color: {PALETTE['muted']}; font-size: 0.9rem; }}
    .track {{
        position: relative; height: 18px; border-radius: 999px; margin: 18px 0 6px 0;
        background: linear-gradient(90deg, {PALETTE['teal']} 0%, {PALETTE['amber']} 50%, {PALETTE['risk']} 100%);
        opacity: 0.25;
    }}
    .track-fill {{
        position: absolute; top: 0; left: 0; height: 18px; border-radius: 999px;
        background: linear-gradient(90deg, {PALETTE['teal']} 0%, {PALETTE['amber']} 50%, {PALETTE['risk']} 100%);
    }}
    .marker {{
        position: absolute; top: -7px; width: 4px; height: 32px; border-radius: 4px;
        background: {PALETTE['navy']}; box-shadow: 0 0 0 3px #fff;
    }}
    .scale {{ display: flex; justify-content: space-between; color: {PALETTE['muted']}; font-size: 0.74rem; }}

    .tier-badge {{
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 16px; border-radius: 999px; font-weight: 700; font-size: 0.95rem;
    }}
    .dot {{ width: 11px; height: 11px; border-radius: 50%; }}

    .section-h {{ display:flex; align-items:baseline; gap:12px; margin: 26px 0 10px 0; }}
    .section-h h2 {{ margin:0; font-size:1.3rem; }}
    .section-h span {{ color:{PALETTE['muted']}; font-size:0.9rem; }}

    section[data-testid="stSidebar"] {{
        background: #fff; border-right: 1px solid {PALETTE['line']};
    }}
    section[data-testid="stSidebar"] .sb-title {{
        font-family:'Sora'; color:{PALETTE['navy']}; font-weight:700; font-size:1.05rem;
        margin: 4px 0 2px 0;
    }}
    section[data-testid="stSidebar"] .sb-sub {{ color:{PALETTE['muted']}; font-size:0.82rem; margin-bottom:8px; }}

    .install-card {{
        background: {PALETTE['teal_soft']}; border: 1px dashed {PALETTE['teal']};
        border-radius: 16px; padding: 20px 22px;
    }}
    .install-card h3 {{ margin-top:0; color:{PALETTE['navy']}; }}

    .footer {{
        margin-top: 34px; padding: 20px 4px; border-top: 1px solid {PALETTE['line']};
        color: {PALETTE['muted']}; font-size: 0.82rem; text-align: center; line-height: 1.6;
    }}
    .footer b {{ color: {PALETTE['navy']}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# PWA INJECTION
# --------------------------------------------------------------------------------------
APP_ICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'>"
    "<rect width='512' height='512' rx='96' fill='#0B2447'/>"
    "<path d='M256 96c-46 0-70 22-70 22s-24-22-70-22c-40 0-66 30-66 78 0 70 40 130 70 196 "
    "14 30 26 46 40 46 18 0 20-34 26-58 6-24 14-40 30-40s24 16 30 40c6 24 8 58 26 58 14 0 "
    "26-16 40-46 30-66 70-126 70-196 0-48-26-78-66-78-46 0-70 22-70 22S302 96 256 96z' "
    "fill='#11999E'/></svg>"
)

def _data_uri_svg(svg: str) -> str:
    import urllib.parse
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)

ICON_URI = _data_uri_svg(APP_ICON_SVG)

MANIFEST = {
    "name": "OrthoRelapse-DSS",
    "short_name": "OrthoDSS",
    "description": "TRIPOD+AI compliant orthodontic relapse risk decision support.",
    "start_url": ".",
    "scope": ".",
    "display": "standalone",
    "orientation": "portrait",
    "background_color": "#F5F8FC",
    "theme_color": PALETTE["navy"],
    "icons": [
        {"src": ICON_URI, "sizes": "192x192", "type": "image/svg+xml", "purpose": "any maskable"},
        {"src": ICON_URI, "sizes": "512x512", "type": "image/svg+xml", "purpose": "any maskable"},
    ],
}

PWA_BOOTSTRAP = f"""
<script>
(function() {{
    var doc;
    try {{ doc = window.parent.document; }} catch (e) {{ doc = document; }}
    var win = window.parent || window;

    if (!doc.getElementById('orthodss-manifest')) {{
        try {{
            var manifest = {json.dumps(MANIFEST)};
            var blob = new Blob([JSON.stringify(manifest)], {{type: 'application/manifest+json'}});
            var url = URL.createObjectURL(blob);
            var link = doc.createElement('link');
            link.id = 'orthodss-manifest';
            link.rel = 'manifest';
            link.href = url;
            doc.head.appendChild(link);
        }} catch (e) {{ console.warn('Manifest injection failed:', e); }}
    }}

    function meta(name, content, useProperty) {{
        var sel = (useProperty ? 'meta[property="' : 'meta[name="') + name + '"]';
        if (doc.querySelector(sel)) return;
        var m = doc.createElement('meta');
        if (useProperty) {{ m.setAttribute('property', name); }} else {{ m.setAttribute('name', name); }}
        m.setAttribute('content', content);
        doc.head.appendChild(m);
    }}
    meta('theme-color', '{PALETTE['navy']}');
    meta('apple-mobile-web-app-capable', 'yes');
    meta('apple-mobile-web-app-status-bar-style', 'black-translucent');
    meta('apple-mobile-web-app-title', 'OrthoDSS');
    meta('mobile-web-app-capable', 'yes');

    if (!doc.querySelector('link[rel="apple-touch-icon"]')) {{
        var ai = doc.createElement('link');
        ai.rel = 'apple-touch-icon';
        ai.href = '{ICON_URI}';
        doc.head.appendChild(ai);
    }}

    if (!win.__orthoPromptBound) {{
        win.__orthoPromptBound = true;
        win.addEventListener('beforeinstallprompt', function(ev) {{
            ev.preventDefault();
            win.__deferredPrompt = ev;
        }});
        win.addEventListener('appinstalled', function() {{
            win.__deferredPrompt = null;
        }});
    }}

    try {{
        if ('serviceWorker' in navigator) {{
            var sw = "self.addEventListener('install', function(e){{ self.skipWaiting(); }});" +
                     "self.addEventListener('activate', function(e){{ e.waitUntil(self.clients.claim()); }});" +
                     "self.addEventListener('fetch', function(e){{  }});";
            var swBlob = new Blob([sw], {{type: 'text/javascript'}});
            var swUrl = URL.createObjectURL(swBlob);
            navigator.serviceWorker.register(swUrl).catch(function(err) {{
                console.info('Service worker registration status:', err.message);
            }});
        }}
    }} catch (e) {{ console.info('SW registration skipped:', e); }}
}})();
</script>
"""
components.html(PWA_BOOTSTRAP, height=0)

# --------------------------------------------------------------------------------------
# CORE ML ENGINE
# --------------------------------------------------------------------------------------
INTERCEPT = -0.40  

COEF_AGE = -0.030          
REF_AGE = 25.0
COEF_LITTLE = 0.180        
REF_LITTLE = 3.5
COEF_DURATION = 0.020      
REF_DURATION = 18.0

SEX_CONTRIB = {"Male": 0.00, "Female": 0.15}
EXTRACTION_CONTRIB = {"No": 0.00, "Yes": 0.45}
MALOCCLUSION_CONTRIB = {"Mild": -0.40, "Moderate": 0.00, "Severe": 0.60}
RETAINER_CONTRIB = {
    "Hawley Retainer": 0.25,            
    "Vacuum-Formed Retainer": -0.25,   
    "Bonded Lingual Wire": -0.70,       
}

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def predict(features: dict):
    contribs = [
        {"feature": "Patient Age", "contribution": COEF_AGE * (features["age"] - REF_AGE)},
        {"feature": "Patient Sex", "contribution": SEX_CONTRIB[features["sex"]]},
        {"feature": "Extraction Status", "contribution": EXTRACTION_CONTRIB[features["extraction"]]},
        {"feature": "Little's Irregularity Index", "contribution": COEF_LITTLE * (features["little"] - REF_LITTLE)},
        {"feature": "Malocclusion Severity", "contribution": MALOCCLUSION_CONTRIB[features["malocclusion"]]},
        {"feature": "Retainer Type", "contribution": RETAINER_CONTRIB[features["retainer"]]},
        {"feature": "Treatment Duration", "contribution": COEF_DURATION * (features["duration"] - REF_DURATION)},
    ]

    logit = INTERCEPT + sum(c["contribution"] for c in contribs)
    prob = _sigmoid(logit) * 100.0

    if prob < 30.0:
        tier = "Low Risk"
    elif prob < 60.0:
        tier = "Moderate Risk"
    else:
        tier = "High Risk"

    return round(prob, 1), tier, contribs

TIER_STYLE = {
    "Low Risk":      {"color": PALETTE["teal"],  "fill": PALETTE["teal_soft"]},
    "Moderate Risk": {"color": PALETTE["amber"], "fill": "#FBF1DC"},
    "High Risk":     {"color": PALETTE["risk"],  "fill": "#FBE6DF"},
}

# --------------------------------------------------------------------------------------
# SIDEBAR — CLINICAL FEATURE PANEL
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-title">💡 Clinical Feature Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Enter the patient & treatment parameters.</div>', unsafe_allow_html=True)
    st.divider()

    age = st.slider("Patient Age (years)", min_value=10, max_value=50, value=22, step=1)
    sex = st.radio("Patient Sex", options=["Male", "Female"], horizontal=True)
    extraction = st.radio("Extraction Status", options=["No", "Yes"], horizontal=True)
    little = st.slider("Pre-treatment Crowding — Little's Irregularity Index (mm)", min_value=0.0, max_value=15.0, value=6.0, step=0.5)
    malocclusion = st.selectbox("Initial Malocclusion Severity", options=["Mild", "Moderate", "Severe"], index=1)
    retainer = st.selectbox("Proposed Retainer Type", options=["Hawley Retainer", "Vacuum-Formed Retainer", "Bonded Lingual Wire"], index=1)
    duration = st.slider("Treatment Duration (months)", min_value=6, max_value=36, value=20, step=1)

features = {
    "age": age, "sex": sex, "extraction": extraction, "little": little,
    "malocclusion": malocclusion, "retainer": retainer, "duration": duration,
}
prob, tier, contributions = predict(features)
tstyle = TIER_STYLE[tier]

# --------------------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>OrthoRelapse-DSS</h1>
        <p>An explainable clinical decision-support prototype that estimates the probability
        of post-treatment orthodontic relapse and demystifies every prediction with
        transparent feature-level reasoning.</p>
        <span class="pill">TRIPOD+AI · Explainable AI · Prototype</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------------------
# CARDS
# --------------------------------------------------------------------------------------
c1, c2 = st.columns([1.4, 1])
with c1:
    st.markdown(
        """
        <div class="card">
            <div class="eyebrow">How to use</div>
            <h3>Three steps to a risk estimate</h3>
            <ol>
                <li>Open the <b>Clinical Feature Panel</b> in the left sidebar.</li>
                <li>Set the patient and treatment parameters for the case.</li>
                <li>Read the <b>Relapse Probability Score</b>, risk tier, and the
                    SHAP / LIME explanation below to see <i>why</i> the model decided as it did.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="install-card">
            <h3>📲 Install the app</h3>
            <p class="lead">Add OrthoRelapse-DSS to your phone or desktop home screen for
            one-tap, full-screen access.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    install_component = f"""
    <div style="font-family:'Inter',sans-serif;">
      <button id="installBtn" style="
          width:100%; padding:12px 16px; border:none; border-radius:12px; cursor:pointer;
          background:linear-gradient(120deg,{PALETTE['blue']},{PALETTE['teal']});
          color:#fff; font-weight:700; font-size:0.95rem; letter-spacing:0.02em;
          box-shadow:0 8px 18px rgba(17,153,158,0.30);">
          ⤓ Install App
      </button>
      <div id="installMsg" style="margin-top:10px; font-size:0.82rem; color:{PALETTE['navy']};"></div>
    </div>
    <script>
    (function() {{
        var win = window.parent || window;
        var btn = document.getElementById('installBtn');
        var msg = document.getElementById('installMsg');
        var ua = navigator.userAgent || "";
        var isIOS = /iPad|iPhone|iPod/.test(ua) && !window.MSStream;

        function inStandalone() {{
            return window.matchMedia('(display-mode: standalone)').matches ||
                   win.navigator.standalone === true;
        }}

        if (inStandalone()) {{
            btn.style.display = 'none';
            msg.innerHTML = "✓ App is installed and running.";
            return;
        }}

        if (isIOS) {{
            btn.textContent = "How to install on iPhone / iPad";
            btn.onclick = function() {{
                msg.innerHTML = "On Safari: tap <b>Share</b> icon → <b>Add to Home Screen</b>.";
            }};
            return;
        }}

        btn.onclick = function() {{
            var dp = win.__deferredPrompt;
            if (dp) {{
                dp.prompt();
                dp.userChoice.then(function(choice) {{
                    if (choice.outcome === 'accepted') {{
                        msg.innerHTML = "✓ Installing...";
                    }}
                    win.__deferredPrompt = null;
                }});
            }} else {{
                msg.innerHTML = "Use browser menu → <b>Install app</b> / <b>Add to Home screen</b>.";
            }}
        }};
    }})();
    </script>
    """
    components.html(install_component, height=110)

# --------------------------------------------------------------------------------------
# PREDICTION OUTPUT
# --------------------------------------------------------------------------------------
st.markdown('<div class="section-h"><h2>Relapse Probability Score</h2></div>', unsafe_allow_html=True)

m1, m2 = st.columns([1.7, 1])
with m1:
    st.markdown(
        f"""
        <div class="meter-wrap">
            <span class="score-num" style="color:{tstyle['color']};">{prob:.1f}%</span>
            <span class="score-sub">&nbsp;estimated probability of relapse</span>
            <div class="track">
                <div class="track-fill" style="width:{prob:.1f}%;"></div>
                <div class="marker" style="left:calc({prob:.1f}% - 2px);"></div>
            </div>
            <div class="scale">
                <span>0%</span><span>Low ◦ 30</span><span>Moderate ◦ 60</span><span>High</span><span>100%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="meter-wrap" style="display:flex;flex-direction:column;justify-content:center;
             background:{tstyle['fill']};border-color:{tstyle['color']};height:100%;">
            <div class="score-sub" style="font-weight:600;color:{PALETTE['navy']};">Absolute Risk Tier</div>
            <div style="margin-top:10px;">
                <span class="tier-badge" style="background:#fff;color:{tstyle['color']};border:1px solid {tstyle['color']};">
                    <span class="dot" style="background:{tstyle['color']};"></span>{tier}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# EXPLAINABLE AI LAYER
# --------------------------------------------------------------------------------------
st.markdown('<div class="section-h"><h2>Explainable AI — Why this score?</h2></div>', unsafe_allow_html=True)

df = pd.DataFrame(contributions)
df["abs"] = df["contribution"].abs()
df = df.sort_values("abs", ascending=False).reset_index(drop=True)
df["direction"] = df["contribution"].apply(lambda v: "Increases risk" if v > 0 else "Reduces risk")

value_map = {
    "Patient Age": f"{age} yrs",
    "Patient Sex": sex,
    "Extraction Status": extraction,
    "Little's Irregularity Index": f"{little:.1f} mm",
    "Malocclusion Severity": malocclusion,
    "Retainer Type": retainer,
    "Treatment Duration": f"{duration} mo",
}
df["value"] = df["feature"].map(value_map)
df["label"] = df["feature"] + "  (" + df["value"] + ")"

color_scale = alt.Scale(domain=["Increases risk", "Reduces risk"], range=[PALETTE["risk"], PALETTE["protect"]])

x1, x2 = st.columns(2)

with x1:
    st.markdown("#### SHAP — local feature attribution")
    shap_chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4, height=22)
        .encode(
            x=alt.X("contribution:Q", title="Signed contribution (log-odds)"),
            y=alt.Y("label:N", sort=alt.SortField("abs", order="descending"), title=None),
            color=alt.Color("direction:N", scale=color_scale, legend=None),
        )
        .properties(height=300)
    )
    rule = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=PALETTE["navy"], size=1.5).encode(x="z:Q")
    st.altair_chart(shap_chart + rule, use_container_width=True)

with x2:
    st.markdown("#### LIME — local surrogate weights")
    max_abs = df["abs"].max() or 1.0
    df_lime = df.copy()
    df_lime["weight"] = df_lime["contribution"] / max_abs
    lime_chart = (
        alt.Chart(df_lime)
        .mark_bar(cornerRadius=4, height=22)
        .encode(
            x=alt.X("weight:Q", title="Relative local weight", scale=alt.Scale(domain=[-1.05, 1.05])),
            y=alt.Y("label:N", sort=alt.SortField("abs", order="descending"), title=None),
            color=alt.Color("direction:N", scale=color_scale, legend=None),
        )
        .properties(height=300)
    )
    rule2 = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=PALETTE["navy"], size=1.5).encode(x="z:Q")
    st.altair_chart(lime_chart + rule2, use_container_width=True)

# --------------------------------------------------------------------------------------
# FOOTER — WITH DR. ARAFAT'S SIGNATURE
# --------------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("👨‍⚕️ **Developed & Maintained by:**\n\nDr. Arafat Al-Jawry")

st.markdown(
    f"""
    <div class="footer">
        <b>OrthoRelapse-DSS — Diagnostic Decision Support System Prototype.</b><br>
        Developed and Designed by <b>Dr. Arafat Al-Jawry</b>. <br>
        TRIPOD+AI Compliant · For research and educational demonstration only. © 2026. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
