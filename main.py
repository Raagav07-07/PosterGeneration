import os
import re
import json
import base64
from io import BytesIO
import asyncio

import streamlit as st
from dotenv import load_dotenv
from jinja2 import Template
from playwright.sync_api import sync_playwright

import google.genai as genai
from google.genai import types as genai_types

# Ensure subprocess support on Windows for Playwright
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# =========================
# ENV & GEMINI CLIENT SETUP
# =========================
load_dotenv()
st.set_page_config(
    page_title="Insurance Agent – Selvaraj",
    page_icon="data/Selvaraj.png"  
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)
TEXT_MODEL = "gemini-2.5-pro"

# 📐 Square poster (Option A: top content + bottom footer)
POSTER_WIDTH = 1080
POSTER_HEIGHT = 1080
FOOTER_HEIGHT = 210  # bottom strip for photo + name


# =========================
# COLOR HELPERS
# =========================

def color_from_name(name: str, fallback: str = "#111827") -> str:
    table = {
        "dark_blue": "#0f172a",
        "navy": "#0f172a",
        "blue": "#1d4ed8",
        "sky_blue": "#0ea5e9",
        "dark_green": "#166534",
        "green": "#16a34a",
        "red": "#b91c1c",
        "maroon": "#7f1d1d",
        "orange": "#ea580c",
        "gold": "#facc15",
        "black": "#111827",
        "dark_gray": "#374151",
        "white": "#f9fafb",
    }
    return table.get(name, fallback)


def theme_colors(theme: str):
    if theme == "green_gold":
        return "#e8f7eb", "#fef6d8", "#14532d"
    if theme == "red_yellow":
        return "#fee2e2", "#fef9c3", "#7f1d1d"
    if theme == "yellow_blue":
        return "#fef9c3", "#dbeafe", "#1e3a8a"
    # default blue_orange
    return "#e0f2fe", "#ffedd5", "#0f172a"


# =========================
# STYLE HELPERS
# =========================

def build_style_block(style_mode: str) -> str:
    """Extra instructions depending on style mode."""
    if style_mode == "Conversation":
        return """
STYLE:
- body_paragraph_ta should look like a SHORT conversation between
  "வாடிக்கையாளர்" மற்றும் "ஆலோசகர்".
- Use speaker labels like:
  "வாடிக்கையாளர்:" and "ஆலோசகர்:".
- 5–7 lines max, casual spoken Tamil, but still neat.
- bullet_points_ta should then summarize 3 key benefits in simple one-liners.
"""
    elif style_mode == "Fact-based awareness":
        return """
STYLE:
- body_paragraph_ta should highlight 2–3 simple facts or scenarios
  (உதாரணம்: hospital bills, எதிர்பாராத விபத்து, children's future).
- Use a slightly serious, informative tone.
- bullet_points_ta should look like 3 crisp benefits/facts.
"""
    else:  # Standard marketing
        return """
STYLE:
- Simple marketing style, friendly and emotional.
- body_paragraph_ta: directly talk to the reader ("நீங்கள்").
- bullet_points_ta: 3 small benefit lines (safety, medical expenses, tax, etc.).
"""


# =========================
# GEMINI TEXT GENERATION
# =========================

def generate_text(style_mode: str):
    """
    Ask Gemini for compact Tamil insurance copy, aware of poster layout
    and style mode. Robustly extracts text from the response.
    """
    style_block = build_style_block(style_mode)

    prompt = f"""
நீ ஒரு அனுபவம் வாய்ந்த தமிழ் இன்ஷூரன்ஸ் மார்க்கெட்டிங் காபி ரைட்டர்.

இந்த போஸ்டர் டிசைன் அளவு:
- width: {POSTER_WIDTH} px
- height: {POSTER_HEIGHT} px
- மேலே ~1200 px: தலைப்பு, subheadline, main body மற்றும் bullet points.
- நடுவில் ~150 px: CTA (call-to-action) line.
- கீழே {FOOTER_HEIGHT} px: ஏஜென்ட் புகைப்படம், பெயர், role, phone number footer க்கு RESERVED.
அதனால் டெக்ஸ்ட் குறுகிய, readable மற்றும் க்ரவுட் இல்லாத மாதிரி இருக்கணும்.

மொழி:
- இயல்பான பேச்சு தமிழில் (தமிழ்நாட்டில் இன்ஷூரன்ஸ் ஏஜென்ட் பேசுற மாதிரி).
- கடினமான / லிடரரி தமிழ் வேண்டாம்.
- English words MIX பண்ணாதே (numbers ok).

{style_block}

TEXT LIMITS (மிக அவசியம் பின்பற்று):
- headline_ta: அதிகபட்சம் 22–25 தமிழ் எழுத்துகள் அளவுக்கு. 1–2 வரி ஹுக் மட்டும்.
- subheadline_ta: max 70–80 எழுத்துகள்.
- body_paragraph_ta: max 220–250 எழுத்துகள் (3–4 சின்ன வாக்கியங்கள் அல்லது 5–7 சின்ன conversation lines).
- bullet_points_ta: 3 bullets மட்டும். ஒவ்வொரு bullet ஒரு வரி (max 45 எழுத்துகள்).
- cta_line_ta: short CTA, max 60 எழுத்துகள்.

CONTENT STRUCTURE (தமிழில் நிரப்ப):

1) headline_ta
2) subheadline_ta
3) body_paragraph_ta
4) bullet_points_ta  (3 strings)
5) cta_line_ta
6) color_theme :  "blue_orange" | "green_gold" | "red_yellow" | "yellow_blue"
7) text_colors:
   {{
     "headline": "dark_blue | navy | red | maroon | dark_green",
     "body": "black | dark_gray | dark_blue",
     "cta": "red | dark_blue | green | orange"
   }}

STRICTLY இந்த JSON format மட்டும் return பண்ணு
(extra text, explanation, markdown எதுவும் இல்லாமல்):

{{
  "headline_ta": "...",
  "subheadline_ta": "...",
  "body_paragraph_ta": "...",
  "bullet_points_ta": ["...", "...", "..."],
  "cta_line_ta": "...",
  "color_theme": "blue_orange",
  "text_colors": {{
    "headline": "dark_blue",
    "body": "black",
    "cta": "red"
  }}
}}
"""

    result = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
    )

    # -------- extract text safely from candidates --------
    chunks = []

    candidates = getattr(result, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            t = getattr(part, "text", None)
            if t:
                chunks.append(t)

    # fallback: some client versions also populate result.text
    if not chunks and getattr(result, "text", None):
        chunks.append(result.text)

    raw = "\n".join(chunks).strip() if chunks else ""

    if not raw:
        raise ValueError(
            "Gemini API did not return any text (possibly blocked / empty). "
            "Try again or change the style."
        )

    # -------- JSON extraction as before --------
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I)
    json_text = None
    if m:
        json_text = m.group(1)
    else:
        start = raw.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_text = raw[start : i + 1]
                        break

    if not json_text:
        raise ValueError("Could not find JSON in Gemini output:\n" + raw)

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        raise ValueError(
            f"Failed to parse JSON from Gemini: {e}\nExtracted:\n{json_text}"
        )

    # -------- normalize fields --------
    theme = parsed.get("color_theme", "blue_orange")
    if theme not in ["blue_orange", "green_gold", "red_yellow", "yellow_blue"]:
        theme = "blue_orange"
    parsed["color_theme"] = theme

    bullets = parsed.get("bullet_points_ta") or []
    if not isinstance(bullets, list):
        bullets = [str(bullets)]
    parsed["bullet_points_ta"] = bullets[:3]

    text_colors = parsed.get("text_colors") or {}
    parsed["text_colors"] = {
        "headline": text_colors.get("headline", "dark_blue"),
        "body": text_colors.get("body", "black"),
        "cta": text_colors.get("cta", "red"),
    }

    return parsed


# =========================
# HTML POSTER TEMPLATE  (square, layout A)
# =========================

POSTER_HTML = Template(
    r"""<!DOCTYPE html>
<html lang="ta">
<head>
  <meta charset="UTF-8" />
  <title>Insurance Poster</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    html, body {
      width: {{ width }}px;
      height: {{ height }}px;
      overflow: hidden;
      font-family: 'NotoTamil', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    @font-face {
      font-family: 'NotoTamil';
      src: url('{{ regular_font_data_url }}') format('truetype');
      font-weight: 400;
      font-style: normal;
    }
    @font-face {
      font-family: 'NotoTamil';
      src: url('{{ bold_font_data_url }}') format('truetype');
      font-weight: 700;
      font-style: normal;
    }

    body {
      background: linear-gradient(to bottom, {{ top_color }}, {{ bottom_color }});
      margin: 0;
    }
    .poster {
      width: {{ width }}px;
      height: {{ height }}px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .main {
      padding: 70px 80px 10px 80px;
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      color: #111827;
    }
    .headline {
      font-size: 46px;
      font-weight: 700;
      text-align: center;
      color: {{ headline_color }};
      margin-bottom: 12px;
      line-height: 1.3;
    }
    .subheadline {
      font-size: 26px;
      text-align: center;
      color: {{ headline_color }};
      margin-bottom: 18px;
      line-height: 1.4;
    }
    .body {
      font-size: 23px;
      color: {{ body_color }};
      margin-bottom: 12px;
      line-height: 1.6;
    }
    .bullets {
      font-size: 22px;
      color: {{ body_color }};
      margin-bottom: 18px;
      line-height: 1.6;
    }
    .bullet-item {
      margin-left: 16px;
      margin-bottom: 4px;
    }
    .cta {
      font-size: 24px;
      font-weight: 700;
      text-align: center;
      color: {{ cta_color }};
      margin-top: 16px;
    }

    .footer {
      height: {{ footer_height }}px;
      background: {{ footer_color }};
      display: flex;
      align-items: center;
      padding: 20px 60px;
      color: #f9fafb;
    }
    .footer-photo {
      width: 150px;
      height: 150px;
      border-radius: 50%;
      background-size: cover;
      background-position: center;
      background-image: url('{{ photo_data_url }}');
      flex-shrink: 0;
    }
    .footer-text {
      margin-left: 32px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 8px;
    }
    .footer-name {
      font-size: 28px;
      font-weight: 700;
    }
    .footer-role {
      font-size: 22px;
      opacity: 0.95;
    }
    .footer-phone {
      font-size: 22px;
    }
  </style>
</head>
<body>
  <div class="poster">
    <div class="main">
      <div class="headline">{{ headline }}</div>
      {% if subheadline %}
      <div class="subheadline">{{ subheadline }}</div>
      {% endif %}
      <div class="body">{{ body_paragraph }}</div>
      <div class="bullets">
        {% for b in bullet_points %}
          <div class="bullet-item">• {{ b }}</div>
        {% endfor %}
      </div>
      <div class="cta">{{ cta_line }}</div>
    </div>
    <div class="footer">
      <div class="footer-photo"></div>
      <div class="footer-text">
        <div class="footer-name">{{ agent_name }}</div>
        <div class="footer-role">{{ agent_role }}</div>
        <div class="footer-phone">📞 {{ agent_phone }}</div>
      </div>
    </div>
  </div>
</body>
</html>
"""
)


# =========================
# RENDER HTML → PNG (Playwright)
# =========================

def html_to_png(html: str, width: int, height: int) -> bytes:
    """
    Render the given HTML to a PNG using headless Chromium via Playwright.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": height, "deviceScaleFactor": 1}
        )
        page.set_content(html, wait_until="networkidle")
        # small delay to ensure fonts are applied
        page.wait_for_timeout(800)
        png_bytes = page.screenshot(full_page=False)
        browser.close()
    return png_bytes


# =========================
# STREAMLIT UI
# =========================

st.title("Selvaraj's Poster Generator")
st.write("உங்களின் பெயர் (English அல்லது தமிழில்) , உங்களுடைய பணி மற்றும் உங்களுடைய போட்டோவை மாற்றி கொள்ளலாம். ")

st.subheader("Details")
name = st.text_input("Name (English or Tamil)", os.getenv("AGENT_NAME", "Selvaraj D"))
role = st.text_input("Role (English or Tamil)", os.getenv("AGENT_ROLE", "இன்ஷூரன்ஸ் முகவர்"))
number = st.text_input("Phone Number", os.getenv("AGENT_PHONE", "9842761070"))

style_mode = st.selectbox(
    "Content style",
    ["Standard marketing", "Conversation", "Fact-based awareness"],
)

uploaded_img = st.file_uploader("Upload Agent Photo", type=["jpg", "jpeg", "png"])

if st.button("Generate Poster"):
    if uploaded_img is None:
        st.error("Please upload the agent photo.")
        st.stop()

    st.info(f"Generating Tamil insurance copy with Gemini… (Style: {style_mode})")
    agent_img_bytes = uploaded_img.read()

    try:
        text_spec = generate_text(style_mode)
    except Exception as e:
        st.error(f"Text generation failed: {e}")
        st.stop()

    # Theme & text colors
    top_color, bottom_color, footer_color = theme_colors(text_spec["color_theme"])
    headline_color = color_from_name(text_spec["text_colors"]["headline"], "#0f172a")
    body_color = color_from_name(text_spec["text_colors"]["body"], "#111827")
    cta_color = color_from_name(text_spec["text_colors"]["cta"], "#b91c1c")

    # Agent photo → data URL
    b64_photo = base64.b64encode(agent_img_bytes).decode("ascii")
    photo_data_url = f"data:image/png;base64,{b64_photo}"

    # Local Tamil fonts → data URLs
    with open("NotoSansTamil-Regular.ttf", "rb") as f:
        reg_b64 = base64.b64encode(f.read()).decode("ascii")
    with open("NotoSansTamil-Bold.ttf", "rb") as f:
        bold_b64 = base64.b64encode(f.read()).decode("ascii")

    regular_font_data_url = f"data:font/ttf;base64,{reg_b64}"
    bold_font_data_url = f"data:font/ttf;base64,{bold_b64}"

    # Render HTML
    html = POSTER_HTML.render(
        width=POSTER_WIDTH,
        height=POSTER_HEIGHT,
        footer_height=FOOTER_HEIGHT,
        top_color=top_color,
        bottom_color=bottom_color,
        footer_color=footer_color,
        headline_color=headline_color,
        body_color=body_color,
        cta_color=cta_color,
        headline=text_spec["headline_ta"],
        subheadline=text_spec.get("subheadline_ta", ""),
        body_paragraph=text_spec["body_paragraph_ta"],
        bullet_points=text_spec["bullet_points_ta"],
        cta_line=text_spec["cta_line_ta"],
        agent_name=name,
        agent_role=role,
        agent_phone=number,
        photo_data_url=photo_data_url,
        regular_font_data_url=regular_font_data_url,
        bold_font_data_url=bold_font_data_url,
    )

    st.info("Rendering square poster via headless Chromium…")

    try:
        poster_bytes = html_to_png(html, POSTER_WIDTH, POSTER_HEIGHT)
    except Exception as e:
        st.error(f"HTML → PNG rendering failed: {e}")
        st.stop()

    st.image(poster_bytes, caption="Generated Poster", use_container_width=True)
    st.download_button(
        label="Download Poster",
        data=poster_bytes,
        file_name="insurance_poster_tamil_square.png",
        mime="image/png",
    )
