import io
import re
import time
import urllib.request
import pymupdf
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pyvis.network import Network
import streamlit.components.v1 as components

class PIDNode(BaseModel):
    id: str = Field(description="Přesný TAG kód prvku dle ISA-5.1, např. PT-101, V-102, P-101")
    category: str = Field(description="Kategorie prvku: 'equipment', 'valve', 'instrument' nebo 'other'")
    type: str = Field(description="Typ zařízení nebo přístroje, např. Pressure Transmitter, Valve, Tank")
    description: str = Field(description="Stručný popis funkce prvku (max 6 slov)")

class PIDEdge(BaseModel):
    source: str = Field(description="ID (TAG) zdrojového prvku")
    target: str = Field(description="ID (TAG) cílového prvku")
    relation_type: str = Field(description="Typ propojení: 'process_pipe' nebo 'signal_line'")

class PageExtractionResult(BaseModel):
    nodes: list[PIDNode]
    edges: list[PIDEdge]

st.set_page_config(
    page_title="P&ID do Znalostní Mapy",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("⚙️ P&ID do Znalostní Mapy")
st.caption("Extrakce komponent a topologie procesního schématu pomocí Gemini Vision API")

@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error("Není nastaven GEMINI_API_KEY v Secrets!")
        return None
    return genai.Client(api_key=api_key)

client = get_gemini_client()

@st.cache_data(ttl=3600)
def get_available_models(_client) -> list[str]:
    preferred = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-pro"]
    if _client is None:
        return ["gemini-3.8-flash"]
    found = []
    try:
        for m in _client.models.list():
            name = getattr(m, "name", "") or ""
            short_name = name.replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in short_name and not any(x in short_name for x in ("embedding", "imagen", "veo", "aqa")):
                if not actions or "generateContent" in actions:
                    found.append(short_name)
    except Exception:
        pass
    if not found:
        return ["gemini-3.8-flash"]
    ordered = [p for p in preferred if p in found]
    others = [f for f in found if f not in ordered]
    return ordered + others

def prepare_image_jpeg(img: Image.Image, max_px: int = 2000) -> tuple[Image.Image, bytes]:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return img, buf.getvalue()

def load_from_url(url: str) -> tuple[bytes, str]:
    drive_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url, re.IGNORECASE) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url, re.IGNORECASE)
    if drive_match:
        file_id = drive_match.group(1)
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content_type = resp.headers.get_content_type()
        data = resp.read()
        if "text/html" in content_type.lower() and data[:4] != b"%PDF":
            raise ValueError("Odkaz vrátil HTML stránku místo souboru – ověř na Google Disku nastavení 'Kdokoliv s odkazem'.")
        return data, content_type

available_models = get_available_models(client)
model_choice = st.selectbox(
    "Model Gemini",
    available_models,
    index=0,
    accept_new_options=True
)

thinking_budget = st.select_slider(
    "Hloubka analýzy (Thinking budget)",
    options=[0, 1500, 4000],
    value=1500,
    format_func=lambda x: "Rychlá (0)" if x == 0 else ("Vyvážená (1500)" if x == 1500 else "Hloubková (4000)")
)

zdroj = st.radio(
    "Zdroj výkresu",
    ["🔗 Odkaz (Google Disk / URL)", "📷 Vyfotit kamerou", "📁 Soubor z telefonu"],
    horizontal=False
)

raw_bytes = None
is_pdf = False

if zdroj == "🔗 Odkaz (Google Disk / URL)":
    file_url = st.text_input("Vložte sdílený odkaz na JPG/PNG/PDF")
    if file_url:
        try:
            with st.spinner("Stahuji výkres z odkazu..."):
                raw_bytes, content_type = load_from_url(file_url.strip())
                is_pdf = "pdf" in content_type.lower() or raw_bytes[:4] == b"%PDF"
                st.info(f"Výkres stažen ({round(len(raw_bytes) / 1024, 1)} KB)")
        except Exception as e:
            st.error(f"Nepodařilo se načíst soubor z odkazu: {e}")

elif zdroj == "📷 Vyfotit kamerou":
    cam_file = st.camera_input("Vyfoťte P&ID schéma")
    if cam_file is not None:
        raw_bytes = cam_file.getvalue()
        st.info(f"Načten snímek z kamery ({round(len(raw_bytes) / 1024, 1)} KB)")

else:
    uploaded_file = st.file_uploader("Nahrajte P&ID výkres (PDF, PNG, JPG)")
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        is_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf")
        st.info(f"Načten soubor: {uploaded_file.name} ({round(len(raw_bytes) / 1024, 1)} KB)")

preview_images = []
jpeg_payloads = []
selected_labels = []

if raw_bytes is not None:
    if is_pdf:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        total_pages = len(doc)
        st.write(f"📄 **Počet stran v PDF:** {total_pages} *(pro rychlou odezvu na mobilu vybírej 1–2 strany)*")

        page_options = [f"Strana {i + 1}" for i in range(total_pages)]
        selected_labels = st.multiselect(
            "Vyberte strany k náhledu a analýze",
            options=page_options,
            default=[page_options[0]]
        )

        for label in selected_labels:
            page_idx = page_options.index(label)
            pix = doc.load_page(page_idx).get_pixmap(dpi=160)
            raw_img = Image.open(io.BytesIO(pix.tobytes("png")))
            prev_img, jpg_bytes = prepare_image_jpeg(raw_img)
            preview_images.append(prev_img)
            jpeg_payloads.append(jpg_bytes)
    else:
        raw_img = Image.open(io.BytesIO(raw_bytes))
        prev_img, jpg_bytes = prepare_image_jpeg(raw_img)
        preview_images = [prev_img]
        jpeg_payloads = [jpg_bytes]
        selected_labels = ["Načtený výkres"]

if preview_images:
    if len(preview_images) == 1:
        st.image(preview_images[0], caption=selected_labels[0])
    else:
        tabs = st.tabs(selected_labels)
        for tab, img, lbl in zip(tabs, preview_images, selected_labels):
            with tab:
                st.image(img, caption=lbl)

    if st.button(f"🔍 Analyzovat ({len(preview_images)} {'stranu' if len(preview_images) == 1 else 'strany/stran'}) a vygenerovat graf", type="primary"):
        t_start = time.time()
        with st.status("Zpracovávám výkres (nechte displej telefonu rozsvícený)...", expanded=True) as status:
            try:
                total_kb = round(sum(len(b) for b in jpeg_payloads) / 1024, 1)
                st.write(f"1/3 Odesílám optimalizovaná data ({total_kb} KB) do modelu **{model_choice}** (thinking budget: {thinking_budget})...")

                image_parts = [
                    types.Part.from_bytes(data=b, mime_type="image/jpeg")
                    for b in jpeg_payloads
                ]

                prompt = """
                Jsi expert na P&ID schémata. Zanalyzuj výkres podle normy ISA-5.1.
                Extrahuj hlavní technologické uzly (Equipment, Valves, Instruments s TAG kódem) a jejich přímá propojení (Piping, Signal lines).
                Neopakuj duplicitní hrany, popisy drž velmi stručné.
                """

                response = client.models.generate_content(
                    model=model_choice,
                    contents=[*image_parts, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PageExtractionResult,
                        temperature=0.1,
                        max_output_tokens=16384,
                        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                    )
                )

                elapsed = round(time.time() - t_start, 1)
                st.write(f"2/3 Odpověď z Gemini přijata za **{elapsed} s**, parsuji strukturu...")

                extracted_data = response.parsed or PageExtractionResult.model_validate_json(response.text)
                st.write(f"3/3 Vykresluji interaktivní graf ({len(extracted_data.nodes)} prvků, {len(extracted_data.edges)} propojení)...")
                status.update(label=f"Hotovo za {elapsed} s! Nalezeno {len(extracted_data.nodes)} prvků a {len(extracted_data.edges)} propojení.", state="complete", expanded=False)

                net = Network(height="500px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
                net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)
                known_ids = set()

                for node in extracted_data.nodes:
                    known_ids.add(node.id)
                    node_color = "#4CAF50" if node.category == "equipment" else "#2196F3" if node.category == "instrument" else "#FF9800"
                    net.add_node(node.id, label=f"{node.id}\n({node.type})", title=node.description, color=node_color)

                for edge in extracted_data.edges:
                    for endpoint in (edge.source, edge.target):
                        if endpoint not in known_ids:
                            known_ids.add(endpoint)
                            net.add_node(endpoint, label=endpoint, title="Odkazovaný prvek / Off-page connector", color="#9E9E9E")
                    net.add_edge(edge.source, edge.target, label=edge.relation_type)

                components.html(net.generate_html(), height=520, scrolling=False)

                st.subheader("📥 Exportovat výsledky")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="Stáhnout JSON",
                        data=response.text,
                        file_name="pid_extraction.json",
                        mime="application/json"
                    )

                with col2:
                    cypher_lines = []
                    for n in extracted_data.nodes:
                        n_id = n.id.replace("'", "")
                        n_type = n.type.replace("'", "")
                        cypher_lines.append(f"CREATE (:Component {{id: '{n_id}', type: '{n_type}'}});")
                    for e in extracted_data.edges:
                        src = e.source.replace("'", "")
                        tgt = e.target.replace("'", "")
                        rel = re.sub(r"[^A-Za-z0-9_]", "_", e.relation_type.upper()) or "CONNECTED_TO"
                        cypher_lines.append(f"MATCH (a:Component {{id: '{src}'}}), (b:Component {{id: '{tgt}'}}) CREATE (a)-[:{rel}]->(b);")
                    st.download_button(
                        label="Stáhnout Cypher",
                        data="\n".join(cypher_lines),
                        file_name="pid_graph.cypher",
                        mime="text/plain"
                    )

            except Exception as e:
                status.update(label="Chyba při zpracování", state="error", expanded=True)
                st.error(f"Chyba při zpracování: {str(e)}")

# --- KONEC SOUBORU ---
