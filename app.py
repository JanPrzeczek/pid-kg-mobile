import io
import re
import time
import threading
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
    description: str = Field(description="Stručný popis funkce prvku (max 5 slov)")

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

def prepare_image_jpeg(img: Image.Image, max_px: int = 1400) -> tuple[Image.Image, bytes]:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80, optimize=True)
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

def build_thinking_config(fast_mode: bool):
    candidates = (
        [{"thinking_level": "MINIMAL"}, {"thinking_level": "LOW"}, {"thinking_budget": 0}]
        if fast_mode
        else [{"thinking_level": "MEDIUM"}, {"thinking_budget": 1024}]
    )
    for kw in candidates:
        try:
            return types.ThinkingConfig(**kw)
        except Exception:
            continue
    return None

def stream_gemini_worker(api_client, model_name: str, jpg_list: list[bytes], prompt_text: str, fast_mode: bool, state: dict):
    try:
        image_parts = [
            types.Part.from_bytes(data=b, mime_type="image/jpeg")
            for b in jpg_list
        ]
        cfg_kwargs = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "max_output_tokens": 4096 if fast_mode else 8192,
        }
        t_cfg = build_thinking_config(fast_mode)
        if t_cfg is not None:
            cfg_kwargs["thinking_config"] = t_cfg

        try:
            stream = api_client.models.generate_content_stream(
                model=model_name,
                contents=[*image_parts, prompt_text],
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            chunks = []
            for chunk in stream:
                if getattr(chunk, "text", None):
                    chunks.append(chunk.text)
                    state["chars"] = sum(len(c) for c in chunks)
        except Exception as first_err:
            if "thinking_config" in cfg_kwargs:
                state["note"] = "Přepínám na výchozí konfiguraci modelu..."
                del cfg_kwargs["thinking_config"]
                stream = api_client.models.generate_content_stream(
                    model=model_name,
                    contents=[*image_parts, prompt_text],
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                chunks = []
                for chunk in stream:
                    if getattr(chunk, "text", None):
                        chunks.append(chunk.text)
                        state["chars"] = sum(len(c) for c in chunks)
            else:
                raise first_err

        state["text"] = "".join(chunks)
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        state["done"] = True

available_models = get_available_models(client)
model_choice = st.selectbox(
    "Model Gemini",
    available_models,
    index=0,
    accept_new_options=True
)

mode_choice = st.radio(
    "Režim zpracování",
    ["⚡ Rychlý (hlavní TAGy, do 25 s)", "🧠 Standardní (detailnější)"],
    index=0,
    horizontal=True
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
extracted_text_hints = []

if raw_bytes is not None:
    if is_pdf:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        total_pages = len(doc)
        st.write(f"📄 **Počet stran v PDF:** {total_pages}")

        page_options = [f"Strana {i + 1}" for i in range(total_pages)]
        default_page = page_options[8] if total_pages >= 9 else page_options[0]
        selected_labels = st.multiselect(
            "Vyberte strany k náhledu a analýze",
            options=page_options,
            default=[default_page]
        )

        for label in selected_labels:
            page_idx = page_options.index(label)
            page = doc.load_page(page_idx)
            page_txt = " ".join(page.get_text("text").split())
            if page_txt:
                extracted_text_hints.append(f"[{label}]: {page_txt[:2200]}")
            pix = page.get_pixmap(dpi=130)
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
                st.write(f"1/3 Odesílám data ({total_kb} KB) do **{model_choice}**...")
                timer_box = st.empty()

                hints_block = ""
                if extracted_text_hints:
                    hints_block = "\nTextová vrstva z PDF (nápověda TAG kódů):\n" + "\n".join(extracted_text_hints)

                prompt = f"""
                Jsi expert na P&ID schémata dle normy ISA-5.1.
                Zanalyzuj přiložený výkres a extrahuj MAXIMÁLNĚ 40 nejdůležitějších uzlů s TAG kódem (hlavní technologie, čerpadla, nádrže, regulační ventily, měřicí přístroje) a jejich přímá propojení (max 50 hran).
                Ignoruj drobné ruční odvzdušňovací/vypouštěcí ventily bez TAGu a texty v razítku výkresu.
                Vrať VÝHRADNĚ validní JSON v této struktuře:
                {{
                  "nodes": [
                    {{"id": "PT-101", "category": "instrument", "type": "Pressure Transmitter", "description": "Měření tlaku"}}
                  ],
                  "edges": [
                    {{"source": "P-101", "target": "PT-101", "relation_type": "process_pipe"}}
                  ]
                }}
                Hodnota category smí být jen: "equipment", "valve", "instrument" nebo "other".
                Popisy drž velmi stručné (max 4 slova).{hints_block}
                """

                fast_mode = mode_choice.startswith("⚡")
                max_wait = 110
                worker_state = {"chars": 0, "note": "", "text": "", "error": None, "done": False}

                thread = threading.Thread(
                    target=stream_gemini_worker,
                    args=(client, model_choice, jpeg_payloads, prompt, fast_mode, worker_state),
                    daemon=True,
                )
                thread.start()

                while not worker_state["done"]:
                    elapsed_now = round(time.time() - t_start, 1)
                    if elapsed_now > max_wait:
                        raise TimeoutError(f"API neodpovědělo do {max_wait} s.")
                    extra = f" | {worker_state['note']}" if worker_state["note"] else ""
                    timer_box.info(
                        f"⏳ Gemini zpracovává... {elapsed_now} s / {max_wait} s | přijato {worker_state['chars']} znaků{extra}"
                    )
                    time.sleep(0.4)

                if worker_state["error"]:
                    raise RuntimeError(worker_state["error"])

                clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", worker_state["text"].strip())
                if not clean_json:
                    raise ValueError("Model vrátil prázdnou odpověď.")

                elapsed = round(time.time() - t_start, 1)
                timer_box.empty()
                st.write(f"2/3 Odpověď přijata za **{elapsed} s** ({len(clean_json)} znaků), ověřuji JSON...")

                extracted_data = PageExtractionResult.model_validate_json(clean_json)
                st.write(f"3/3 Vykresluji graf ({len(extracted_data.nodes)} prvků, {len(extracted_data.edges)} propojení)...")
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
                        data=clean_json,
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
