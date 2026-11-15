import io
import math
import re
import time
import threading
import urllib.request
import pymupdf
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
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
    page: int = Field(default=1, description="Číslo přiloženého obrázku/strany (1, 2, ...)")
    box_2d: list[int] = Field(
        default_factory=list,
        description="Souřadnice prvku na obrázku [ymin, xmin, ymax, xmax] v rozsahu 0 až 1000"
    )

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
    preferred = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-pro", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
    if _client is None:
        return ["gemini-3.5-flash"]
    found = []
    try:
        for m in _client.models.list():
            name = getattr(m, "name", "") or ""
            short_name = name.replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in short_name and not any(x in short_name for x in ("embedding", "imagen", "veo", "aqa", "tts", "live")):
                if not actions or "generateContent" in actions:
                    found.append(short_name)
    except Exception:
        pass
    if not found:
        return ["gemini-3.5-flash"]
    ordered = [p for p in preferred if p in found]
    others = [f for f in found if f not in ordered]
    return ordered + others

def get_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def prepare_images(img: Image.Image, ai_max_px: int = 1500, hd_max_px: int = 2200) -> tuple[Image.Image, bytes]:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size

    hd_img = img
    if max(w, h) > hd_max_px:
        scale_hd = hd_max_px / float(max(w, h))
        hd_img = img.resize((int(w * scale_hd), int(h * scale_hd)), Image.Resampling.LANCZOS)

    ai_img = img
    if max(w, h) > ai_max_px:
        scale_ai = ai_max_px / float(max(w, h))
        ai_img = img.resize((int(w * scale_ai), int(h * scale_ai)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    ai_img.save(buf, format="JPEG", quality=82, optimize=True)
    return hd_img, buf.getvalue()

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

def draw_dashed_line(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], fill: str, width: int, dash_len: int = 18):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    steps = max(1, int(dist / dash_len))
    for i in range(0, steps, 2):
        t_start = i / steps
        t_end = min(1.0, (i + 1) / steps)
        sx = int(x1 + dx * t_start)
        sy = int(y1 + dy * t_start)
        ex = int(x1 + dx * t_end)
        ey = int(y1 + dy * t_end)
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

def draw_arrow(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], fill: str, width: int, dashed: bool = False):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 8:
        return

    ux, uy = dx / dist, dy / dist
    tip_offset = min(22.0, dist * 0.25)
    tx = int(x2 - ux * tip_offset)
    ty = int(y2 - uy * tip_offset)

    if dashed:
        draw_dashed_line(draw, (x1, y1), (tx, ty), fill=fill, width=width)
    else:
        draw.line([(x1, y1), (tx, ty)], fill=fill, width=width)

    arrow_len = max(18, width * 4)
    arrow_half_w = max(9, width * 2)
    bx = tx - ux * arrow_len
    by = ty - uy * arrow_len
    px, py = -uy, ux

    left_pt = (int(bx + px * arrow_half_w), int(by + py * arrow_half_w))
    right_pt = (int(bx - px * arrow_half_w), int(by - py * arrow_half_w))
    draw.polygon([(tx, ty), left_pt, right_pt], fill=fill)

def render_annotated_drawing(
    base_img: Image.Image,
    page_num: int,
    nodes: list[PIDNode],
    edges: list[PIDEdge],
    pdf_word_boxes: dict[str, tuple[float, float, float, float]],
    dim_background: bool = True,
    show_edge_labels: bool = False,
    highlight_tag: str = "Všechny prvky"
) -> tuple[Image.Image, dict[str, str], int]:
    img = base_img.copy()
    if dim_background:
        white_layer = Image.new("RGB", img.size, (245, 247, 250))
        img = Image.blend(img, white_layer, alpha=0.28)
        img = ImageEnhance.Contrast(img).enhance(0.92)

    draw = ImageDraw.Draw(img)
    w, h = img.size
    line_w = max(4, int(w * 0.0024))
    box_w = max(3, int(w * 0.0018))
    font_size = max(16, int(w * 0.0115))
    small_font_size = max(13, int(w * 0.009))
    font = get_font(font_size)
    small_font = get_font(small_font_size)

    node_boxes: dict[str, tuple[int, int, int, int]] = {}
    node_centers: dict[str, tuple[int, int]] = {}
    node_loc_source: dict[str, str] = {}
    node_cat_map: dict[str, str] = {}

    for n in nodes:
        if n.page != page_num:
            continue
        node_cat_map[n.id] = n.category
        clean_key = n.id.strip().upper()
        if clean_key in pdf_word_boxes:
            nx0, ny0, nx1, ny1 = pdf_word_boxes[clean_key]
            pad_x = max(14, int(w * 0.01))
            pad_y = max(14, int(h * 0.012))
            x1 = max(0, int(nx0 * w) - pad_x)
            y1 = max(0, int(ny0 * h) - pad_y)
            x2 = min(w - 1, int(nx1 * w) + pad_x)
            y2 = min(h - 1, int(ny1 * h) + pad_y)
            node_boxes[n.id] = (x1, y1, x2, y2)
            node_centers[n.id] = ((x1 + x2) // 2, (y1 + y2) // 2)
            node_loc_source[n.id] = "PDF souřadnice"
        elif n.box_2d and len(n.box_2d) == 4:
            ymin, xmin, ymax, xmax = [max(0, min(1000, int(round(float(v))))) for v in n.box_2d]
            if xmax > xmin and ymax > ymin:
                x1 = int(xmin / 1000.0 * w)
                y1 = int(ymin / 1000.0 * h)
                x2 = int(xmax / 1000.0 * w)
                y2 = int(ymax / 1000.0 * h)
                node_boxes[n.id] = (x1, y1, x2, y2)
                node_centers[n.id] = ((x1 + x2) // 2, (y1 + y2) // 2)
                node_loc_source[n.id] = "Vision AI"

    drawn_edges = 0
    for e in edges:
        if e.source in node_centers and e.target in node_centers:
            if highlight_tag != "Všechny prvky" and highlight_tag not in (e.source, e.target):
                continue
            p1 = node_centers[e.source]
            p2 = node_centers[e.target]
            is_signal = "signal" in e.relation_type.lower()
            edge_color = "#7B1FA2" if is_signal else "#E53935"
            draw_arrow(draw, p1, p2, fill=edge_color, width=line_w, dashed=is_signal)
            drawn_edges += 1

            if show_edge_labels:
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                lbl = e.relation_type
                bbox = draw.textbbox((mx, my), lbl, font=small_font)
                draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill="#212121", outline=edge_color, width=2)
                draw.text((mx, my), lbl, fill="#FFFFFF", font=small_font)

    cat_colors = {
        "equipment": "#00A152",
        "valve": "#EF6C00",
        "instrument": "#1565C0",
        "other": "#6A1B9A",
    }

    for nid, (x1, y1, x2, y2) in node_boxes.items():
        if highlight_tag != "Všechny prvky" and nid != highlight_tag:
            is_neighbor = any(
                (e.source == highlight_tag and e.target == nid) or (e.target == highlight_tag and e.source == nid)
                for e in edges
            )
            if not is_neighbor:
                continue

        color = cat_colors.get(node_cat_map.get(nid, "other"), "#1565C0")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_w)

        tbbox = draw.textbbox((0, 0), nid, font=font)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]
        bx1 = max(2, min(w - tw - 12, x1))
        by1 = max(2, y1 - th - 8)
        draw.rectangle([bx1, by1, bx1 + tw + 10, by1 + th + 6], fill=color)
        draw.text((bx1 + 5, by1 + 2), nid, fill="#FFFFFF", font=font)

    return img, node_loc_source, drawn_edges

def stream_with_retry_worker(api_client, models_to_try: list[str], jpg_list: list[bytes], prompt_text: str, fast_mode: bool, state: dict):
    image_parts = [
        types.Part.from_bytes(data=b, mime_type="image/jpeg")
        for b in jpg_list
    ]
    last_error = None

    for model_name in models_to_try:
        state["note"] = f"Zkouším {model_name}..."
        state["chars"] = 0
        state["thoughts"] = 0
        cfg_kwargs = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "max_output_tokens": 5120 if fast_mode else 8192,
        }
        if fast_mode:
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
            except Exception:
                pass

        try:
            t_model_start = time.time()
            stream = api_client.models.generate_content_stream(
                model=model_name,
                contents=[*image_parts, prompt_text],
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            chunks = []
            aborted_slow = False
            for chunk in stream:
                if state["chars"] == 0 and state["thoughts"] == 0 and (time.time() - t_model_start) > 25:
                    aborted_slow = True
                    last_error = TimeoutError(f"{model_name} neodpověděl do 25 s.")
                    break

                got_part_text = False
                candidates = getattr(chunk, "candidates", None) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) or []
                    for p in parts:
                        p_text = getattr(p, "text", None) or ""
                        if getattr(p, "thought", False):
                            state["thoughts"] += len(p_text)
                        elif p_text:
                            chunks.append(p_text)
                            state["chars"] += len(p_text)
                            got_part_text = True

                if not got_part_text and getattr(chunk, "text", None):
                    chunks.append(chunk.text)
                    state["chars"] += len(chunk.text)

            if aborted_slow:
                state["note"] = f"{model_name} neodpovídá, přepínám..."
                continue

            result_text = "".join(chunks).strip()
            if result_text:
                state["text"] = result_text
                state["used_model"] = model_name
                state["done"] = True
                return
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if "thinking" in err_str.lower() and "thinking_config" in cfg_kwargs:
                del cfg_kwargs["thinking_config"]
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
                            state["chars"] += len(chunk.text)
                    result_text = "".join(chunks).strip()
                    if result_text:
                        state["text"] = result_text
                        state["used_model"] = model_name
                        state["done"] = True
                        return
                except Exception as exc2:
                    last_error = exc2

            state["note"] = f"{model_name} nedostupný, zkouším další..."
            time.sleep(1)
            continue

    state["error"] = str(last_error) if last_error else "Nepodařilo se získat odpověď z žádného modelu."
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
    ["⚡ Rychlý (do 20 s)", "🧠 Standardní (detailnější)"],
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

hd_images: list[Image.Image] = []
jpeg_payloads: list[bytes] = []
selected_labels: list[str] = []
extracted_text_hints: list[str] = []
pdf_word_boxes_per_page: list[dict[str, tuple[float, float, float, float]]] = []

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

        for idx_sel, label in enumerate(selected_labels, start=1):
            page_idx = page_options.index(label)
            page = doc.load_page(page_idx)
            pw, ph = page.rect.width, page.rect.height

            word_map: dict[str, tuple[float, float, float, float]] = {}
            for w_item in page.get_text("words"):
                wx0, wy0, wx1, wy1, wtext = w_item[0], w_item[1], w_item[2], w_item[3], w_item[4]
                token = wtext.strip(".,:;()[]").upper()
                if len(token) >= 2 and token not in word_map:
                    word_map[token] = (wx0 / pw, wy0 / ph, wx1 / pw, wy1 / ph)
            pdf_word_boxes_per_page.append(word_map)

            page_txt = " ".join(page.get_text("text").split())
            if page_txt:
                extracted_text_hints.append(f"[Obrázek {idx_sel} - {label}]: {page_txt[:2200]}")

            pix = page.get_pixmap(dpi=160)
            raw_img = Image.open(io.BytesIO(pix.tobytes("png")))
            hd_img, jpg_bytes = prepare_images(raw_img)
            hd_images.append(hd_img)
            jpeg_payloads.append(jpg_bytes)
    else:
        raw_img = Image.open(io.BytesIO(raw_bytes))
        hd_img, jpg_bytes = prepare_images(raw_img)
        hd_images = [hd_img]
        jpeg_payloads = [jpg_bytes]
        selected_labels = ["Načtený výkres"]
        pdf_word_boxes_per_page = [{}]

if hd_images:
    with st.expander("🖼️ Původní načtený výkres (před analýzou)", expanded=False):
        if len(hd_images) == 1:
            st.image(hd_images[0], caption=selected_labels[0], use_container_width=True)
        else:
            tabs_orig = st.tabs(selected_labels)
            for tab, img, lbl in zip(tabs_orig, hd_images, selected_labels):
                with tab:
                    st.image(img, caption=lbl, use_container_width=True)

    if st.button(f"🔍 Analyzovat ({len(hd_images)} {'stranu' if len(hd_images) == 1 else 'strany/stran'}) a zakreslit do výkresu", type="primary", use_container_width=True):
        t_start = time.time()
        with st.status("Zpracovávám výkres (nechte displej telefonu rozsvícený)...", expanded=True) as status:
            try:
                total_kb = round(sum(len(b) for b in jpeg_payloads) / 1024, 1)
                st.write(f"1/3 Odesílám data ({total_kb} KB)...")
                timer_box = st.empty()

                hints_block = ""
                if extracted_text_hints:
                    hints_block = "\nTextová vrstva z PDF (nápověda TAG kódů):\n" + "\n".join(extracted_text_hints)

                prompt = f"""
                Jsi expert na P&ID schémata dle normy ISA-5.1.
                Zanalyzuj přiložené výkresy (číslované jako page: 1, 2...) a extrahuj MAXIMÁLNĚ 40 hlavních uzlů s TAG kódem (technologie, čerpadla, nádrže, regulační ventily, měřicí přístroje) a jejich přímá propojení (max 50 hran).
                Ignoruj drobné ruční ventily bez TAGu a texty v razítku výkresu.
                U KAŽDÉHO uzlu uveď přesnou polohu prvku/bubliny na obrázku v poli "box_2d": [ymin, xmin, ymax, xmax] v rozsahu 0 až 1000 (0=horní/levý okraj, 1000=dolní/pravý okraj).
                Vrať VÝHRADNĚ validní JSON v této struktuře:
                {{
                  "nodes": [
                    {{"id": "PT-101", "category": "instrument", "type": "Pressure Transmitter", "description": "Měření tlaku", "page": 1, "box_2d": [320, 410, 365, 455]}}
                  ],
                  "edges": [
                    {{"source": "P-101", "target": "PT-101", "relation_type": "process_pipe"}}
                  ]
                }}
                Hodnota category smí být jen: "equipment", "valve", "instrument" nebo "other".
                Popisy drž stručné (max 4 slova).{hints_block}
                """

                fast_mode = mode_choice.startswith("⚡")
                max_wait = 110
                models_queue = [model_choice] + [m for m in available_models if m != model_choice][:3]
                worker_state = {"chars": 0, "thoughts": 0, "note": "", "text": "", "used_model": model_choice, "error": None, "done": False}

                thread = threading.Thread(
                    target=stream_with_retry_worker,
                    args=(client, models_queue, jpeg_payloads, prompt, fast_mode, worker_state),
                    daemon=True,
                )
                thread.start()

                while not worker_state["done"]:
                    elapsed_now = round(time.time() - t_start, 1)
                    if elapsed_now > max_wait:
                        raise TimeoutError(f"API neodpovědělo do {max_wait} s.")
                    extra = f" | {worker_state['note']}" if worker_state["note"] else ""
                    thought_txt = f" (úvaha: {worker_state['thoughts']} zn.)" if worker_state["thoughts"] else ""
                    timer_box.info(
                        f"⏳ Gemini zpracovává... {elapsed_now} s / {max_wait} s | výstup: {worker_state['chars']} zn.{thought_txt}{extra}"
                    )
                    time.sleep(0.4)

                if worker_state["error"]:
                    raise RuntimeError(worker_state["error"])

                clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", worker_state["text"].strip())
                if not clean_json:
                    raise ValueError("Model vrátil prázdnou odpověď.")

                elapsed = round(time.time() - t_start, 1)
                timer_box.empty()
                st.write(f"2/3 Odpověď z modelu **{worker_state['used_model']}** přijata za **{elapsed} s**, ověřuji JSON...")

                extracted_data = PageExtractionResult.model_validate_json(clean_json)
                st.session_state["pid_analysis"] = {
                    "data": extracted_data,
                    "raw_json": clean_json,
                    "labels": selected_labels,
                    "elapsed": elapsed,
                    "model": worker_state["used_model"],
                }
                status.update(
                    label=f"Hotovo za {elapsed} s! Nalezeno {len(extracted_data.nodes)} prvků a {len(extracted_data.edges)} propojení.",
                    state="complete",
                    expanded=False
                )

            except Exception as e:
                status.update(label="Chyba při zpracování", state="error", expanded=True)
                st.error(f"Chyba při zpracování: {str(e)}")

if "pid_analysis" in st.session_state:
    res = st.session_state["pid_analysis"]
    extracted_data: PageExtractionResult = res["data"]
    saved_labels: list[str] = res["labels"]

    if len(hd_images) == len(saved_labels) and len(hd_images) > 0:
        st.subheader("📐 Propojení zakreslená přímo v původním výkrese")
        st.caption("🟢 Technologie (equipment) | 🔵 Přístroje (instrument) | 🟠 Armatury (valve) | 🔴 Procesní potrubí | 🟣 Signálové vedení (přerušovaně)")

        c_opt1, c_opt2, c_opt3 = st.columns([1, 1, 1.4])
        with c_opt1:
            dim_bg = st.checkbox("Ztlumit podklad výkresu", value=True)
        with c_opt2:
            show_edge_lbl = st.checkbox("Popisky typů vedení", value=False)
        with c_opt3:
            tag_filter = st.selectbox(
                "Zvýraznit prvek a jeho vazby",
                ["Všechny prvky"] + [n.id for n in extracted_data.nodes],
                index=0
            )

        all_loc_sources: dict[str, str] = {}
        total_drawn_edges = 0

        if len(hd_images) == 1:
            annotated_img, loc_map, d_edges = render_annotated_drawing(
                hd_images[0],
                page_num=1,
                nodes=extracted_data.nodes,
                edges=extracted_data.edges,
                pdf_word_boxes=pdf_word_boxes_per_page[0],
                dim_background=dim_bg,
                show_edge_labels=show_edge_lbl,
                highlight_tag=tag_filter,
            )
            all_loc_sources.update(loc_map)
            total_drawn_edges += d_edges
            st.image(annotated_img, caption=f"{saved_labels[0]} — zakresleno {len(loc_map)} prvků a {d_edges} přímých vazeb", use_container_width=True)

            img_buf = io.BytesIO()
            annotated_img.save(img_buf, format="PNG")
            st.download_button(
                label="🖼️ Stáhnout výkres se zakreslenými vazbami (PNG)",
                data=img_buf.getvalue(),
                file_name="pid_annotated.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            tabs_ann = st.tabs(saved_labels)
            for idx_p, (tab, img, lbl, wboxes) in enumerate(zip(tabs_ann, hd_images, saved_labels, pdf_word_boxes_per_page), start=1):
                with tab:
                    annotated_img, loc_map, d_edges = render_annotated_drawing(
                        img,
                        page_num=idx_p,
                        nodes=extracted_data.nodes,
                        edges=extracted_data.edges,
                        pdf_word_boxes=wboxes,
                        dim_background=dim_bg,
                        show_edge_labels=show_edge_lbl,
                        highlight_tag=tag_filter,
                    )
                    all_loc_sources.update(loc_map)
                    total_drawn_edges += d_edges
                    st.image(annotated_img, caption=f"{lbl} — zakresleno {len(loc_map)} prvků a {d_edges} vazeb", use_container_width=True)
    else:
        all_loc_sources = {}

    st.subheader("🕸️ Interaktivní znalostní mapa")
    net = Network(height="500px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 18,
        "font": {
          "size": 24,
          "color": "#FFFFFF",
          "face": "arial",
          "strokeWidth": 4,
          "strokeColor": "#000000"
        },
        "scaling": {
          "label": {
            "enabled": false
          }
        }
      },
      "edges": {
        "font": {
          "size": 14,
          "color": "#CCCCCC",
          "strokeWidth": 2,
          "strokeColor": "#1e1e1e"
        },
        "smooth": {
          "type": "dynamic"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3500,
          "centralGravity": 0.25,
          "springLength": 140
        }
      }
    }
    """)
    known_ids = set()

    for node in extracted_data.nodes:
        known_ids.add(node.id)
        node_color = "#4CAF50" if node.category == "equipment" else "#2196F3" if node.category == "instrument" else "#FF9800"
        net.add_node(node.id, label=node.id, title=f"{node.id} ({node.type}): {node.description}", color=node_color)

    for edge in extracted_data.edges:
        for endpoint in (edge.source, edge.target):
            if endpoint not in known_ids:
                known_ids.add(endpoint)
                net.add_node(endpoint, label=endpoint, title="Odkazovaný prvek / Off-page connector", color="#9E9E9E")
        net.add_edge(edge.source, edge.target, label=edge.relation_type)

    components.html(net.generate_html(), height=520, scrolling=False)

    st.subheader("📋 Přehled extrahovaných prvků a propojení")
    tab_nodes, tab_edges = st.tabs([
        f"Prvky ({len(extracted_data.nodes)})",
        f"Propojení ({len(extracted_data.edges)})"
    ])

    with tab_nodes:
        nodes_rows = [
            {
                "TAG": n.id,
                "Kategorie": n.category,
                "Typ": n.type,
                "Popis": n.description,
                "Poloha ve výkrese": all_loc_sources.get(n.id, "Nenalezeno"),
            }
            for n in extracted_data.nodes
        ]
        st.dataframe(nodes_rows, use_container_width=True, hide_index=True)

    with tab_edges:
        edges_rows = [
            {
                "Zdroj": e.source,
                "Směr": "→",
                "Cíl": e.target,
                "Typ propojení": e.relation_type,
                "Ve výkrese": "Zakresleno" if (e.source in all_loc_sources and e.target in all_loc_sources) else "Pouze v grafu",
            }
            for e in extracted_data.edges
        ]
        st.dataframe(edges_rows, use_container_width=True, hide_index=True)

    st.subheader("📥 Exportovat výsledky")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Stáhnout JSON",
            data=res["raw_json"],
            file_name="pid_extraction.json",
            mime="application/json",
            use_container_width=True
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
            mime="text/plain",
            use_container_width=True
        )

# --- KONEC SOUBORU ---
