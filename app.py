import base64
import io
import json
import re
import threading
import time
import urllib.request
from collections import defaultdict, deque

import pymupdf
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field


# ==============================================================================
# 1. DATOVÉ SCHÉMA ZNALOSTNÍHO GRAFU
# ==============================================================================
class PIDNode(BaseModel):
    id: str = Field(description="Přesný TAG kód prvku dle ISA-5.1, např. TK-100, P-101, LIC-200, FV-101")
    category: str = Field(
        description="Kategorie: 'equipment' (nádrže, čerpadla, dmychadla), 'valve' (ventily, klapky), 'instrument' (snímače, regulátory), 'off_page' (odkaz na jiný list) nebo 'boundary' (vstup/výstup procesu)"
    )
    type: str = Field(description="Konkrétní typ zařízení/přístroje, např. Feed Tank, Centrifugal Pump, Flow Controller")
    line_spec: str = Field(default="", description="Označení potrubí u prvku, např. 4\"-PW-001, pokud existuje")
    description: str = Field(description="Stručný technický popis funkce prvku (max 6 slov)")
    page: int = Field(default=1, description="Číslo vybrané strany (1, 2, ...)")
    box_2d: list[int] = Field(
        default_factory=list,
        description="Souřadnice symbolu/bubliny na přiloženém obrázku [ymin, xmin, ymax, xmax] v rozsahu 0 až 1000"
    )


class PIDEdge(BaseModel):
    source: str = Field(description="ID (TAG) zdrojového uzlu")
    target: str = Field(description="ID (TAG) cílového uzlu")
    relation_type: str = Field(
        description="Typ vazby: 'flows_to' (procesní tok), 'signal' (řídicí signál), 'measures' (měřicí odběr) nebo 'interlock' (bezpečnostní blokace/trip)"
    )
    label: str = Field(default="", description="Označení potrubí nebo smyčky na trase, např. 3\"-PW-002")


class PageExtractionResult(BaseModel):
    nodes: list[PIDNode]
    edges: list[PIDEdge]


class GraphQueryResponse(BaseModel):
    answer: str = Field(description="Stručná, přesná technická odpověď v češtině s uvedením TAGů")
    cli_equivalent: str = Field(description="Ukázkový příkaz dotazu, např. kg_query.py downstream TK-100")
    highlighted_node_ids: list[str] = Field(description="Seznam všech TAG ID uzlů, které tvoří nalezenou cestu")
    highlighted_edge_pairs: list[list[str]] = Field(
        description="Seznam dvojic [source_id, target_id] hran, které mají svítit ve výkrese"
    )


# ==============================================================================
# 2. NASTAVENÍ STRÁNKY A GEMINI KLIENT
# ==============================================================================
st.set_page_config(
    page_title="P&ID → Knowledge Graph",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .kpi-val { font-size: 1.65rem; font-weight: 700; color: #f8fafc; line-height: 1.1; }
    .kpi-lbl { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ P&ID → Znalostní Mapa (Spatial KG)")
st.caption("Dlaždicová extrakce schématu, prostorový graf přímo nad výkresem a interaktivní trasování procesu")


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


# ==============================================================================
# 3. PŘÍPRAVA OBRAZU, DLAŽDIC (TILES) A STAŽENÍ Z URL
# ==============================================================================
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


def encode_jpeg_bytes(img: Image.Image, max_px: int = 1500, quality: int = 82) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def create_tile_jobs(img: Image.Image, page_num: int, tile_mode: bool) -> list[dict]:
    """Vytvoří seznam samostatných úloh (každá má 1 obrázek + přesné normované okno na listu)."""
    if not tile_mode:
        return [{
            "label": f"Strana {page_num} (celek)",
            "page": page_num,
            "rect": (0.0, 0.0, 1.0, 1.0),
            "jpeg": encode_jpeg_bytes(img, max_px=1550, quality=82),
        }]

    w, h = img.size
    # 4 překrývající se kvadranty (15% překryv uprostřed, aby se neztratily hrany přes střed)
    quads = [
        ("Dlaždice 1/4 (levá horní)", (0.0, 0.0, 0.575, 0.575)),
        ("Dlaždice 2/4 (pravá horní)", (0.425, 0.0, 1.0, 0.575)),
        ("Dlaždice 3/4 (levá dolní)", (0.0, 0.425, 0.575, 1.0)),
        ("Dlaždice 4/4 (pravá dolní)", (0.425, 0.425, 1.0, 1.0)),
    ]
    jobs = []
    for lbl, (x0, y0, x1, y1) in quads:
        crop_box = (int(w * x0), int(h * y0), int(w * x1), int(h * y1))
        crop_img = img.crop(crop_box)
        jobs.append({
            "label": f"Strana {page_num} – {lbl}",
            "page": page_num,
            "rect": (x0, y0, x1, y1),
            "jpeg": encode_jpeg_bytes(crop_img, max_px=1250, quality=82),
        })
    return jobs


# ==============================================================================
# 4. VOLÁNÍ GEMINI S TVRDÝM ČASOVAČEM MIMO STREAM + SLOUČENÍ DLAŽDIC
# ==============================================================================
def _single_attempt_worker(api_client, model_name: str, jpg_bytes: bytes, prompt_text: str, out_state: dict):
    try:
        img_part = types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")
        cfg_kwargs = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "max_output_tokens": 5120,
        }
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
        except Exception:
            pass

        try:
            stream = api_client.models.generate_content_stream(
                model=model_name,
                contents=[img_part, prompt_text],
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            chunks = []
            for chunk in stream:
                if getattr(chunk, "text", None):
                    chunks.append(chunk.text)
                    out_state["chars"] = sum(len(c) for c in chunks)
        except Exception as first_err:
            if "thinking" in str(first_err).lower() and "thinking_config" in cfg_kwargs:
                del cfg_kwargs["thinking_config"]
                stream = api_client.models.generate_content_stream(
                    model=model_name,
                    contents=[img_part, prompt_text],
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                chunks = []
                for chunk in stream:
                    if getattr(chunk, "text", None):
                        chunks.append(chunk.text)
                        out_state["chars"] = sum(len(c) for c in chunks)
            else:
                raise first_err

        out_state["text"] = "".join(chunks).strip()
    except Exception as exc:
        out_state["error"] = str(exc)
    finally:
        out_state["done"] = True


def run_tile_with_hard_timeout(
    api_client,
    models_to_try: list[str],
    job: dict,
    text_hint: str,
    status_placeholder,
    job_idx: int,
    total_jobs: int,
    per_model_timeout: float = 32.0,
) -> tuple[PageExtractionResult, str]:
    x0, y0, x1, y1 = job["rect"]
    prompt = f"""
    Jsi procesní inženýr a expert na P&ID schémata dle normy ISA-5.1.
    Zanalyzuj tento výřez P&ID výkresu ({job['label']}) a extrahuj znalostní graf.

    PRAVIDLA:
    1. Uzly (nodes): Všechna hlavní zařízení (equipment: nádrže, čerpadla, dmychadla, reaktory), armatury (valve: regulační i uzavírací ventily, zpětné klapky), přístroje (instrument: FT, PT, LT, LIC, FIC, LSLL, PI, TI...), odkazy na jiné listy (off_page) a vstupy/výstupy (boundary).
    2. Souřadnice (box_2d): U každého uzlu vrať polohu symbolu/bubliny PŘÍMO NA TOMTO OBRÁZKU [ymin, xmin, ymax, xmax] v rozsahu 0 až 1000.
    3. Hrany (edges): Propoj uzly podle skutečného vedení na výkrese:
       - "flows_to": směr toku média v procesním potrubí
       - "signal": řídicí nebo komunikační signál
       - "measures": měřicí odběr snímače na nádrž/potrubí
       - "interlock": bezpečnostní odstavení / blokace / trip
    4. Ignoruj razítko výkresu a obecné textové poznámky mimo schéma.

    Vrať VÝHRADNĚ validní JSON:
    {{
      "nodes": [
        {{"id": "TK-100", "category": "equipment", "type": "Feed Tank", "line_spec": "", "description": "Vstupní nádrž", "page": {job['page']}, "box_2d": [210, 140, 490, 260]}}
      ],
      "edges": [
        {{"source": "TK-100", "target": "P-101", "relation_type": "flows_to", "label": "4\\"-PW-001"}}
      ]
    }}{text_hint}
    """

    last_err = None
    for model_name in models_to_try:
        attempt_state = {"chars": 0, "text": "", "error": None, "done": False}
        t0 = time.time()
        th = threading.Thread(
            target=_single_attempt_worker,
            args=(api_client, model_name, job["jpeg"], prompt, attempt_state),
            daemon=True,
        )
        th.start()

        while th.is_alive():
            elapsed = round(time.time() - t0, 1)
            if elapsed > per_model_timeout:
                last_err = TimeoutError(f"{model_name} neodpověděl do {int(per_model_timeout)} s.")
                break
            status_placeholder.info(
                f"⏳ [{job_idx}/{total_jobs}] {job['label']} | model **{model_name}** | {elapsed} s / {int(per_model_timeout)} s | přijato {attempt_state['chars']} zn."
            )
            time.sleep(0.35)

        if not attempt_state["done"]:
            status_placeholder.warning(f"⚠️ {model_name} překročil {int(per_model_timeout)} s, přepínám na záložní model...")
            continue

        if attempt_state["error"]:
            last_err = RuntimeError(attempt_state["error"])
            status_placeholder.warning(f"⚠️ {model_name} vrátil chybu, zkouším další model...")
            time.sleep(0.6)
            continue

        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", attempt_state["text"]).strip()
        if not clean_json:
            last_err = ValueError(f"{model_name} vrátil prázdnou odpověď.")
            continue

        parsed = PageExtractionResult.model_validate_json(clean_json)

        # Přepočet lokálních souřadnic dlaždice [0..1000] na globální souřadnice celého listu [0..1000]
        span_x = x1 - x0
        span_y = y1 - y0
        for n in parsed.nodes:
            n.page = job["page"]
            if n.box_2d and len(n.box_2d) == 4:
                lymin, lxmin, lymax, lxmax = [max(0.0, min(1000.0, float(v))) for v in n.box_2d]
                gymin = int(round((y0 + (lymin / 1000.0) * span_y) * 1000))
                gxmin = int(round((x0 + (lxmin / 1000.0) * span_x) * 1000))
                gymax = int(round((y0 + (lymax / 1000.0) * span_y) * 1000))
                gxmax = int(round((x0 + (lxmax / 1000.0) * span_x) * 1000))
                n.box_2d = [gymin, gxmin, gymax, gxmax]

        return parsed, model_name

    raise RuntimeError(f"Úloha '{job['label']}' selhala na všech modelech: {last_err}")


def merge_extraction_results(results: list[PageExtractionResult]) -> PageExtractionResult:
    """Sloučí uzly a hrany ze všech dlaždic do jediného čistého grafu bez duplicit."""
    merged_nodes: dict[tuple[int, str], PIDNode] = {}
    id_canonical: dict[str, str] = {}

    for res in results:
        for n in res.nodes:
            clean_id = n.id.strip()
            if not clean_id:
                continue
            norm_key = clean_id.upper()
            canonical = id_canonical.setdefault(norm_key, clean_id)
            n.id = canonical
            map_key = (n.page, norm_key)

            if map_key not in merged_nodes:
                merged_nodes[map_key] = n
            else:
                existing = merged_nodes[map_key]
                if not existing.box_2d and n.box_2d:
                    existing.box_2d = n.box_2d
                if len(n.description) > len(existing.description):
                    existing.description = n.description
                if not existing.line_spec and n.line_spec:
                    existing.line_spec = n.line_spec

    merged_edges: dict[tuple[str, str, str], PIDEdge] = {}
    for res in results:
        for e in res.edges:
            s_norm = e.source.strip().upper()
            t_norm = e.target.strip().upper()
            if not s_norm or not t_norm or s_norm == t_norm:
                continue
            s_id = id_canonical.get(s_norm, e.source.strip())
            t_id = id_canonical.get(t_norm, e.target.strip())
            rel = e.relation_type.strip().lower() or "flows_to"
            edge_key = (s_id.upper(), t_id.upper(), rel)
            if edge_key not in merged_edges:
                merged_edges[edge_key] = PIDEdge(
                    source=s_id,
                    target=t_id,
                    relation_type=rel,
                    label=e.label.strip(),
                )
            elif not merged_edges[edge_key].label and e.label.strip():
                merged_edges[edge_key].label = e.label.strip()

    return PageExtractionResult(
        nodes=list(merged_nodes.values()),
        edges=list(merged_edges.values()),
    )


# ==============================================================================
# 5. ANALÝZA GRAFU (CONTROL LOOPS, QA CHYBY, DETERMINISTICKÉ TRASOVÁNÍ)
# ==============================================================================
def analyze_graph_metrics(nodes: list[PIDNode], edges: list[PIDEdge]) -> dict:
    node_map = {n.id: n for n in nodes}
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    adj_out = defaultdict(list)
    adj_in = defaultdict(list)

    qa_issues: list[str] = []
    valid_edges: list[PIDEdge] = []

    for e in edges:
        if e.source not in node_map:
            qa_issues.append(f"Hrana odkazuje na neznámý zdrojový uzel '{e.source}'.")
        if e.target not in node_map:
            qa_issues.append(f"Hrana odkazuje na neznámý cílový uzel '{e.target}'.")
        if e.source in node_map and e.target in node_map and e.source != e.target:
            valid_edges.append(e)
            out_deg[e.source] += 1
            in_deg[e.target] += 1
            adj_out[e.source].append(e)
            adj_in[e.target].append(e)

    for n in nodes:
        if in_deg[n.id] == 0 and out_deg[n.id] == 0:
            qa_issues.append(f"Izolovaný prvek '{n.id}' ({n.type}) nemá žádné napojení.")

    loop_groups: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        if n.category in ("instrument", "valve"):
            m = re.search(r"-(\d{2,4}[A-Z]?)$", n.id.strip().upper())
            if m:
                loop_groups[m.group(1)].append(n.id)

    control_loops = {k: v for k, v in loop_groups.items() if len(v) >= 2}

    if not control_loops:
        sig_nodes = set()
        for e in valid_edges:
            if e.relation_type in ("signal", "measures", "interlock"):
                sig_nodes.add(e.source)
                sig_nodes.add(e.target)
        if sig_nodes:
            visited = set()
            idx_loop = 1
            for sn in sig_nodes:
                if sn not in visited:
                    comp = []
                    q = deque([sn])
                    visited.add(sn)
                    while q:
                        curr = q.popleft()
                        comp.append(curr)
                        for e in adj_out[curr] + adj_in[curr]:
                            if e.relation_type in ("signal", "measures", "interlock"):
                                nxt = e.target if e.source == curr else e.source
                                if nxt not in visited:
                                    visited.add(nxt)
                                    q.append(nxt)
                    if len(comp) >= 2:
                        control_loops[f"LOOP-{idx_loop}"] = comp
                        idx_loop += 1

    return {
        "node_map": node_map,
        "valid_edges": valid_edges,
        "adj_out": adj_out,
        "adj_in": adj_in,
        "control_loops": control_loops,
        "qa_issues": qa_issues,
    }


def trace_subgraph(metrics: dict, start_id: str, mode: str) -> tuple[list[str], list[list[str]], str]:
    adj_out = metrics["adj_out"]
    adj_in = metrics["adj_in"]
    node_map = metrics["node_map"]

    if start_id not in node_map:
        return [], [], f"Prvek {start_id} nebyl v grafu nalezen."

    visited_nodes = {start_id}
    highlighted_pairs: list[list[str]] = []
    q = deque([start_id])

    while q:
        curr = q.popleft()
        if mode == "downstream":
            for e in adj_out[curr]:
                if e.relation_type == "flows_to":
                    highlighted_pairs.append([e.source, e.target])
                    if e.target not in visited_nodes:
                        visited_nodes.add(e.target)
                        q.append(e.target)
        elif mode == "upstream":
            for e in adj_in[curr]:
                if e.relation_type == "flows_to":
                    highlighted_pairs.append([e.source, e.target])
                    if e.source not in visited_nodes:
                        visited_nodes.add(e.source)
                        q.append(e.source)
        elif mode == "control":
            for e in adj_out[curr] + adj_in[curr]:
                if e.relation_type in ("signal", "measures"):
                    highlighted_pairs.append([e.source, e.target])
                    nxt = e.target if e.source == curr else e.source
                    if nxt not in visited_nodes:
                        visited_nodes.add(nxt)
                        q.append(nxt)
        elif mode == "interlock":
            for e in adj_in[curr] + adj_out[curr]:
                if e.relation_type == "interlock" or (e.relation_type == "signal" and "LS" in e.source.upper()):
                    highlighted_pairs.append([e.source, e.target])
                    nxt = e.target if e.source == curr else e.source
                    if nxt not in visited_nodes:
                        visited_nodes.add(nxt)
                        q.append(nxt)

    ordered_nodes = list(visited_nodes)
    others = [n for n in ordered_nodes if n != start_id]
    if not others:
        summary = f"Pro prvek **{start_id}** nebyly v tomto směru nalezeny žádné navazující prvky."
    else:
        summary = f"Nalezená cesta pro **{start_id}**: " + " → ".join(ordered_nodes)

    return ordered_nodes, highlighted_pairs, summary


# ==============================================================================
# 6. INTERAKTIVNÍ PROSTOROVÝ CANVAS OVERLAY (HTML5 + SVG)
# ==============================================================================
def render_spatial_kg_component(
    base_img: Image.Image,
    page_num: int,
    nodes: list[PIDNode],
    edges: list[PIDEdge],
    pdf_word_boxes: dict[str, tuple[float, float, float, float]],
    highlighted_nodes: list[str],
    highlighted_edges: list[list[str]],
    dark_blueprint: bool = True,
    show_boxes: bool = True,
    show_graph: bool = True,
    show_edge_labels: bool = False,
):
    w, h = base_img.size
    bg_bytes = encode_jpeg_bytes(base_img, max_px=1800, quality=82)
    bg_b64 = base64.b64encode(bg_bytes).decode("ascii")

    prepared_nodes = []
    coord_source_map = {}
    valid_ids = set()

    for n in nodes:
        if n.page != page_num:
            continue
        clean_key = n.id.strip().upper()
        box = None
        if clean_key in pdf_word_boxes:
            nx0, ny0, nx1, ny1 = pdf_word_boxes[clean_key]
            pad_x = 12
            pad_y = 14
            ymin = max(0, int(ny0 * 1000) - pad_y)
            xmin = max(0, int(nx0 * 1000) - pad_x)
            ymax = min(1000, int(ny1 * 1000) + pad_y)
            xmax = min(1000, int(nx1 * 1000) + pad_x)
            box = [ymin, xmin, ymax, xmax]
            coord_source_map[n.id] = "PDF vektor"
        elif n.box_2d and len(n.box_2d) == 4:
            ymin, xmin, ymax, xmax = [max(0, min(1000, int(round(float(v))))) for v in n.box_2d]
            if xmax > xmin and ymax > ymin:
                box = [ymin, xmin, ymax, xmax]
                coord_source_map[n.id] = "Vision AI"

        if box is not None:
            valid_ids.add(n.id)
            cx = (box[1] + box[3]) / 2.0
            cy = (box[0] + box[2]) / 2.0
            prepared_nodes.append({
                "id": n.id,
                "category": n.category,
                "type": n.type,
                "line_spec": n.line_spec,
                "description": n.description,
                "box": box,
                "cx": cx,
                "cy": cy,
            })

    prepared_edges = [
        {
            "source": e.source,
            "target": e.target,
            "relation_type": e.relation_type,
            "label": e.label,
        }
        for e in edges
        if e.source in valid_ids and e.target in valid_ids
    ]

    payload = {
        "imgW": w,
        "imgH": h,
        "nodes": prepared_nodes,
        "edges": prepared_edges,
        "hlNodes": highlighted_nodes,
        "hlEdges": highlighted_edges,
        "darkBlueprint": dark_blueprint,
        "showBoxes": show_boxes,
        "showGraph": show_graph,
        "showEdgeLabels": show_edge_labels,
    }

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  body {{ background: #080d16; color: #e2e8f0; overflow: hidden; }}
  #toolbar {{
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center; justify-content: space-between;
    padding: 8px 10px; background: #0f172a; border-bottom: 1px solid #1e293b; font-size: 12px;
  }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
  .leg-item {{ display: flex; align-items: center; gap: 5px; color: #cbd5e1; font-weight: 500; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .line-solid {{ width: 18px; height: 3px; background: #38bdf8; display: inline-block; }}
  .line-dash {{ width: 18px; height: 0; border-top: 2px dashed #f59e0b; display: inline-block; }}
  .line-dot {{ width: 18px; height: 0; border-top: 2px dotted #a855f7; display: inline-block; }}
  .line-trip {{ width: 18px; height: 0; border-top: 2px dashed #f43f5e; display: inline-block; }}
  .btns {{ display: flex; gap: 4px; }}
  button {{
    background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px;
    padding: 4px 9px; font-size: 12px; cursor: pointer; font-weight: 600;
  }}
  button:hover {{ background: #334155; }}
  #viewport {{
    position: relative; width: 100%; height: 520px; overflow: hidden;
    background: #080d16; touch-action: none; cursor: grab;
  }}
  #viewport:active {{ cursor: grabbing; }}
  #stage {{
    position: absolute; top: 0; left: 0; transform-origin: 0 0;
  }}
  #bg-img {{
    display: block; pointer-events: none; user-select: none;
  }}
  #overlay-svg {{
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  }}
  #inspector {{
    position: absolute; bottom: 10px; left: 10px; right: 10px;
    background: rgba(15, 23, 42, 0.94); border: 1px solid #38bdf8; border-radius: 8px;
    padding: 8px 12px; font-size: 12px; display: none; backdrop-filter: blur(6px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
  }}
  #inspector strong {{ color: #38bdf8; font-size: 13px; }}
</style>
</head>
<body>
<div id="toolbar">
  <div class="legend">
    <span class="leg-item"><span class="dot" style="background:#10b981"></span>Technologie</span>
    <span class="leg-item"><span class="dot" style="background:#3b82f6"></span>Armatura</span>
    <span class="leg-item"><span class="dot" style="background:#8b5cf6"></span>Přístroj</span>
    <span class="leg-item"><span class="line-solid"></span>flows_to</span>
    <span class="leg-item"><span class="line-dash"></span>signal</span>
    <span class="leg-item"><span class="line-dot"></span>measures</span>
    <span class="leg-item"><span class="line-trip"></span>interlock</span>
  </div>
  <div class="btns">
    <button onclick="zoomBy(1.25)">+</button>
    <button onclick="zoomBy(0.8)">−</button>
    <button onclick="resetView()">Celý list</button>
    <button onclick="clearSelection()">Reset výběru</button>
  </div>
</div>

<div id="viewport">
  <div id="stage">
    <img id="bg-img" src="data:image/jpeg;base64,{bg_b64}" />
    <svg id="overlay-svg"></svg>
  </div>
  <div id="inspector"></div>
</div>

<script>
const DATA = {json.dumps(payload)};
const viewport = document.getElementById('viewport');
const stage = document.getElementById('stage');
const bgImg = document.getElementById('bg-img');
const svg = document.getElementById('overlay-svg');
const inspector = document.getElementById('inspector');

const W = DATA.imgW;
const H = DATA.imgH;
stage.style.width = W + 'px';
stage.style.height = H + 'px';
bgImg.style.width = W + 'px';
bgImg.style.height = H + 'px';
svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);

if (DATA.darkBlueprint) {{
  bgImg.style.filter = 'invert(0.92) hue-rotate(195deg) contrast(1.05) brightness(0.42)';
}} else {{
  bgImg.style.filter = 'opacity(0.45)';
}}

let scale = 1, panX = 0, panY = 0;
let activeNodeId = null;

function applyTransform() {{
  stage.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
}}

function resetView() {{
  const vw = viewport.clientWidth || 800;
  const vh = viewport.clientHeight || 520;
  scale = Math.min(vw / W, vh / H) * 0.98;
  panX = (vw - W * scale) / 2;
  panY = (vh - H * scale) / 2;
  applyTransform();
}}

function zoomBy(factor) {{
  const vw = viewport.clientWidth || 800;
  const vh = viewport.clientHeight || 520;
  const cx = vw / 2, cy = vh / 2;
  const newScale = Math.max(0.15, Math.min(6.0, scale * factor));
  panX = cx - (cx - panX) * (newScale / scale);
  panY = cy - (cy - panY) * (newScale / scale);
  scale = newScale;
  applyTransform();
}}

function clearSelection() {{
  activeNodeId = null;
  inspector.style.display = 'none';
  renderOverlay();
}}

function catColor(cat) {{
  if (cat === 'equipment') return '#10b981';
  if (cat === 'valve') return '#3b82f6';
  if (cat === 'instrument') return '#8b5cf6';
  return '#64748b';
}}

function edgeStyle(rel) {{
  if (rel === 'signal') return {{ color: '#f59e0b', dash: '10,6', marker: 'arrow-signal' }};
  if (rel === 'measures') return {{ color: '#a855f7', dash: '4,6', marker: 'arrow-measures' }};
  if (rel === 'interlock') return {{ color: '#f43f5e', dash: '8,4', marker: 'arrow-interlock' }};
  return {{ color: '#38bdf8', dash: '', marker: 'arrow-flow' }};
}}

function renderOverlay() {{
  const hasExternalHL = (DATA.hlNodes && DATA.hlNodes.length > 0);
  const hlNodeSet = new Set(DATA.hlNodes || []);
  const hlEdgeSet = new Set((DATA.hlEdges || []).map(p => `${{p[0]}}->${{p[1]}}`));

  if (activeNodeId) {{
    hlNodeSet.clear();
    hlEdgeSet.clear();
    hlNodeSet.add(activeNodeId);
    DATA.edges.forEach(e => {{
      if (e.source === activeNodeId || e.target === activeNodeId) {{
        hlNodeSet.add(e.source);
        hlNodeSet.add(e.target);
        hlEdgeSet.add(`${{e.source}}->${{e.target}}`);
      }}
    }});
  }}
  const isFiltering = hasExternalHL || Boolean(activeNodeId);

  const nodeMap = {{}};
  DATA.nodes.forEach(n => {{ nodeMap[n.id] = n; }});

  const baseStroke = Math.max(2.5, W * 0.0018);
  const nodeR = Math.max(7, W * 0.0045);
  const fontSize = Math.max(13, W * 0.0085);

  let html = `
  <defs>
    <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrow-flow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#38bdf8"/>
    </marker>
    <marker id="arrow-signal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#f59e0b"/>
    </marker>
    <marker id="arrow-measures" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 2 L 8 5 L 0 8 z" fill="#a855f7"/>
    </marker>
    <marker id="arrow-interlock" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#f43f5e"/>
    </marker>
  </defs>`;

  if (DATA.showBoxes) {{
    DATA.nodes.forEach(n => {{
      const [ymin, xmin, ymax, xmax] = n.box;
      const x = (xmin / 1000) * W;
      const y = (ymin / 1000) * H;
      const bw = Math.max(14, ((xmax - xmin) / 1000) * W);
      const bh = Math.max(14, ((ymax - ymin) / 1000) * H);
      const col = catColor(n.category);
      const dimmed = isFiltering && !hlNodeSet.has(n.id);
      const op = dimmed ? 0.12 : 0.85;
      html += `<rect x="${{x}}" y="${{y}}" width="${{bw}}" height="${{bh}}" rx="4"
        fill="${{col}}" fill-opacity="${{dimmed ? 0.02 : 0.14}}" stroke="${{col}}" stroke-width="${{baseStroke * 0.7}}" stroke-opacity="${{op}}" />`;
    }});
  }}

  if (DATA.showGraph) {{
    DATA.edges.forEach(e => {{
      const s = nodeMap[e.source];
      const t = nodeMap[e.target];
      if (!s || !t) return;
      const x1 = (s.cx / 1000) * W;
      const y1 = (s.cy / 1000) * H;
      const x2 = (t.cx / 1000) * W;
      const y2 = (t.cy / 1000) * H;
      const dx = x2 - x1, dy = y2 - y1;
      const dist = Math.hypot(dx, dy);
      if (dist < 4) return;

      const pad = nodeR + 4;
      const tx = x2 - (dx / dist) * pad;
      const ty = y2 - (dy / dist) * pad;

      const st = edgeStyle(e.relation_type);
      const isHL = hlEdgeSet.has(`${{e.source}}->${{e.target}}`);
      const dimmed = isFiltering && !isHL;
      const opacity = dimmed ? 0.10 : 0.95;
      const sw = isHL ? baseStroke * 1.65 : baseStroke;
      const filt = isHL ? 'filter="url(#glow)"' : '';

      html += `<line x1="${{x1}}" y1="${{y1}}" x2="${{tx}}" y2="${{ty}}"
        stroke="${{st.color}}" stroke-width="${{sw}}" stroke-dasharray="${{st.dash}}"
        stroke-opacity="${{opacity}}" marker-end="url(#${{st.marker}})" ${{filt}} />`;

      if (DATA.showEdgeLabels && !dimmed) {{
        const mx = (x1 + tx) / 2;
        const my = (y1 + ty) / 2;
        const txt = e.label || e.relation_type;
        html += `<text x="${{mx}}" y="${{my - 4}}" fill="${{st.color}}" font-size="${{fontSize * 0.78}}"
          font-weight="600" text-anchor="middle" stroke="#080d16" stroke-width="3" paint-order="stroke">${{txt}}</text>`;
      }}
    }});

    DATA.nodes.forEach(n => {{
      const cx = (n.cx / 1000) * W;
      const cy = (n.cy / 1000) * H;
      const col = catColor(n.category);
      const isHL = hlNodeSet.has(n.id);
      const dimmed = isFiltering && !isHL;
      const opacity = dimmed ? 0.15 : 1.0;
      const r = isHL ? nodeR * 1.35 : nodeR;
      const filt = isHL ? 'filter="url(#glow)"' : '';

      html += `<g style="cursor:pointer; opacity:${{opacity}}" onclick="selectNode('${{n.id.replace(/'/g, "\\\\'")}}')">
        <circle cx="${{cx}}" cy="${{cy}}" r="${{r + 4}}" fill="${{col}}" fill-opacity="0.25" />
        <circle cx="${{cx}}" cy="${{cy}}" r="${{r}}" fill="${{col}}" stroke="#ffffff" stroke-width="${{isHL ? 2.5 : 1.5}}" ${{filt}} />
        <text x="${{cx}}" y="${{cy - r - 6}}" fill="#ffffff" font-size="${{fontSize}}" font-weight="700"
          text-anchor="middle" stroke="#080d16" stroke-width="4" paint-order="stroke">${{n.id}}</text>
      </g>`;
    }});
  }}

  svg.innerHTML = html;
}}

function selectNode(nid) {{
  activeNodeId = nid;
  const n = DATA.nodes.find(x => x.id === nid);
  if (n) {{
    const outE = DATA.edges.filter(e => e.source === nid).map(e => `${{e.relation_type}} → ${{e.target}}`);
    const inE = DATA.edges.filter(e => e.target === nid).map(e => `${{e.source}} → ${{e.relation_type}}`);
    inspector.innerHTML = `<strong>${{n.id}}</strong> (${{n.type}}) — ${{n.description}}
      ${{n.line_spec ? ` | Potrubí: <b>${{n.line_spec}}</b>` : ''}}<br/>
      <span style="color:#94a3b8">Vstupy: ${{inE.join(', ') || 'žádné'}} &nbsp;|&nbsp; Výstupy: ${{outE.join(', ') || 'žádné'}}</span>`;
    inspector.style.display = 'block';
  }}
  renderOverlay();
}}

let isDragging = false, startX = 0, startY = 0;
viewport.addEventListener('pointerdown', e => {{
  if (e.target.tagName === 'circle' || e.target.tagName === 'text') return;
  isDragging = true;
  startX = e.clientX - panX;
  startY = e.clientY - panY;
}});
window.addEventListener('pointermove', e => {{
  if (!isDragging) return;
  panX = e.clientX - startX;
  panY = e.clientY - startY;
  applyTransform();
}});
window.addEventListener('pointerup', () => {{ isDragging = false; }});
viewport.addEventListener('wheel', e => {{
  e.preventDefault();
  zoomBy(e.deltaY < 0 ? 1.15 : 0.87);
}}, {{ passive: false }});

resetView();
renderOverlay();
</script>
</body>
</html>
"""
    components.html(html_code, height=565, scrolling=False)
    return coord_source_map


# ==============================================================================
# 7. UŽIVATELSKÉ ROZHRANÍ APLIKACE
# ==============================================================================
available_models = get_available_models(client)
c_top1, c_top2 = st.columns([1.2, 1.8])
with c_top1:
    model_choice = st.selectbox(
        "Model Gemini",
        available_models,
        index=0,
        accept_new_options=True
    )
with c_top2:
    scan_mode = st.radio(
        "Režim čtení výkresu",
        ["⚡ Rychlý průchod (1 celek, ~15 s)", "🔬 Dlaždicový sken (Tile-by-Tile 2×2)"],
        index=1,
        horizontal=True
    )

zdroj = st.radio(
    "Zdroj P&ID výkresu",
    ["🔗 Odkaz (Google Disk / URL)", "📷 Vyfotit kamerou", "📁 Soubor z telefonu"],
    horizontal=True
)

raw_bytes = None
is_pdf = False

if zdroj == "🔗 Odkaz (Google Disk / URL)":
    file_url = st.text_input("Vložte sdílený odkaz na PDF / JPG / PNG")
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
                extracted_text_hints.append(f"\nTextová vrstva z PDF (nápověda TAGů):\n{page_txt[:2400]}")
            else:
                extracted_text_hints.append("")

            pix = page.get_pixmap(dpi=180)
            raw_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            hd_images.append(raw_img)
    else:
        raw_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        hd_images = [raw_img]
        selected_labels = ["Načtený výkres"]
        pdf_word_boxes_per_page = [{}]
        extracted_text_hints = [""]

if hd_images:
    if st.button(
        f"🚀 Vytvořit prostorový znalostní graf ({len(hd_images)} {'strana' if len(hd_images) == 1 else 'strany'})",
        type="primary",
        use_container_width=True
    ):
        t_start = time.time()
        tile_mode = scan_mode.startswith("🔬")
        with st.status("Skenuji výkres a sestavuji prostorový graf...", expanded=True) as status:
            try:
                all_jobs = []
                job_hints = []
                for idx_p, img in enumerate(hd_images, start=1):
                    p_jobs = create_tile_jobs(img, page_num=idx_p, tile_mode=tile_mode)
                    all_jobs.extend(p_jobs)
                    hint_for_page = extracted_text_hints[idx_p - 1] if idx_p - 1 < len(extracted_text_hints) else ""
                    job_hints.extend([hint_for_page] * len(p_jobs))

                models_queue = [model_choice] + [m for m in available_models if m != model_choice][:3]
                tile_results: list[PageExtractionResult] = []
                used_models_set = set()
                progress_box = st.empty()

                for j_idx, (job, hint_txt) in enumerate(zip(all_jobs, job_hints), start=1):
                    res_part, used_m = run_tile_with_hard_timeout(
                        api_client=client,
                        models_to_try=models_queue,
                        job=job,
                        text_hint=hint_txt,
                        status_placeholder=progress_box,
                        job_idx=j_idx,
                        total_jobs=len(all_jobs),
                        per_model_timeout=32.0,
                    )
                    tile_results.append(res_part)
                    used_models_set.add(used_m)
                    st.write(f"✅ **{job['label']}** hotovo ({len(res_part.nodes)} prvků, {len(res_part.edges)} vazeb | {used_m})")

                progress_box.empty()
                extracted_data = merge_extraction_results(tile_results)
                metrics = analyze_graph_metrics(extracted_data.nodes, extracted_data.edges)
                elapsed = round(time.time() - t_start, 1)
                used_models_str = ", ".join(sorted(used_models_set))

                st.session_state["pid_kg"] = {
                    "data": extracted_data,
                    "metrics": metrics,
                    "raw_json": extracted_data.model_dump_json(indent=2),
                    "labels": selected_labels,
                    "elapsed": elapsed,
                    "model": used_models_str,
                    "hl_nodes": [],
                    "hl_edges": [],
                    "qa_answer": None,
                }
                status.update(
                    label=f"Hotovo za {elapsed} s ({used_models_str}) — {len(extracted_data.nodes)} uzlů, {len(metrics['valid_edges'])} hran",
                    state="complete",
                    expanded=False
                )

            except Exception as e:
                status.update(label="Chyba při zpracování", state="error", expanded=True)
                st.error(f"Chyba při zpracování: {str(e)}")


# ==============================================================================
# 8. VÝSTUP: PROSTOROVÝ GRAF + KPI DASHBOARD + ASK THE GRAPH
# ==============================================================================
if "pid_kg" in st.session_state:
    kg = st.session_state["pid_kg"]
    extracted_data: PageExtractionResult = kg["data"]
    metrics: dict = kg["metrics"]
    saved_labels: list[str] = kg["labels"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(extracted_data.nodes)}</div><div class="kpi-lbl">Nodes (Prvky)</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(metrics["valid_edges"])}</div><div class="kpi-lbl">Edges (Vazby)</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(metrics["control_loops"])}</div><div class="kpi-lbl">Control Loops</div></div>', unsafe_allow_html=True)
    with k4:
        qa_cnt = len(metrics["qa_issues"])
        qa_col = "#10b981" if qa_cnt == 0 else "#f59e0b"
        st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{qa_col}">{qa_cnt}</div><div class="kpi-lbl">QA Errors</div></div>', unsafe_allow_html=True)

    st.write("")

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        dark_bp = st.checkbox("Tmavý Blueprint režim", value=True)
    with v2:
        show_boxes = st.checkbox("Rámečky symbolů", value=True)
    with v3:
        show_graph = st.checkbox("Prostorový graf", value=True)
    with v4:
        show_edge_lbl = st.checkbox("Popisky hran", value=False)

    all_coord_sources = {}
    if len(hd_images) == len(saved_labels) and len(hd_images) > 0:
        if len(hd_images) == 1:
            all_coord_sources = render_spatial_kg_component(
                hd_images[0],
                page_num=1,
                nodes=extracted_data.nodes,
                edges=metrics["valid_edges"],
                pdf_word_boxes=pdf_word_boxes_per_page[0],
                highlighted_nodes=kg["hl_nodes"],
                highlighted_edges=kg["hl_edges"],
                dark_blueprint=dark_bp,
                show_boxes=show_boxes,
                show_graph=show_graph,
                show_edge_labels=show_edge_lbl,
            )
        else:
            tabs_pages = st.tabs(saved_labels)
            for idx_p, (tab, img, wboxes) in enumerate(zip(tabs_pages, hd_images, pdf_word_boxes_per_page), start=1):
                with tab:
                    src_map = render_spatial_kg_component(
                        img,
                        page_num=idx_p,
                        nodes=extracted_data.nodes,
                        edges=metrics["valid_edges"],
                        pdf_word_boxes=wboxes,
                        highlighted_nodes=kg["hl_nodes"],
                        highlighted_edges=kg["hl_edges"],
                        dark_blueprint=dark_bp,
                        show_boxes=show_boxes,
                        show_graph=show_graph,
                        show_edge_labels=show_edge_lbl,
                    )
                    all_coord_sources.update(src_map)

    st.subheader("🔎 Ask the Graph (Trasování procesu a smyček přímo ve výkrese)")

    node_ids_list = [n.id for n in extracted_data.nodes]
    q_col1, q_col2 = st.columns([1.2, 2.8])
    with q_col1:
        target_tag = st.selectbox("Vyber uzel pro rychlý dotaz", node_ids_list, index=0) if node_ids_list else None

    with q_col2:
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("⬇️ Downstream (po proudu)", use_container_width=True) and target_tag:
                hn, he, ans = trace_subgraph(metrics, target_tag, "downstream")
                kg["hl_nodes"], kg["hl_edges"] = hn, he
                kg["qa_answer"] = {"q": f"What's downstream of {target_tag}?", "a": ans, "cli": f"kg_query.py downstream {target_tag}"}
                st.rerun()
        with b2:
            if st.button("⬆️ Upstream (proti proudu)", use_container_width=True) and target_tag:
                hn, he, ans = trace_subgraph(metrics, target_tag, "upstream")
                kg["hl_nodes"], kg["hl_edges"] = hn, he
                kg["qa_answer"] = {"q": f"What's upstream of {target_tag}?", "a": ans, "cli": f"kg_query.py upstream {target_tag}"}
                st.rerun()
        with b3:
            if st.button("🎛️ Regulační smyčka", use_container_width=True) and target_tag:
                hn, he, ans = trace_subgraph(metrics, target_tag, "control")
                kg["hl_nodes"], kg["hl_edges"] = hn, he
                kg["qa_answer"] = {"q": f"What does {target_tag} control / measure?", "a": ans, "cli": f"kg_query.py loop {target_tag}"}
                st.rerun()
        with b4:
            if st.button("🛑 Trip / Interlock", use_container_width=True) and target_tag:
                hn, he, ans = trace_subgraph(metrics, target_tag, "interlock")
                kg["hl_nodes"], kg["hl_edges"] = hn, he
                kg["qa_answer"] = {"q": f"What can trip {target_tag}?", "a": ans, "cli": f"kg_query.py interlock {target_tag}"}
                st.rerun()

    user_question = st.text_input(
        "Nebo polož vlastní dotaz nad grafem (např. 'Co ovládá ventil FV-101?' nebo 'Co je za čerpadlem P-101?')",
        placeholder="Napiš dotaz a stiskni Enter..."
    )
    if st.button("Zeptat se grafu a rozsvítit cestu", use_container_width=True) and user_question.strip():
        with st.spinner("Trasuji znalostní graf..."):
            try:
                graph_summary = {
                    "nodes": [{"id": n.id, "type": n.type, "category": n.category, "desc": n.description} for n in extracted_data.nodes],
                    "edges": [{"source": e.source, "target": e.target, "relation": e.relation_type, "label": e.label} for e in metrics["valid_edges"]],
                }
                qa_prompt = f"""
                Máš před sebou znalostní graf P&ID schématu:
                {json.dumps(graph_summary, ensure_ascii=False)}

                Dotaz uživatele: "{user_question.strip()}"
                Odpověz přesně na základě hran a uzlů v grafu a vrať seznam všech ID uzlů a dvojic hran [source, target], které tvoří odpověď, aby se rozsvítily na výkrese.
                """
                qa_resp = client.models.generate_content(
                    model=model_choice,
                    contents=[qa_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GraphQueryResponse,
                        temperature=0.0,
                    ),
                )
                parsed_qa = qa_resp.parsed or GraphQueryResponse.model_validate_json(qa_resp.text)
                kg["hl_nodes"] = parsed_qa.highlighted_node_ids
                kg["hl_edges"] = parsed_qa.highlighted_edge_pairs
                kg["qa_answer"] = {
                    "q": user_question.strip(),
                    "a": parsed_qa.answer,
                    "cli": parsed_qa.cli_equivalent,
                }
                st.rerun()
            except Exception as qa_err:
                st.error(f"Chyba při dotazu na graf: {qa_err}")

    if kg.get("qa_answer"):
        qa_box = kg["qa_answer"]
        st.info(f"**DOTAZ:** {qa_box['q']}\n\n**ODPOVĚĎ:** {qa_box['a']}\n\n`$ {qa_box['cli']}`")
        if st.button("Zobrazit opět celý neztmavený graf"):
            kg["hl_nodes"] = []
            kg["hl_edges"] = []
            kg["qa_answer"] = None
            st.rerun()

    t_nodes, t_edges, t_loops, t_qa = st.tabs([
        f"Prvky ({len(extracted_data.nodes)})",
        f"Vazby ({len(metrics['valid_edges'])})",
        f"Control Loops ({len(metrics['control_loops'])})",
        f"QA kontrola ({len(metrics['qa_issues'])})",
    ])

    with t_nodes:
        st.dataframe(
            [
                {
                    "TAG": n.id,
                    "Kategorie": n.category,
                    "Typ": n.type,
                    "Potrubí": n.line_spec,
                    "Popis": n.description,
                    "Zaměření": all_coord_sources.get(n.id, "Vision AI"),
                }
                for n in extracted_data.nodes
            ],
            use_container_width=True,
            hide_index=True,
        )

    with t_edges:
        st.dataframe(
            [
                {
                    "Zdroj": e.source,
                    "Typ vazby": e.relation_type,
                    "Cíl": e.target,
                    "Označení": e.label,
                }
                for e in metrics["valid_edges"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with t_loops:
        if metrics["control_loops"]:
            for loop_id, members in metrics["control_loops"].items():
                st.write(f"**Smyčka `{loop_id}`:** " + " • ".join(members))
        else:
            st.caption("Na tomto listu nebyly detekovány uzavřené regulační smyčky.")

    with t_qa:
        if metrics["qa_issues"]:
            for issue in metrics["qa_issues"]:
                st.warning(issue)
        else:
            st.success("0 QA chyb — všechny uzly a hrany jsou konzistentně propojené.")

    st.subheader("📥 Exportovat výsledky")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Stáhnout graph.json",
            data=kg["raw_json"],
            file_name="graph.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        cypher_lines = []
        for n in extracted_data.nodes:
            n_id = n.id.replace("'", "")
            n_type = n.type.replace("'", "")
            n_cat = n.category.replace("'", "")
            cypher_lines.append(f"CREATE (:Component {{id: '{n_id}', category: '{n_cat}', type: '{n_type}'}});")
        for e in metrics["valid_edges"]:
            src = e.source.replace("'", "")
            tgt = e.target.replace("'", "")
            rel = re.sub(r"[^A-Za-z0-9_]", "_", e.relation_type.upper()) or "CONNECTED_TO"
            cypher_lines.append(f"MATCH (a:Component {{id: '{src}'}}), (b:Component {{id: '{tgt}'}}) CREATE (a)-[:{rel}]->(b);")
        st.download_button(
            label="Stáhnout graph.cypher",
            data="\n".join(cypher_lines),
            file_name="graph.cypher",
            mime="text/plain",
            use_container_width=True,
        )

# --- KONEC SOUBORU ---
