import io
import re
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
    description: str = Field(description="Stručný popis funkce prvku ve schématu")

class PIDEdge(BaseModel):
    source: str = Field(description="ID (TAG) zdrojového prvku")
    target: str = Field(description="ID (TAG) cílového prvku")
    relation_type: str = Field(description="Typ propojení, např. 'process_pipe' nebo 'signal_line'")

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
    return genai.Client(api_key=api_key)

client = get_gemini_client()

def load_from_url(url: str) -> tuple[bytes, str]:
    drive_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if drive_match:
        file_id = drive_match.group(1)
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        content_type = resp.headers.get_content_type()
        data = resp.read()
        if "text/html" in content_type.lower():
            raise ValueError("Odkaz vrátil přihlašovací stránku místo souboru – nastav na Google Disku přístup 'Kdokoliv s odkazem'.")
        return data, content_type

model_choice = st.selectbox(
    "Model Gemini",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    index=0
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

images = []
selected_labels = []

if raw_bytes is not None:
    if is_pdf:
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        total_pages = len(doc)
        st.write(f"📄 **Počet stran v PDF:** {total_pages}")

        page_options = [f"Strana {i + 1}" for i in range(total_pages)]
        selected_labels = st.multiselect(
            "Vyberte strany k náhledu a analýze",
            options=page_options,
            default=page_options
        )

        for label in selected_labels:
            page_idx = page_options.index(label)
            pix = doc.load_page(page_idx).get_pixmap(dpi=220)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    else:
        images = [Image.open(io.BytesIO(raw_bytes))]
        selected_labels = ["Načtený výkres"]

if images:
    if len(images) == 1:
        st.image(images[0], caption=selected_labels[0])
    else:
        tabs = st.tabs(selected_labels)
        for tab, img, lbl in zip(tabs, images, selected_labels):
            with tab:
                st.image(img, caption=lbl)

    if st.button(f"🔍 Analyzovat ({len(images)} {'stranu' if len(images) == 1 else 'strany/stran'}) a vygenerovat graf", type="primary"):
        with st.spinner("Gemini čte technické schéma a sestavuje graf prvků..."):
            try:
                prompt = """
                Jsi expert na P&ID schémata. Zanalyzuj všechny přiložené strany výkresu podle normy ISA-5.1.
                Extrahuj všechny uzly (Equipment, Valves, Instruments) a jejich propojení (Piping, Signal lines)
                napříč všemi stranami do jednoho sjednoceného znalostního grafu (propoj i návaznosti mezi listy / Off-Page Connectors).
                U každého prvku uveď přesný TAG kód (např. PT-101, V-102, T-100).
                """

                response = client.models.generate_content(
                    model=model_choice,
                    contents=[*images, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PageExtractionResult,
                        temperature=0.1,
                    )
                )

                extracted_data = response.parsed or PageExtractionResult.model_validate_json(response.text)
                st.success(f"Extrakce dokončena! Nalezeno {len(extracted_data.nodes)} prvků a {len(extracted_data.edges)} propojení.")

                net = Network(height="500px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
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

                net.toggle_physics(True)
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
                        n_type = n
