import io
import pymupdf
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pyvis.network import Network
import streamlit.components.v1 as components

# Definice datového schématu pro strukturovaný výstup z Gemini
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

def convert_pdf_to_image(pdf_bytes: bytes) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    return Image.open(io.BytesIO(pix.tobytes("png")))

model_choice = st.selectbox(
    "Model Gemini",
    ["gemini-2.5-flash", "gemini-2.5-pro"],
    index=0
)

uploaded_file = st.file_uploader(
    "Nahrajte P&ID výkres (PDF, PNG, JPG)",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        image = convert_pdf_to_image(uploaded_file.read())
    else:
        image = Image.open(uploaded_file)

    st.image(image, caption="Načtený výkres", use_container_width=True)

    if st.button("🔍 Analyzovat a vygenerovat graf", type="primary", use_container_width=True):
        with st.spinner("Gemini čte technické schéma a sestavuje graf prvků..."):
            try:
                prompt = """
                Jsi expert na P&ID schémata. Zanalyzuj tento výkres podle normy ISA-5.1.
                Extrahuj všechny uzly (Equipment, Valves, Instruments) a jejich propojení (Piping, Signal lines).
                U každého prvku uveď přesný TAG kód (např. PT-101, V-102, T-100).
                """

                response = client.models.generate_content(
                    model=model_choice,
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PageExtractionResult,
                        temperature=0.1,
                    )
                )

                extracted_data = response.parsed or PageExtractionResult.model_validate_json(response.text)
                st.success(f"Extrakce dokončena! Nalezeno {len(extracted_data.nodes)} prvků a {len(extracted_data.edges)} propojení.")

                net = Network(height="500px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")

                for node in extracted_data.nodes:
                    node_color = "#4CAF50" if node.category == "equipment" else "#2196F3" if node.category == "instrument" else "#FF9800"
                    net.add_node(node.id, label=f"{node.id}\n({node.type})", title=node.description, color=node_color)

                for edge in extracted_data.edges:
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
                        mime="application/json",
                        use_container_width=True
                    )

                with col2:
                    cypher_lines = [f"CREATE (:Component {{id: '{n.id}', type: '{n.type}'}});" for n in extracted_data.nodes]
                    cypher_lines += [
                        f"MATCH (a:Component {{id: '{e.source}'}}), (b:Component {{id: '{e.target}'}}) CREATE (a)-[:{e.relation_type.upper()}]->(b);"
                        for e in extracted_data.edges
                    ]
                    st.download_button(
                        label="Stáhnout Cypher",
                        data="\n".join(cypher_lines),
                        file_name="pid_graph.cypher",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Chyba při zpracování: {str(e)}")
