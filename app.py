import io
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from pyvis.network import Network
import streamlit.components.v1 as components

# Import Pydantic modelů z přeneseného balíčku pidkg
from pidkg.models import PageExtractionResult

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
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

uploaded_file = st.file_uploader(
    "Nahrajte P&ID výkres (PDF, PNG, JPG)", 
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        image = convert_pdf_to_image(uploaded_file.read())
    else:
        image = Image.open(uploaded_file)

    st.image(image, caption="Načtený výkres", use_column_width=True)

    if st.button("🔍 Analyzovat a vygenerovat graf", type="primary", use_container_width=True):
        with st.spinner("Gemini čte technické schéma a sestavuje graf prvků..."):
            try:
                prompt = """
                Jsi expert na P&ID schémata. Zanalyzuj tento výkres podle normy ISA-5.1.
                Extrahuj všechny uzly (Equipment, Valves, Instruments) a jejich propojení (Piping, Signal lines).
                U každého prvku uveď přesný TAG kód (např. PT-101, V-102, T-100).
                """

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PageExtractionResult,
                        temperature=0.1,
                    )
                )

                extracted_data: PageExtractionResult = response.parsed
                st.success("Extrakce dokončena!")

                net = Network(height="500px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")

                for node in extracted_data.nodes:
                    category = getattr(node, 'category', 'other')
                    node_color = "#4CAF50" if category == "equipment" else "#2196F3" if category == "instrument" else "#FF9800"
                    
                    label = f"{node.id}\n({getattr(node, 'type', '')})"
                    title = getattr(node, 'description', node.id)
                    net.add_node(node.id, label=label, title=title, color=node_color)

                for edge in extracted_data.edges:
                    rel_type = getattr(edge, 'relation_type', '')
                    net.add_edge(edge.source, edge.target, label=rel_type)

                net.toggle_physics(True)
                
                html_graph = net.generate_html()
                components.html(html_graph, height=520, scrolling=False)

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
                    cypher_lines = []
                    for node in extracted_data.nodes:
                        cypher_lines.append(f"CREATE (:Component {{id: '{node.id}'}});")
                    for edge in extracted_data.edges:
                        cypher_lines.append(f"MATCH (a:Component {{id: '{edge.source}'}}), (b:Component {{id: '{edge.target}'}}) CREATE (a)-[:CONNECTED_TO]->(b);")
                    
                    st.download_button(
                        label="Stáhnout Cypher",
                        data="\n".join(cypher_lines),
                        file_name="pid_graph.cypher",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Chyba při zpracování: {str(e)}")
