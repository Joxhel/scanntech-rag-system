import streamlit as st
import pandas as pd
import os
from src.query_rag import RAGSystem
from src.config import Config
REPO_NAME = "scanntech-rag-system"
# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="ISLR Expert Assistant",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def get_rag_engine():
    # Inicializa el motor RAG una sola vez y lo guarda en caché
    return RAGSystem()

try:
    rag_engine = get_rag_engine()
except Exception as e:
    st.error(f"Error initializing RAG Engine: {e}")
    st.stop()

# --- SIDEBAR: OBSERVABILIDAD Y MÉTRICAS ---
with st.sidebar:
    st.title("🛡️ Observability")
    st.markdown("### System Performance")
    
    # Intenta cargar las métricas desde el archivo CSV generado por RAGAS
    if os.path.exists(Config.MASTER_REPORT):
        df = pd.read_csv(Config.MASTER_REPORT)
        
        # Layout de métricas en 2 columnas para ahorrar espacio
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Faithfulness", f"{df['faithfulness'].mean():.2f}")
            st.metric("Context Prec.", f"{df['context_precision'].mean():.2f}")
        with col2:
            st.metric("Answer Rel.", f"{df['answer_relevancy'].mean():.2f}")
            st.metric("Context Recall", f"{df['context_recall'].mean():.2f}")
            
        st.success("RAGAS Metrics updated.")
    else:
        st.warning("⚠️ Benchmark not detected. Run Option 2 in `main.py` to generate metrics.")

    st.divider()

    # Sección de características clave para resaltar el valor técnico
    st.markdown("### 🛠️ Key Features")
    st.markdown("""
    - **Dense Retrieval**: Vector search via ChromaDB.
    - **Semantic Re-ranking**: Cross-encoder optimization (MS-MARCO).
    - **XAI**: Full context transparency & XML wrapping.
    - **Automated Eval**: RAGAS + Local Model.
    """)

    st.divider()
    st.markdown("### 🔗 References")
    st.info("Technical Assessment: AI Engineer - Scanntech")
    
    # Enlaces de contacto
    st.markdown("[👤 Jose Luis Cabrera Vega](https://www.linkedin.com/in/josecabrerav)")
    st.markdown("[📂 Project Repository](https://github.com/Joxhel)")
    
    st.divider()
    # Muestra los modelos utilizados desde la configuración
    st.markdown("### 🤖 Model Stack")
    st.caption(f"**Generation (RAG):** {Config.RAG_LLM}")
    st.caption(f"**Embeddings:** {Config.EMBED_MODEL}")
    st.caption(f"**Evaluation (Judge):** {Config.JUDGE_LLM}")

# --- INTERFAZ DE CHAT ---
st.title("📚 ISLR Technical Assistant")
st.caption("Advanced Architecture: RAG + Semantic Re-ranking + RAGAS Monitoring")

# Inicialización del historial del chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderiza los mensajes previos guardados en el historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura de la entrada del usuario
if prompt := st.chat_input("Ask about Statistics, SVM, Regression, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Spinner para indicar procesamiento de IA
        with st.spinner("Retrieving technical knowledge and applying re-ranking..."):
            result = rag_engine.query(prompt)
            st.markdown(result["answer"])
            
            # Expander para la transparencia del contexto (Explicabilidad - XAI)
            with st.expander("🔍 View Retrieved Context (Post Re-ranking)"):
                for idx, ctx in enumerate(result["contexts"]):
                    st.markdown(f"**Chunk {idx+1}**")
                    st.info(ctx)

    # Guarda la respuesta en el historial
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})