import re
import fitz
import torch
import os
from tqdm import tqdm
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import Config
from concurrent.futures import ThreadPoolExecutor

def clean_technical_text(text):
    """
    Limpieza profunda preservando sintaxis técnica y formateo markdown.
    Protege la estructura de código y fórmulas detectadas por PyMuPDF4LLM.
    """
    # Eliminar caracteres de control y no imprimibles
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Normalización de headers (Convierte el estilo específico de PyMuPDF4LLM a Markdown estándar)
    text = re.sub(r'^_(\d+\.[\d\.]+)_ _(.*)_', r'### \1 \2', text, flags=re.MULTILINE)
    
    # Eliminar rastro de copyright y URLs DOI (ajustado para ser general)
    text = re.sub(r'©.*|https?://doi\.org/\S+', '', text, flags=re.IGNORECASE)
    
    # Corregir saltos de línea que rompen palabras (hyphenation)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Limpiar secuencias de puntos (comunes en tablas de contenido extraídas como texto)
    text = re.sub(r'\.{3,}', '', text)
    
    # Normalización de espacios (preservando indentaciones simples)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()

def get_excluded_pages(toc_data):
    """
    Identifica rangos de páginas a omitir (índices, contenidos) basándose en el TOC.
    """
    excluded = set()
    for i, (level, title, start_p) in enumerate(toc_data):
        if title.lower() in ["contents", "index", "table of contents"]:
            # Buscamos el final de la sección (donde empieza la siguiente entrada del TOC)
            end_p = toc_data[i+1][2] if i+1 < len(toc_data) else start_p + 10
            for p in range(start_p, end_p + 1):
                excluded.add(p)
    return excluded

def get_hierarchy(p_num, toc_data):
    """
    Asigna contexto jerárquico a una página basándose en la posición del TOC.
    Soporta hasta 3 niveles de profundidad (Capítulo, Subcapítulo, Sección).
    """
    ctx = {"chapter": "Front Matter", "subchapter": "General", "section": "General"}
    
    for level, title, start_p in toc_data:
        # Limpiamos el título de puntos sobrantes del TOC
        clean_title = re.sub(r'\.{2,}', '', title).strip()
        
        if start_p <= p_num:
            if level == 1:
                ctx["chapter"] = clean_title
                ctx.update({"subchapter": clean_title, "section": clean_title})
            elif level == 2:
                ctx["subchapter"] = clean_title
                ctx["section"] = clean_title
            elif level == 3:
                ctx["section"] = clean_title
        else:
            # Como el TOC está ordenado, podemos romper el ciclo al superar la página
            break
    return ctx

def run_ingestion():
    Config.init_workspace()
    device = Config.DEVICE
    
    print(f"--- Iniciando Ingesta Técnica: {os.path.basename(Config.PDF_PATH)} ---")
    #LIMIT_PAGES = 20
    # 1. Análisis de Estructura (TOC)
    doc_fitz = fitz.open(Config.PDF_PATH)
    total_pages = len(doc_fitz)
    internal_toc = doc_fitz.get_toc()
    doc_fitz.close()
    pages_to_skip = get_excluded_pages(internal_toc)

    # 2. Carga con PyMuPDF4LLM (Formato enriquecido)
    loader = PyMuPDF4LLMLoader(Config.PDF_PATH)
    #raw_pages = []

    # Este bucle es el que toma el tiempo y muestra la barra
    #for page in tqdm(loader.lazy_load(), total=total_pages, desc="Cargando PDF a Markdown", unit="pág"):
    #    raw_pages.append(page)
    
    processed_docs = []
    for i, page in enumerate(tqdm(loader.lazy_load(), total=total_pages, desc="Ingestando Documento", unit="pág")):
        physical_p = i + 1 # Contador físico real
        #if LIMIT_PAGES and i >= LIMIT_PAGES:
        #    break
        # Omitir si está en el rango de exclusión
        if physical_p in pages_to_skip:
            continue
            
        content = clean_technical_text(page.page_content)
        
        # Omitir páginas con contenido irrelevante tras limpieza
        if len(content) < 150:
            continue
            
        # Enriquecimiento de Metadatos
        hierarchy = get_hierarchy(physical_p, internal_toc)
        
        # Mantenemos 'page' original del loader y añadimos 'physical_page' para trazabilidad
        page.page_content = content
        page.metadata.update(hierarchy)
        page.metadata["original_page"] = page.metadata.get("page", None)
        page.metadata["page"] = physical_p 
        

        
        processed_docs.append(page)

    # 3. Segmentación (Chunking) con preservación de contexto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, 
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(processed_docs)
    
    # 4. Refinamiento de Metadatos por Chunk
    # Ajusta el contexto si un chunk detecta el inicio de una nueva sección en la misma página
    for chunk in chunks:
        p_num = chunk.metadata.get("physical_page")
        for level, title, start_p in internal_toc:
            clean_t = re.sub(r'\.{2,}', '', title).strip()
            if start_p == p_num and clean_t in chunk.page_content:
                if level == 1: chunk.metadata["chapter"] = clean_t
                elif level == 2: chunk.metadata["subchapter"] = clean_t
                elif level == 3: chunk.metadata["section"] = clean_t

    # 5. Generación de Embeddings e Indexación
    print(f"--- Generando Embeddings ({Config.EMBED_MODEL}) ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBED_MODEL,
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma(
        embedding_function=embeddings, 
        persist_directory=Config.DB_DIR
    )
    def index_batch(batch):
        vectorstore.add_documents(batch)

    batch_size = 100
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    print(f"--- Indexando en paralelo con hilos ---")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(index_batch, batches), total=len(batches), desc="Indexando", unit="lote"))
if __name__ == "__main__":
    run_ingestion()