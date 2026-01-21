import json
import os
from datetime import datetime
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from src.config import Config

class RAGSystem:
    def __init__(self):
        # 1. Configuración de Componentes de Recuperación
        # Se asegura el uso de trust_remote_code para compatibilidad con modelos de HuggingFace
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={'device': Config.DEVICE, 'trust_remote_code': True}
        )
        self.vectorstore = Chroma(
            persist_directory=Config.DB_DIR, 
            embedding_function=self.embeddings
        )
        # El Cross-Encoder actúa como el filtro de calidad semántica definitivo
        self.reranker = CrossEncoder(Config.RERANK_MODEL, device=Config.DEVICE)
        
        # 2. Motor de Inferencia (Configurado con Temperatura 0 para fidelidad técnica)
        self.llm = OllamaLLM(model=Config.RAG_LLM, temperature=0)
        
        # 3. Prompt de Grado Científico con Cadena de Verificación (CoV)
        template = """
            <SYSTEM_ROLE>
            You are a Senior AI Scientist and pedagogical expert specializing in the textbook "An Introduction to Statistical Learning" (ISL). Your goal is to explain complex statistical concepts with technical precision yet accessible clarity, maintaining absolute academic rigor.
            </SYSTEM_ROLE>
            
            <CONTEXT_STREAMS>
            {context}
            </CONTEXT_STREAMS>
            
            <THOUGHT_PROCESS>
            Before generating the technical response, perform these mental steps:
            1. Identify the core concepts in the User Question.
            2. Scan the <CONTEXT_STREAMS> for these specific concepts across Chapters, Subchapters, and Sections.
            3. Verify if the information exists in the provided fragments.
            4. If a fact is NOT explicitly stated in the XML, mark it as "Not Found" internally and do not use it.
            </THOUGHT_PROCESS>
            
            <STRICT_CONSTRAINTS>
            1. ABSOLUTE SOURCE ADHERENCE: Use ONLY information provided in <CONTEXT_STREAMS>. 
            2. ZERO TOLERANCE: Do not rely on external knowledge or pre-trained data. If it is not in the XML, it does not exist for this response.
            3. UNCERTAINTY PROTOCOL: If the answer is not contained within the context, respond exactly: "I apologize, but the requested information is not available in the retrieved fragments of the book."
            4. MANDATORY CITATIONS: You MUST cite the specific page number for every claim using the format [Page X] at the end of each paragraph.
            5. MATHEMATICAL NOTATION: Render all statistical formulas and equations using LaTeX ($ for inline, $$ for blocks). 
            6. NO LATEX FOR TEXT: Do NOT use LaTeX commands like \\text{{}}, \\tag{{}}, or \\mathrm{{}} for regular prose. Use standard Markdown for text formatting.
            7. LANGUAGE CONSISTENCY: Respond in the same language as the User Question.
            </STRICT_CONSTRAINTS>
            
            USER QUESTION: {question}
            
            Technical Response:"""
        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, query):
        """
        Ejecuta el pipeline RAG optimizado: Recuperación -> Re-ranking -> Generación Jerárquica.
        """
        # A. Recuperación Vectorial Inicial (Fase 1: K=15 para amplitud semántica)
        initial_docs = self.vectorstore.similarity_search(query, k=15)
        
        # B. Re-ranking Semántico (Fase 2: Filtro de precisión)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        for i, doc in enumerate(initial_docs):
            doc.metadata["score"] = float(scores[i])
            
        # Filtro estricto: Umbral de -3.5 para garantizar relevancia contextual
        # Esto reduce drásticamente las alucinaciones por "ruido" en los fragmentos.
        final_docs = [d for d in sorted(initial_docs, key=lambda x: x.metadata["score"], reverse=True) 
                      if d.metadata["score"] > -3.5][:5]

        if not final_docs:
            return {
                "answer": "I apologize, but the requested information is not available in the retrieved fragments of the book.", 
                "contexts": []
            }

        # C. Construcción del Contexto con Jerarquía Completa (XML Enriquecido)
        # Se inyecta la traza completa: Página, Capítulo, Subcapítulo y Sección.
        context_str = "\n".join([
            f"<DOCUMENT "
            f"page='{d.metadata.get('physical_page', d.metadata.get('page', 'N/A'))}' "
            f"chapter='{d.metadata.get('chapter', 'N/A')}' "
            f"subchapter='{d.metadata.get('subchapter', 'N/A')}' "
            f"section='{d.metadata.get('section', 'N/A')}'>\n"
            f"{d.page_content}\n"
            f"</DOCUMENT>" 
            for d in final_docs
        ])

        # D. Generación de Respuesta Controlada
        response = self.chain.invoke({"context": context_str, "question": query})
        
        # E. Registro de Auditoría para RAGAS
        self._log(query, response, final_docs)
        
        return {
            "answer": response,
            "contexts": [f"Pag {d.metadata.get('physical_page', d.metadata.get('page', 'N/A'))}: {d.page_content[:200]}..." for d in final_docs]
        }

    def _log(self, q, a, docs):
        """Almacena la traza de la consulta para análisis de fidelidad."""
        entry = {
            "timestamp": datetime.now().isoformat(), 
            "question": q, 
            "answer": a, 
            "contexts": [d.page_content for d in docs]
        }
        with open(Config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")