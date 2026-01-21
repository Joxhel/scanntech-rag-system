import torch
import os

class Config:
    # --- RUTAS DE SISTEMA ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db_storage")
    EVAL_DIR = os.path.join(BASE_DIR, "eval")
    
    PDF_PATH = os.path.join(DATA_DIR, "PDF-GenAI-Challenge (1).pdf")
    GT_PATH = os.path.join(EVAL_DIR, "benchmark", "ground_truth.json")
    LOG_PATH = os.path.join(EVAL_DIR, "logs", "interactions.jsonl")
    REPORTS_DIR = os.path.join(EVAL_DIR, "reports")
    MASTER_REPORT = os.path.join(REPORTS_DIR, "master_benchmark.csv")
    
    # --- ESTRATEGIA DE MODELOS (Industry Standard) ---
    # Modelo para Inferencia (Velocidad y Eficiencia)
    # Usamos Llama 3.2 3B para minimizar latencia en el chat
    RAG_LLM = "llama3.2:3b" 
    
    # Modelo para Evaluación (Razonamiento Superior - Juez)
    # Usamos Llama 3.1 8B como juez local por su superior capacidad de seguimiento de instrucciones
    # En entornos cloud, aquí se escalaría a Llama 3.1 70B o modelos GPT-4 class.
    JUDGE_LLM = "llama3.1:8b" 
    
    EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # --- PARÁMETROS RAG ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def init_workspace(cls):
        """Crea la estructura de carpetas necesaria para el proyecto."""
        folders = [cls.DATA_DIR, cls.DB_DIR, cls.REPORTS_DIR, 
                   os.path.dirname(cls.LOG_PATH), os.path.dirname(cls.GT_PATH)]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)