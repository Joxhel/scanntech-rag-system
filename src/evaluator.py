import os
import json
import pandas as pd
import asyncio
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from src.query_rag import RAGSystem
from src.config import Config

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def sanitize_for_eval(text):
    """
    Transforma texto técnico complejo en texto plano limpio.
    Es más eficiente que BeautifulSoup para Markdown/LaTeX ya que evita la 
    construcción de un árbol DOM innecesario para datos no-HTML.
    """
    if not text:
        return ""
    
    # 1. Normalización de fórmulas LaTeX: Reemplaza bloques y fórmulas en línea
    # Se usa un marcador [MATH] para que el Juez sepa que hay lógica matemática presente
    text = re.sub(r'\$\$.*?\$\$', ' [MATH_BLOCK] ', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', ' [MATH] ', text)
    
    # 2. Limpieza de Markdown: Eliminar negritas, cursivas, código y bloques de código
    text = re.sub(r'```.*?```', ' [CODE_BLOCK] ', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', ' [CODE] ', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    
    # 3. Limpieza de estructura: Encabezados y enlaces
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # 4. Eliminación de citaciones de página para evitar que el Juez las cuente como "hechos"
    text = re.sub(r'\[Page \d+\]', '', text)
    
    # 5. Normalización final de espacios
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class CleanChatOllama(ChatOllama):
    """
    Wrapper para asegurar que la salida del Juez sea JSON puro y legible.
    """
    def __init__(self, *args, **kwargs):
        kwargs["format"] = "json"
        super().__init__(*args, **kwargs)

    def _generate(self, *args, **kwargs):
        res = super()._generate(*args, **kwargs)
        for gen in res.generations:
            text = gen.text.strip()
            try:
                # Extracción de seguridad del bloque JSON
                start, end = text.find("{"), text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start:end+1]
                
                # Limpieza de caracteres de escape problemáticos para el parseador de Python
                text = text.replace('\\n', ' ').replace('\\r', ' ')
                text = re.sub(r'[^\x00-\x7F]+', ' ', text)
                json.loads(text)
            except:
                pass 
            gen.text = text
        return res

class RAGEvaluator:
    def __init__(self):
        Config.init_workspace()
        self.llm_judge = CleanChatOllama(model=Config.JUDGE_LLM, temperature=0)
        
        self.embed_judge = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={
                'device': Config.DEVICE,
                'trust_remote_code': True
            },
            encode_kwargs={'normalize_embeddings': True}
        )

    def run_master_benchmark(self, force=False):
        """
        Ejecuta el Benchmark Maestro sanitizando las entradas. 
        Este proceso previene que el Juez se distraiga con la sintaxis técnica.
        """
        print(f"Iniciando Benchmark Maestro (Modo: Texto Plano Optimizado)")
        
        if not os.path.exists(Config.GT_PATH):
            print(f"Error: Ground Truth no encontrado.")
            return

        with open(Config.GT_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        rag = RAGSystem()
        samples = []
        
        for item in tqdm(gt_data, desc="Procesando consultas RAG", unit="preg"):
            res = rag.query(item['question'])
            
            # Sanitización de la respuesta generada y de la verdad de referencia
            clean_response = sanitize_for_eval(res["answer"])
            clean_reference = sanitize_for_eval(item['ground_truth'])
            
            samples.append({
                "user_input": item['question'],
                "response": clean_response,
                "retrieved_contexts": res["contexts"],
                "reference": clean_reference
            })

        print("\n📈 Evaluando métricas sobre datos sanitizados...")
        dataset = EvaluationDataset.from_list(samples)
        
        metrics = [
            Faithfulness(llm=self.llm_judge), 
            AnswerRelevancy(llm=self.llm_judge, embeddings=self.embed_judge),
            ContextPrecision(llm=self.llm_judge), 
            ContextRecall(llm=self.llm_judge)
        ]
        
        results = evaluate(
            dataset=dataset, 
            metrics=metrics, 
            run_config=RunConfig(max_workers=1, timeout=300)
        )
        
        df = results.to_pandas()
        df.to_csv(Config.MASTER_REPORT, index=False)
        self._show_report(df)

    def _show_report(self, df):
        cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        avg = df[cols].mean(skipna=True).to_frame().T
        
        print(f"\n{'='*65}")
        print(f"📊 REPORTE DE CALIDAD (SANITIZADO)")
        print(f"{'='*65}")
        print(tabulate(avg, headers='keys', tablefmt='psql', showindex=False))
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.barplot(data=avg, palette="rocket")
        plt.ylim(0, 1)
        plt.title(f"RAG Quality Performance (Sanitized) - Judge: {Config.JUDGE_LLM}")
        
        report_img = os.path.join(Config.REPORTS_DIR, "benchmark_sanitized.png")
        plt.savefig(report_img)
        print(f"\nGráfico generado en: {report_img}")

if __name__ == "__main__":
    RAGEvaluator().run_master_benchmark()