import os
import time
import subprocess
from src.ingestion import run_ingestion
from src.evaluator import RAGEvaluator
from src.config import Config

def clear_screen():
    # Limpia la pantalla de la consola según el sistema operativo
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Inicializa el espacio de trabajo y las carpetas necesarias
    Config.init_workspace()
    
    while True:
        clear_screen()
        print("="*65)
        print("🚀 SCANNTECH AI ENGINEER CHALLENGE - RAG ECOSYSTEM")
        print("Developed by: Jose Luis Cabrera Vega")
        print("="*65)
        print("1. 🛠️  INGESTION: Process PDF and create Vector Database")
        print("2. 📊 EVALUATION: Run Master Benchmark (RAGAS + Llama 3.1)")
        print("3. 💬 CHAT: Launch User Interface (Streamlit)")
        print("4. 🚪 Exit")
        print("-" * 65)
        
        opcion = input("Please select an option: ")

        if opcion == "1":
            print("\n[INFO] Starting Advanced Ingestion Pipeline...")
            # Ejecuta el script de procesamiento de documentos y embeddings
            run_ingestion()
            input("\nPress Enter to return to the menu...")
            
        elif opcion == "2":
            print("\n[INFO] Starting RAGAS Evaluation (This may take a few minutes)...")
            evaluator = RAGEvaluator()
            # Ejecuta el benchmark maestro para obtener métricas de fidelidad y precisión
            evaluator.run_master_benchmark()
            input("\nPress Enter to return to the menu...")
            
        elif opcion == "3":
            print("\n[INFO] Launching Streamlit Dashboard...")
            # Inicia la interfaz gráfica en un subproceso
            try:
                subprocess.run(["streamlit", "run", "app.py"])
            except KeyboardInterrupt:
                # Maneja la interrupción del teclado para volver al menú suavemente
                pass
                
        elif opcion == "4":
            print("Closing AI System. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
            time.sleep(1.5)

if __name__ == "__main__":
    main()