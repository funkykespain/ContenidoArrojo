import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Colores
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_status(component, status, message=""):
    color = GREEN if status == "OK" else RED
    print(f"[{component}]... {color}{status}{RESET} {message}")

print(f"{YELLOW}--- DIAGNÓSTICO ARROJO V2 ---{RESET}\n")

# 1. CARGA DE ENTORNO
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

# Limpieza preventiva de URL
if qdrant_url and qdrant_url.endswith("/"):
    qdrant_url = qdrant_url[:-1]
    print(f"{YELLOW}Aviso: Se eliminó la barra final '/' de QDRANT_URL para evitar errores.{RESET}")

print(f"URL Objetivo: {qdrant_url}")


# 2. PRUEBA DE CONEXIÓN QDRANT (HTTPS EXPLÍCITO)
print(f"\n{YELLOW}--- PROBANDO CONEXIÓN A QDRANT (MODO EASJPANEL/HTTPS) ---{RESET}")
try:
    # FORZAMOS PUERTO 443 (El estándar de HTTPS)
    client = QdrantClient(
        url=qdrant_url,
        port=443,        # <--- LA CLAVE: Forzar entrada por HTTPS estándar
        https=True,      # <--- Confirmar protocolo seguro
        api_key=qdrant_key if qdrant_key else None,
        timeout=10
    )
    
    # Prueba ligera: obtener colecciones
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    
    if "arrojo" in names:
        print_status("CONEXIÓN", "OK", f"¡Conectado por puerto 443! Colecciones: {names}")
    else:
        print_status("CONEXIÓN", "WARNING", f"Conectado, pero NO veo 'arrojo'. Disponibles: {names}")

except Exception as e:
    print_status("CONEXIÓN", "ERROR", f"Sigue fallando: {str(e)}")

# 3. PRUEBA DE BÚSQUEDA (RAG)
print(f"\n{YELLOW}--- PROBANDO BÚSQUEDA VECTORIAL ---{RESET}")
try:
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        openai_api_key=api_key,
        openai_api_base=os.getenv("OPENROUTER_BASE_URL")
    )
    
    print("Generando embedding (esto puede tardar unos segundos)...")
    query_vector = embeddings.embed_query("test")
    print(f"Vector generado. Dimensión: {len(query_vector)}")

    # Usamos search() que es el método estándar moderno
    hits = client.search(
        collection_name="arrojo",
        query_vector=query_vector,
        limit=1
    )
    
    if hits:
        print_status("RAG", "OK", f"¡ÉXITO! Documento encontrado. Score: {hits[0].score}")
        print(f"Contenido (payload): {hits[0].payload}")
    else:
        print_status("RAG", "WARNING", "Búsqueda OK, pero la colección parece estar vacía (0 resultados).")

except AttributeError:
    print_status("RAG", "ERROR", "Error de versión de librería. Ejecuta: pip install --upgrade qdrant-client")
except Exception as e:
    print_status("RAG", "ERROR", str(e))