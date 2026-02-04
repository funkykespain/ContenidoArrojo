import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# --- CONFIGURACIÓN VISUAL ---
# Códigos ANSI para dar color a los mensajes en la terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_status(component, status, message=""):
    """Función auxiliar para imprimir mensajes de estado formateados."""
    color = GREEN if status == "OK" else RED
    print(f"[{component}]... {color}{status}{RESET} {message}")

print(f"{YELLOW}--- DIAGNÓSTICO DE SISTEMA: ARROJO CONTENT GENERATOR ---{RESET}\n")

# --- 1. CARGA DE VARIABLES Y CONFIGURACIÓN ---
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

# Validamos y limpiamos la URL de Qdrant para evitar errores de formato
if qdrant_url and qdrant_url.endswith("/"):
    qdrant_url = qdrant_url[:-1]
    print(f"{YELLOW}Aviso: Se eliminó la barra final '/' de QDRANT_URL para normalizar la dirección.{RESET}")

print(f"URL Objetivo: {qdrant_url}")

# --- 2. PRUEBA DE CONEXIÓN A QDRANT ---
print(f"\n{YELLOW}--- VERIFICANDO CONEXIÓN A BASE DE DATOS VECTORIAL ---{RESET}")

try:
    # Configuramos el cliente explícitamente para HTTPS (Puerto 443)
    # Esto asegura la conexión segura con Easypanel.
    client = QdrantClient(
        url=qdrant_url,
        port=443,        
        https=True,      
        api_key=qdrant_key if qdrant_key else None,
        timeout=10
    )
    
    # Intentamos listar las colecciones para confirmar que la autenticación funciona
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    
    # Verificamos si existe nuestra colección principal 'arrojo'
    if "arrojo" in names:
        print_status("CONEXIÓN", "OK", f"¡Conexión establecida! Colecciones encontradas: {names}")
    else:
        print_status("CONEXIÓN", "WARNING", f"Conectado, pero NO encontramos la colección 'arrojo'. Disponibles: {names}")

except Exception as e:
    print_status("CONEXIÓN", "ERROR", f"Fallo crítico al conectar: {str(e)}")

# --- 3. PRUEBA DE FLUJO RAG (EMBEDDINGS + BÚSQUEDA) ---
print(f"\n{YELLOW}--- VERIFICANDO GENERACIÓN Y BÚSQUEDA VECTORIAL ---{RESET}")

try:
    # Inicializamos el modelo de Embeddings (Qwen)
    # Es vital que sea el mismo modelo usado para subir los datos.
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        openai_api_key=api_key,
        openai_api_base=os.getenv("OPENROUTER_BASE_URL")
    )
    
    print("Generando vector de prueba (esto verifica la conexión con OpenRouter)...")
    query_vector = embeddings.embed_query("test de diagnóstico")
    print(f"Vector generado correctamente. Dimensión: {len(query_vector)}")

    # Ejecutamos una búsqueda real en Qdrant para validar que devuelve datos
    hits = client.search(
        collection_name="arrojo",
        query_vector=query_vector,
        limit=1
    )
    
    if hits:
        print_status("RAG", "OK", f"¡Búsqueda exitosa! Score de similitud: {hits[0].score}")
        print(f"Datos recuperados (Payload): {hits[0].payload}")
    else:
        print_status("RAG", "WARNING", "La búsqueda funcionó, pero la colección parece estar vacía (0 resultados).")

except AttributeError:
    print_status("RAG", "ERROR", "Error de compatibilidad. Posible librería desactualizada.")
except Exception as e:
    print_status("RAG", "ERROR", f"Fallo durante la prueba de búsqueda: {str(e)}")