# Arrojo Content Generator (AI Agent)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Orchestrator-LangChain-green)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)](https://www.docker.com/)

Herramienta interna de IA desarrollada para la banda de rock **Arrojo**. Este agente actúa como Community Manager y Copywriter, generando contenido optimizado para redes sociales (Instagram, TikTok, YouTube, WhatsApp) manteniendo una identidad de marca consistente ("Estilo Arrojero").

Utiliza **RAG (Retrieval-Augmented Generation)** para consultar una base de conocimiento vectorial y datos en tiempo real (agenda de conciertos), garantizando que la información generada sea verídica y actualizada.

![Interfaz de Usuario](preview.png)
*(Interfaz construida con Streamlit simulando la estética corporativa de la banda)*

## 🚀 Funcionalidades Clave

* **🧠 Identidad de Marca Persistente:** Prompt del sistema diseñado para adherirse estrictamente al tono de voz de la banda (cercano, rockero, uso específico de emojis).
* **📚 RAG (Base de Conocimiento):** Conexión con **Qdrant** para recuperar biografía, discografía y letras de canciones.
* **📅 Datos en Tiempo Real:** Ingesta de CSV en vivo (Google Sheets) para consultar fechas de conciertos pasados y futuros.
* **🎨 UI Personalizada:** Interfaz en **Streamlit** con inyección de CSS para replicar el branding oficial de ArrojoRock.es.
* **📱 Multi-Plataforma:** Generación de estructuras JSON específicas para cada red social (hashtags, longitud, formato).

## 🛠️ Stack Tecnológico

* **Frontend:** Streamlit (Python).
* **Orquestación:** LangChain.
* **LLM:** Meta Llama 3.3 70B (vía OpenRouter) para razonamiento complejo en español.
* **Vector Store:** Qdrant (Base de datos vectorial).
* **Embeddings:** Qwen3-8B.
* **Despliegue:** Docker en VPS.

## ⚙️ Configuración Local

Si quieres clonar y probar este proyecto (necesitarás tus propias API Keys):

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/funkykespain/ContenidoArrojo.git](https://github.com/funkykespain/ContenidoArrojo.git)
    cd ContenidoArrojo
    ```

2.  **Crear entorno virtual e instalar dependencias:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configurar variables de entorno:**
    Crea un archivo `.env` en la raíz y añade:
    ```ini
    OPENROUTER_API_KEY="sk-..."
    OPENROUTER_BASE_URL="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)"
    QDRANT_URL="[https://tu-instancia-qdrant.com](https://tu-instancia-qdrant.com)"
    QDRANT_API_KEY="tu-key"
    QDRANT_COLLECTION="nombre-coleccion"
    AGENDA_CONCIERTOS="url-csv-publico"
    ```

4.  **Ejecutar:**
    ```bash
    streamlit run app.py
    ```

## 📂 Estructura del Proyecto

```text
├── app.py              # Lógica principal (Frontend + Backend LangChain)
├── Dockerfile          # Configuración para despliegue en contenedores
├── requirements.txt    # Dependencias de Python
└── .env                # Variables de entorno (No incluido en repo)

```

## 🛡️ Licencia

Este proyecto es una herramienta interna para [Arrojo](https://arrojorock.es). El código es libre para fines educativos.

```
