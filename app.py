import streamlit as st
import os
import re
from datetime import datetime
from dotenv import load_dotenv
import requests
from operator import itemgetter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# --- 0. FUNCIONES DE UTILIDAD ---

# Función para obtener datos en tiempo real (Agenda de Conciertos)
# Utilizamos caché para no saturar la API de Google Sheets en cada recarga.
@st.cache_data(ttl=3600) 
def fetch_agenda_data():
    url = os.getenv("AGENDA_CONCIERTOS")
    if not url:
        return "No hay URL de agenda configurada."
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text # Devolvemos el CSV crudo como texto
    except Exception as e:
        return f"Error leyendo agenda: {str(e)}"

# --- FUNCIÓN DE LIMPIEZA DE FORMATO (FILTRO DE SEGURIDAD) ---
def clean_format_for_platform(text, platform):
    """
    Elimina Markdown y gestiona enlaces según las restricciones técnicas de la red social.
    Esto actúa como una barrera de seguridad por si el LLM ignora las instrucciones.
    """
    # 1. Definir si la plataforma soporta Markdown (Solo WhatsApp lo soporta bien)
    supports_markdown = "WhatsApp" in platform
    
    # 2. Definir si la plataforma soporta enlaces clicables en el cuerpo del texto
    clickable_links = any(p in platform for p in ["Facebook", "WhatsApp", "YouTube", "Twitter", "X"])

    clean_text = text

    # --- LIMPIEZA DE ESTILOS (Negrita, Cursiva) ---
    if not supports_markdown:
        # Eliminar negritas (**texto** o __texto__) -> texto
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
        clean_text = re.sub(r'__(.*?)__', r'\1', clean_text)
        # Eliminar cursivas (*texto* o _texto_) -> texto
        clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
        clean_text = re.sub(r'_(.*?)_', r'\1', clean_text)
        # Eliminar encabezados Markdown (# Titulo)
        clean_text = re.sub(r'^#+\s+', '', clean_text, flags=re.MULTILINE)

    # --- GESTIÓN INTELIGENTE DE ENLACES ---
    # Patrón para encontrar enlaces markdown: [Texto Ancla](URL)
    link_pattern = r'\[(.*?)\]\((.*?)\)'
    
    def link_replacer(match):
        anchor_text = match.group(1)
        url = match.group(2)
        
        if clickable_links:
            # En Facebook/YouTube: Convertimos "[Entradas](url)" en "Entradas (url)"
            # O si prefieres solo la URL: return f"{anchor_text}: {url}"
            return f"{anchor_text} ({url})"
        else:
            # En Instagram/TikTok: La URL no sirve de nada. 
            # Convertimos "[Entradas](url)" en "Entradas".
            return anchor_text

    clean_text = re.sub(link_pattern, link_replacer, clean_text)

    return clean_text

# --- LÓGICA DE OPTIMIZACIÓN (CON REGLAS DE FORMATO) ---
def get_optimization_instruction(platform, media_type):
    """
    Devuelve la instrucción técnica específica basada en la combinación
    de Plataforma y Tipo de Medio seleccionados.
    """
    
    # Diccionario de reglas basado en documentación técnica 2026
    # Clave compuesta: "PLATAFORMA|TIPO_MEDIO"
    # Usamos '|' como separador para evitar conflictos.

    instructions = {
        # CASO: Instagram (Feed) + Carrusel
        "Instagram (Feed)|Carrusel": """
        OPTIMIZACIÓN: Estructura de Carrusel Educativo.
        * Formato técnico: PROHIBIDO usar Markdown (**negrita**). Usa MAYÚSCULAS para resaltar.
        * Enlaces: NO pongas URLs. Escribe "Link en Bio" o "Comenta FUEGO".
        * Campo 'hashtags': Usa 5-10 hashtags clásicos (#Rock #Musica).
        * Objetivo: Maximizar 'Guardados' (Saves).
        * Estructura: Genera texto para 8-10 diapositivas secuenciales.
        * Slide 1: Gancho visual de alto contraste (<10 palabras).
        * Cuerpo: Una idea por slide. Usa listas y síntesis.
        * Slide Final: CTA explícito para GUARDAR el post.
        * Caption: Estilo micro-blogging. Primera frase debe ser un gancho SEO.
        """,

        # CASO: Instagram (Stories) - Vídeo/Foto
        "Instagram (Stories)|Vídeo": """
        OPTIMIZACIÓN: Retención y Fidelización.
        * Tono: Auténtico, 'crudo' y conversacional.
        * Interacción: DEBES sugerir explícitamente qué Sticker usar (Encuesta, Caja de Preguntas, Tu Turno).
        * Duración/Texto: Breve, directo.
        * Campo 'hashtags': DEBE ESTAR VACÍO (cadena vacía ""). No uses hashtags en stories.
        * Formato: Texto plano.
        * Enlaces: NO escribas la URL. Indica "Usa el Sticker de Enlace".
        * Objetivo: Generar respuesta directa (DM) o toque en sticker.
        """,
        "Instagram (Stories)|Foto": """
        OPTIMIZACIÓN: Retención y Fidelización.
        * Tono: Auténtico, 'crudo' y conversacional.
        * Interacción: DEBES sugerir explícitamente qué Sticker usar (Encuesta, Caja de Preguntas, Tu Turno).
        * Duración/Texto: Breve, directo.
        * Campo 'hashtags': DEBE ESTAR VACÍO (cadena vacía ""). No uses hashtags en stories.
        * Formato: Texto plano.
        * Enlaces: NO escribas la URL. Indica "Usa el Sticker de Enlace".
        * Objetivo: Generar respuesta directa (DM) o toque en sticker.
        """,
        
        # CASO: Instagram (Feed) - Genérico (Si existiera vídeo en feed)
        "Instagram (Feed)|Vídeo": """
        OPTIMIZACIÓN: Reels / Feed Video.
        * Formato: Texto plano estricto (Sin negritas). Usa Emojis y SALTOS DE LÍNEA.
        * Enlaces: PROHIBIDO poner URLs en el texto. Usa "Link en la Bio".
        * CTA: Pide que visiten el perfil.
        * Campo 'hashtags': Usa hashtags mixtos (Nicho + Amplios) con #.
        """,

        # CASO: TikTok + Vídeo
        "TikTok|Vídeo": """
        OPTIMIZACIÓN: Motor de Búsqueda y Retención (SEO + Watch Time).
        * Gancho: Escribe un gancho (visual/auditivo) para los primeros 2 segundos. Debe ser disruptivo.
        * Texto en Pantalla: Sugiere keywords para poner sobre el vídeo (para el OCR de TikTok).
        * Campo 'hashtags': Usa la regla 3-3-3 (3 amplios, 3 nicho, 3 específicos o #Nicho #Viral #Marca).
        * Formato: Texto plano estricto.
        * Enlaces: NO pongas URLs. "Link en perfil".
        * SEO: La descripción debe actuar como meta-data. Incluye palabras clave long-tail naturales en el texto.
        """,

        # CASO: Facebook + Vídeo
        "Facebook|Vídeo": """
        OPTIMIZACIÓN: Discovery Engine.
        * Formato: Tratamiento de Reel unificado en Texto plano.
        * Narrativa: Estructura de historia completa (Inicio-Nudo-Desenlace) para retener +90 segundos.
        * Tono: Más universal/emocional, menos jerga Gen Z.
        * Enlaces: SÍ puedes poner URLs completas al final del post (son clicables).
        * Campo 'hashtags': Máximo 1 (#ArrojoRock) o ninguno. Facebook penaliza el exceso.
        """,

        # CASO: YouTube Shorts + Vídeo
        "YouTube (Shorts)|Vídeo": """
        OPTIMIZACIÓN: Tráfico y Suscripción.
        * Loop: El guion debe terminar de forma que enlace con el principio (Loop perfecto).
        * CTA: Enfocado a 'Suscribirse' o 'Ver vídeo relacionado'.
        * SEO: Título de <60 caracteres cargado de intención de búsqueda.
        * Formato: Texto plano.
        * Enlaces: NO en el título. Ponlos en comentario fijado o descripción.
        * Campo 'hashtags': Palabras clave (Tags) separadas por COMAS sin almohadilla (concierto, rock, musica en vivo, rock español, banda emergente, madrid). NO uses #.
        """,

        # CASO: YouTube (Video) + Vídeo
        "YouTube (Video)|Vídeo": """
        OPTIMIZACIÓN: SEO y Key Moments.
        * Estructura: Divide el guion en 'Capítulos' claros con marcas de tiempo sugeridas.
        * Descripción: Primeros 150 caracteres con la keyword principal.
        * Título: Optimizado para CTR (Click Through Rate).
        * Formato: Texto plano.
        * Enlaces: NO en el título. Ponlos en comentario fijado o descripción.
        * Campo 'hashtags': Palabras clave (Tags) separadas por COMAS sin almohadilla (concierto, rock, musica en vivo, rock español, banda emergente, madrid). NO uses #.
        """,

        # CASO: WhatsApp Channel + Solo Texto
        "WhatsApp Channel|Solo Texto": """
        OPTIMIZACIÓN: Boletín de Alta Fricción.
        * Longitud: ESTRICTAMENTE menos de 500 caracteres.
        * Interacción: Pide reacción con Emojis específicos (ej: 'Pulsa 🔥').
        * Prohibido: No usar hashtags. No pedir comentarios (es unidireccional).
        * Formato: USA Markdown de WhatsApp (*negrita* para títulos, _cursiva_).
        * Enlaces: URLs completas y clicables.
        * Campo 'hashtags': DEBE ESTAR VACÍO (cadena vacía ""). WhatsApp no usa etiquetas.
        """
    }

    # Construir clave de búsqueda
    key = f"{platform}|{media_type}"
    
    # Retornar instrucción específica o un fallback genérico si la combinación no tiene regla estricta
    return instructions.get(key, f"""
    OPTIMIZACIÓN: Estándar para {platform}.
    * FORMATO: Adaptado a {media_type}. Si es Instagram/TikTok -> SOLO TEXTO PLANO (Sin negritas). Si es WhatsApp -> Markdown OK.
    * ENLACES: Si es Instagram/TikTok -> "Link en Bio". Si es Facebook/YT -> URL completa al final.
    * Objetivo: Maximizar engagement según las mejores prácticas generales de la plataforma.
    * CTA: Claro y directo.
    * Campo 'hashtags': Si es YouTube -> Keywords separadas por comas. Si es WhatsApp/Stories -> Dejar vacío. Resto -> Hashtags con #.
    """)

# --- 1. CONFIGURACIÓN INICIAL DEL PROYECTO ---
load_dotenv()

# URLs de los activos de marca (Logos oficiales)
LOGO_URL_LARGE = "https://arrojorock.es/android-chrome-192x192.png"
LOGO_URL_SMALL = "https://arrojorock.es/favicon-32x32.png"

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Arrojo Content Generator",
    page_icon=LOGO_URL_SMALL,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ESTILOS VISUALES (CSS INYECTADO) ---
# Adaptamos la interfaz de Streamlit para que coincida con el branding de ArrojoRock.es
st.markdown("""
<style>
    /* 1. IMPORTAR FUENTES Y ICONOS */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&family=Roboto:wght@300;400;500&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    /* 2. VARIABLES DE DISEÑO (Extraídas de Stitch) */
    :root {
        --primary: #ed1c24;
        --primary-hover: #c41219;
        --bg-dark: #0a0a0a;
        --surface-dark: #121212;
        --border-dark: #2d2d2d;
        --input-bg: #1a1a1a;
        --text-muted: #9ca3af;
    }

    /* 3. ESTILO GLOBAL */
    .stApp {
        background-color: var(--bg-dark);
        font-family: 'Roboto', sans-serif;
    }

    /* --- TIPOGRAFÍA Y JERARQUÍA RESPONSIVE --- */
    
    /* Estilo base para todos los títulos */
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.2 !important;
    }

    /* ESCALADO FLUIDO (Clamp) para el cuerpo principal */
    h1 { font-size: clamp(1.5rem, 5vw, 2.5rem) !important; }
    h2 { font-size: clamp(1.3rem, 4vw, 2rem) !important; } 
    h3 { font-size: clamp(1.1rem, 3.5vw, 1.6rem) !important; }

    /* --- CORRECCIÓN ESPECÍFICA SIDEBAR --- */
    /* Evita que "Configuración" se rompa en PC porque el sidebar es estrecho */
    section[data-testid="stSidebar"] h2 {
        font-size: 1.3rem !important; /* Tamaño más pequeño y seguro */
        white-space: nowrap; /* Obliga a mantenerse en una línea */
    }

    /* AJUSTE LOGO MÓVIL */
    @media (max-width: 480px) {
        img[alt="Logo Arrojo"] {
            width: 40px !important;
            height: 40px !important;
        }
        div[style*="display: flex"] {
            gap: 10px !important;
        }
    }

    /* 4. BARRA LATERAL (Aside en Stitch) */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid var(--border-dark);
        padding-top: 2rem;
    }

    /* 5. INPUTS Y SELECTS (Estilo Tailwind Dark) */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-dark) !important;
        border-radius: 6px !important;
        color: white !important;
        transition: all 0.2s ease;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }

    /* 6. BOTÓN GENERAR (El estilo "Fire" de Stitch) */
    div.stButton > button {
        background-color: #000000 !important;
        color: #eeeeee !important;
        border: 1px solid #4b5563 !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 1rem;
    }

    div.stButton > button:hover {
        border-color: var(--primary) !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(237, 28, 36, 0.3) !important;
        transform: translateY(-2px);
    }

    /* 7. TARJETAS DE CONTENIDO (Secciones en Stitch) */
    div[data-testid="stVerticalBlock"] > div.stVerticalBlock {
        gap: 1.5rem;
    }
    
    /* Efecto de línea superior roja que vimos en Stitch */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
        z-index: 999;
        opacity: 0.6;
    }

    /* SLIDERS PERSONALIZADOS */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--primary) !important;
        border: 2px solid white !important;
    }
            
    /* 8. FOOTER FIJADO AL FONDO */
    
    /* A. Ajustamos el contenido para que tenga espacio al final */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-bottom: 100px !important; /* Espacio para que el footer no tape nada */
    }

    /* B. Estilamos el footer para que flote sobre todo lo demás */
    .sidebar-footer {
        position: fixed; /* Se fija a la ventana/sidebar */
        bottom: 0;
        left: 0;
        width: 100%; /* Ocupará el ancho del sidebar padre */
        background-color: #000000; /* Fondo negro para tapar el scroll que pase por debajo */
        padding: 15px 20px;
        z-index: 999; /* Aseguramos que esté por encima de todo */
        border-top: 1px solid #333; /* Un borde sutil para separar */
        
        /* Truco para limitarlo al ancho del sidebar si este cambia */
        width: inherit; 
    }
    
    /* C. Aseguramos que el sidebar sea el contexto de referencia */
    section[data-testid="stSidebar"] {
        position: relative;
    }

</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND (LANGCHAIN + RAG) ---

@st.cache_resource
def get_chain():
    """
    Configura y devuelve la cadena de procesamiento (Chain).
    Se usa @st.cache_resource para mantener la conexión abierta y no reconectar en cada interacción.
    """
    # --- Credenciales y Configuración ---
    
    # Básicas
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    
    # Qdrant: Conversión de tipos para evitar errores de conexión
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION")
    qdrant_https = os.getenv("QDRANT_HTTPS", "False").lower() == "true"
    qdrant_timeout = int(os.getenv("QDRANT_TIMEOUT", 60))

    # Modelos: Definición de nombres y parámetros técnicos
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    llm_model_name = os.getenv("LLM_MODEL", "mistralai/mistral-small-creative")
    
    # Parámetros del LLM: Conversión a numéricos
    llm_temp = float(os.getenv("LLM_TEMPERATURE", 0.7))
    llm_timeout = int(os.getenv("LLM_TIMEOUT", 120))
    llm_retries = int(os.getenv("LLM_MAX_RETRIES", 3))

    # A. Modelo de Embeddings
    # Debe coincidir exactamente con el usado en la ingesta de datos hecha para otro proyecto paralelo.
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=api_key,
        openai_api_base=base_url
    )

    # B. Conexión a Base de Datos Vectorial (Qdrant)
    # Configuramos el cliente con soporte HTTPS y puerto seguro.
    client = QdrantClient(
        url=qdrant_url,
        port=6333,
        https=qdrant_https,
        api_key=qdrant_key,
        timeout=qdrant_timeout
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    # El retriever buscará los 3 fragmentos más relevantes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # C. Modelo de Lenguaje (LLM)
    llm = ChatOpenAI(
        model=llm_model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=llm_temp,
        timeout=llm_timeout,
        max_retries=llm_retries,
        # Fuerza a la API a esperar un objeto JSON
        model_kwargs={"response_format": {"type": "json_object"}} 
    )

    # D. Definición de la Estructura de Salida
    # Obligamos al LLM a devolver un JSON estricto.
    class SocialPost(BaseModel):
        platform: str = Field(description="Plataforma seleccionada")
        copy_text: str = Field(description="El texto del post listo para copiar, con emojis y estructura")
        hashtags: str = Field(description="Etiquetas, Keywords (separadas por comas) o Hashtags (con #), según corresponda a la plataforma.")
        visual_suggestion: str = Field(description="Sugerencia breve para la imagen/video si no se provee")

    # Usamos JsonOutputParser en lugar de structured_llm
    # structured_llm = llm.with_structured_output(SocialPost, method="json_mode")
    # Este parser elimina automáticamente los ```json ``` si el modelo los pone.
    parser = JsonOutputParser(pydantic_object=SocialPost)

    # E. Prompt del Sistema
    # Define la voz, el tono y las reglas de negocio del agente.
    system_prompt = """
    ### CONTEXTO
    FECHA HOY: {current_date} (Usar para tiempos relativos: mañana, viernes, etc) y para saber si conciertos son pasados o futuros.
    
    ### ROL
    CM banda rock "Arrojo". TONO: {tone_modifier} + "Estilo Arrojer" (canalla, pasional, colega de bar, cercano, CERO corporativo).
    
    ### REGLAS DE ORO (NO ROMPER)
    1. REGLA DEL TÚ: SIEMPRE 2ª persona singular ("te espera"). PROHIBIDO plural ("os esperamos", "preparaos").
    2. PALABRA "ARROJERS": Máx 1 vez. NUNCA en inicio/título.
    3. EMOJIS: Máx 2-3. Solo para énfasis real.
    4. ANTI-CLICHÉ: PROHIBIDO "Noche inolvidable", "Lo vamos a romper", "Velada mágica". Sé crudo, específico y real, como el rock castizo/cañero.
    
    ### ESTRATEGIA
    Si MOTIVO="1. Concierto":
    - CTA OBLIGATORIO: Link entradas o de localización de la sala.
    - CTA CREATIVO: Sugerir escuchar temas en Spotify antes...
    
    ### OPTIMIZACIÓN PLATAFORMA ({platform} - {media_type})
    {optimization_instruction}
    
    ### FUENTES DE DATOS
    [INFO RAG]: {context}
    [AGENDA]: {agenda_context}
    
    ### LINKS DEFAULT (Usar si no hay específicos)
    Web/Entradas/Info oficial: https://arrojorock.es
    Spotify: https://open.spotify.com/artist/4s0uEp9gcIcvU1ZEsDKQXv
    YouTube: https://www.youtube.com/channel/UCJnAZC6v6OfKxNydcD6CFqQ
    
    ### TAREA
    Genera el JSON final para:
    - MOTIVO: {reason}
    - INPUT DATOS: {specific_data}
    - VISUAL: {visual_context}
    - EXTRA: {user_instructions}

    ### FORMATO DE SALIDA
    {format_instructions}
    
    Asegúrate de que el contenido del JSON cumpla estas reglas:
    1. "copy_text": Debe tener el texto completo, con saltos de línea (\n) y emojis.
    2. "hashtags": Una lista de etiquetas según se indique en la optimización de plataforma.
    3. "visual_suggestion": Descripción breve.
    4. "platform": La plataforma seleccionada.
    """

    # Inyectamos format_instructions automáticamente para reforzar la estructura
    prompt = ChatPromptTemplate.from_template(system_prompt, partial_variables={"format_instructions": parser.get_format_instructions()})

    # F. Construcción de la Cadena (Chain)
    # Generamos un string de búsqueda optimizado para RAG concatenando los inputs clave.
    # Extraemos solo los valores del diccionario, ignorando las claves y símbolos.
    rag_query_generator = (
        lambda x: (
            f"{x['reason']} "
            f"{' '.join([str(v) for v in x['specific_data'].values() if v])} "
            f"{x['user_instructions']}"
        )
    )

    chain = (
        {
            # Pasamos la query generada al retriever para buscar contexto
            "context": rag_query_generator | retriever,
            # Pasamos el resto de variables directamente
            "agenda_context": itemgetter("agenda_context"),
            "current_date": itemgetter("current_date"),
            "platform": itemgetter("platform"),
            "media_type": itemgetter("media_type"),
            # Inyectamos la instrucción calculada dinámicamente
            "optimization_instruction": itemgetter("optimization_instruction"),
            "reason": itemgetter("reason"),
            "specific_data": itemgetter("specific_data"),
            "visual_context": itemgetter("visual_context"),
            "user_instructions": itemgetter("user_instructions"),
            "tone_modifier": itemgetter("tone_modifier")
        }
        | prompt
        | llm       # Usamos el LLM base
        | parser    # El parser limpia el markdown y devuelve un Diccionario
        | (lambda x: SocialPost(**x)) # Convertimos el Diccionario a Objeto Pydantic para no romper tu UI
    )
    
    return chain

# --- 4. FRONTEND: INTERFAZ DE USUARIO (STREAMLIT) ---

# Cabecera con Logo y Título (RESPONSIVE)
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <img src="{LOGO_URL_LARGE}" alt="Logo Arrojo" style="width: 60px; height: 60px; border-radius: 10px; object-fit: cover;">
        <h1 style="margin: 0; padding: 0;">Arrojo Content Generator</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Generador de copys con **RAG**, **Estilo Arrojer** y **Optimización de Formato**.")

# --- Barra Lateral: Configuración General ---
with st.sidebar:
    st.header("📢 Configuración")
    platform = st.selectbox("Plataforma", ["Instagram (Feed)", "Instagram (Stories)", "TikTok", "Facebook", "WhatsApp Channel", "YouTube (Video)", "YouTube (Shorts)"])
    media_type = st.selectbox("Formato Multimedia", ["Vídeo", "Foto", "Carrusel", "Solo Texto"])
    tone = st.select_slider("Tono del Mensaje", options=["Serio/Informativo", "Normal", "Canalla (Default)", "Urgente/Hype", "Emotivo"], value="Canalla (Default)")

    # Enlace discreto a documentación técnica (FOOTER FIJO)
    st.markdown(
        """
        <div class="sidebar-footer">
            <div style="text-align: center;">
                <a href="https://github.com/funkykespain/ContenidoArrojo" target="_blank" 
                   style="color: #666; text-decoration: none; font-size: 0.75rem; font-family: 'Roboto', sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px; transition: color 0.3s;">
                   <svg height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true" fill="#666"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
                   Repositorio & Docs
                </a>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- Área Principal: Formulario de Contenido ---

# Selección del Caso de Uso
reason = st.selectbox("¿Cuál es el motivo de la publicación?", [
    "1. Concierto",
    "2. Anuncio de Novedad",
    "3. Engagement / Busqueda de Likes",
    "4. Próximo Lanzamiento (Pre-save)",
    "5. Lanzamiento (Ya disponible)",
    "6. Crónica de Concierto Pasado",
    "7. Merchandising / Tienda",
    "8. Prensa / Entrevistas"
])

st.divider()

# Contenedor para datos específicos según el motivo seleccionado
specific_data = {} 

with st.form("main_form"):
    st.subheader("📝 Detalles del Contenido")

    col1, col2 = st.columns(2)

    # Campos comunes para cualquier tipo de publicación
    with col1:
        visual_context = st.text_area("Contexto Visual (¿Qué se ve?)", placeholder="Ej: Foto de Guille haciendo splash...")
    with col2:
        user_instructions = st.text_area("Instrucciones Extra", placeholder="Ej: Haz énfasis en que es gratis...")

    st.markdown("### 🎯 Datos Específicos")

    # Lógica condicional para mostrar solo los campos necesarios
    if reason == "1. Concierto":
        c1, c2 = st.columns(2)
        specific_data = {
            "date": c1.text_input("Fecha", placeholder="DD/MM"),
            "city": c2.text_input("Ciudad"),
            "venue": c1.text_input("Lugar/Sala"),
            "link_type": c2.radio("Tipo de Enlace", ["Venta de Entradas", "Ubicación/Web Sala"], horizontal=True),
            "link_url": st.text_input("URL del enlace")
        }
    elif reason == "2. Anuncio de Novedad":
        specific_data = {
            "description": st.text_area("¿Qué ha pasado?"),
            "tags": st.text_input("Etiquetas / Menciones")
        }
    elif reason == "3. Engagement / Busqueda de Likes":
        specific_data = {"hook": st.text_input("Gancho o Idea principal")}
    elif reason == "4. Próximo Lanzamiento (Pre-save)":
        c1, c2 = st.columns(2)
        specific_data = {
            "title": c1.text_input("Título"),
            "release_date": c2.text_input("Fecha Lanzamiento"),
            "type": c1.selectbox("Tipo", ["Single", "Videoclip", "Álbum"]),
            "link": c2.text_input("Link Pre-save")
        }
    elif reason == "5. Lanzamiento (Ya disponible)":
        c1, c2 = st.columns(2)
        specific_data = {
            "title": c1.text_input("Título"),
            "type": c2.selectbox("Tipo", ["Single", "Videoclip", "Álbum"]),
            "link": st.text_input("Link Escucha/Ver")
        }
    elif reason == "6. Crónica de Concierto Pasado":
        c1, c2 = st.columns(2)
        specific_data = {
            "city": c1.text_input("Ciudad/Sala"),
            "highlight": c2.text_input("Dato destacado"),
            "link": st.text_input("Link (si hay video/fotos)")
        }
    elif reason == "7. Merchandising / Tienda":
        c1, c2 = st.columns(2)
        specific_data = {
            "product": c1.text_input("Producto"),
            "price": c2.text_input("Precio"),
            "link": st.text_input("Link Tienda")
        }
    elif reason == "8. Prensa / Entrevistas":
        specific_data = {
            "media_name": st.text_input("Medio / Programa"),
            "link": st.text_input("Link a la entrevista/noticia"),
            "quote": st.text_area("Cita destacada")
        }

    # Botón de Acción Principal
    submitted = st.form_submit_button("🔥 Generar Copy Arrojer")

# --- 5. EJECUCIÓN Y VISUALIZACIÓN ---

if submitted:
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("❌ Falta la API Key en el archivo .env")
    else:
        with st.spinner("🎸 Afinando guitarras, leyendo la agenda y aplicando filtro anti-markdown..."):
            try:
                # 1. Inicializar la cadena de LangChain
                chain = get_chain()
                # 2. Obtener datos auxiliares
                # Descargar datos de agenda en tiempo real
                agenda_text = fetch_agenda_data()
                # Obtener fecha actual en formato legible
                today_str = datetime.now().strftime("%d/%m/%Y")
                # 3. Obtener instrucción de optimización
                # Calculamos la regla técnica según lo que el usuario eligió
                opt_instruction = get_optimization_instruction(platform, media_type)

                 # 4. Invocar al Agente con todos los datos necesarios
                response = chain.invoke({
                    "platform": platform,
                    "media_type": media_type,
                    "reason": reason,
                    "specific_data": specific_data,
                    "visual_context": visual_context,
                    "user_instructions": user_instructions,
                    "tone_modifier": tone,
                    "agenda_context": agenda_text,
                    "current_date": today_str,
                    "optimization_instruction": opt_instruction
                })
                
                # --- LIMPIEZA FINAL ---
                # Pasamos el texto generado por el filtro para asegurar formato correcto
                final_clean_text = clean_format_for_platform(response.copy_text, platform)
                
                # 5. Renderizar Resultados (Estilo Tarjeta)
                st.success("¡Copy Generado con éxito! 🤘")
                
                st.markdown("### 📋 Copy Final")
                
                # Usamos text_area para facilitar el copiado (sin formato de código)
                st.text_area("Texto optimizado:", value=final_clean_text, height=300)
                
                # Columnas para metadatos (Hashtags y Sugerencia visual)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; border: 1px solid #333;">
                        <h4 style="color: #e74c3c; margin: 0;">#️⃣ Hashtags</h4>
                        <p style="margin-top: 5px; font-size: 0.9em;">{response.hashtags}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""
                    <div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; border: 1px solid #333;">
                        <h4 style="color: #e74c3c; margin: 0;">💡 Idea Visual</h4>
                        <p style="margin-top: 5px; font-size: 0.9em;">{response.visual_suggestion}</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error al generar: {str(e)}")