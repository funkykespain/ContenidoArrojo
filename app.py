import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
from operator import itemgetter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
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

    /* Títulos con el estilo Montserrat Black de Stitch */
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
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
    
    /* 1. Configuramos el contenedor del sidebar para que ocupe toda la altura */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        min-height: 100vh;
        position: relative; /* Referencia para el footer */
        padding-bottom: 80px; /* Dejamos hueco abajo para que el footer no tape contenido si hay scroll */
    }

    /* 2. Posicionamos el footer de forma absoluta al final del contenedor */
    .sidebar-footer {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 20px;
        background-color: #000000; /* Fondo negro para integrar con el sidebar */
        z-index: 10;
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
    # Credenciales y Configuración
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION")

    # A. Modelo de Embeddings
    # Debe coincidir exactamente con el usado en la ingesta de datos hecha para otro proyecto paralelo.
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        openai_api_key=api_key,
        openai_api_base=base_url
    )

    # B. Conexión a Base de Datos Vectorial (Qdrant)
    # Configuramos el cliente con soporte HTTPS y puerto seguro.
    client = QdrantClient(
        url=qdrant_url,
        port=443,
        https=True,
        api_key=qdrant_key,
        timeout=20
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    # El retriever buscará los 3 fragmentos más relevantes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # C. Modelo de Lenguaje (LLM)
    # Usamos Llama 3.3 70B por su capacidad de razonamiento y coste eficiente.
    llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct",
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.7 # Temperatura media para balancear creatividad y precisión
    )

    # D. Definición de la Estructura de Salida
    # Obligamos al LLM a devolver un JSON estricto.
    class SocialPost(BaseModel):
        platform: str = Field(description="Plataforma seleccionada")
        copy_text: str = Field(description="El texto del post listo para copiar, con emojis y estructura")
        hashtags: str = Field(description="Lista de hashtags optimizados separada por espacios")
        visual_suggestion: str = Field(description="Sugerencia breve para la imagen/video si no se provee")

    structured_llm = llm.with_structured_output(SocialPost)

    # E. Prompt del Sistema (Personalidad "Arrojer")
    # Define la voz, el tono y las reglas de negocio del agente.
    # E. Prompt del Sistema (OPTIMIZADO V2)
    system_prompt = """
    # Contexto temporal
    HOY: {current_date}. Usa esta fecha para tiempos relativos ("mañana", "este viernes") y para saber si conciertos son pasados o futuros.

    # Identidad
    Eres el Community Manager de la banda de rock "Arrojo". Voz: "Estilo Arrojer": canalla, pasional y CERCANA. Hablas como un colega en la barra de un bar, no como un altavoz de noticias.

    # Reglas CRÍTICAS (OBLIGATORIAS)
    1) Regla del TÚ: Prohibido plural ("os esperamos", "preparaos", "arrojers"). Siempre 2ª persona singular ("te esperamos", "prepárate"). Le hablas a una sola persona.
    2) "Arrojers": Prohibido en título o frase inicial. Máx 1 vez por post, mejor en cierre o cuerpo, nunca destacándolo.
    3) Emojis: Máx 2-3 en todo el texto. No de relleno; solo para énfasis emocional real.
    4) Cero clichés IA: Prohibidas frases tipo "Noche inolvidable", "Lo vamos a romper", "Prepárense". Sé específico, real y rock castizo/cañero.

    # Estrategia y CTAs
    Si es CONCIERTO (reason=1):
    - CTA obligatorio: link de entradas.
    - CTA creativo: invitar a escuchar los temas antes (Spotify) para ir con las letras aprendidas.

    # Contexto RAG (datos reales)
    {context}

    # Agenda de conciertos (CSV oficial)
    Usa esta tabla para validar fechas, detectar ciudades repetidas o recordar hitos:
    {agenda_context}

    # Solicitud
    Post para {platform} en formato {media_type}.
    MOTIVO: {reason}
    DETALLES: {specific_data}
    VISUAL: {visual_context}
    EXTRA: {user_instructions}
    TONO: {tone_modifier}

    # Links fallback
    Entradas/Web/Info oficial: https://arrojorock.es
    Spotify: https://open.spotify.com/artist/4s0uEp9gcIcvU1ZEsDKQXv
    YouTube: https://www.youtube.com/channel/UCJnAZC6v6OfKxNydcD6CFqQ

    Devuelve el objeto JSON final.
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)

    # F. Construcción de la Cadena (Chain)
    # Generamos un string de búsqueda optimizado para RAG concatenando los inputs clave.
    rag_query_generator = (
        lambda x: f"{x['reason']} {x['specific_data']} {x['user_instructions']}"
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
            "reason": itemgetter("reason"),
            "specific_data": itemgetter("specific_data"),
            "visual_context": itemgetter("visual_context"),
            "user_instructions": itemgetter("user_instructions"),
            "tone_modifier": itemgetter("tone_modifier")
        }
        | prompt
        | structured_llm
    )
    
    return chain

# --- 4. FRONTEND: INTERFAZ DE USUARIO (STREAMLIT) ---

# Cabecera con Logo y Título
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="{LOGO_URL_LARGE}" alt="Logo Arrojo" style="width: 60px; height: 60px; border-radius: 10px;">
        <h1 style="margin: 0; padding: 0;">Arrojo Content Generator</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Generador de copys con **RAG** (Base de Conocimiento) y **Estilo Arrojer**.")

# --- Barra Lateral: Configuración General ---
with st.sidebar:
    st.header("📢 Configuración")
    platform = st.selectbox("Plataforma", ["Instagram (Feed)", "Instagram (Stories)", "TikTok", "Facebook", "WhatsApp Channel", "YouTube (Video)", "YouTube (Shorts)"])
    media_type = st.selectbox("Formato Multimedia", ["Vídeo", "Foto", "Carrusel", "Solo Texto"])
    tone = st.select_slider("Tono del Mensaje", options=["Serio/Informativo", "Normal", "Canalla (Default)", "Urgente/Hype", "Emotivo"], value="Canalla (Default)")

    # Enlace discreto a documentación técnica (CLASE ESPECÍFICA PARA POSICIONAMIENTO ABSOLUTO)
    st.markdown(
        """
        <div class="sidebar-footer">
            <hr style="margin-bottom: 15px; border-color: #333;">
            <div style="text-align: center;">
                <a href="https://github.com/funkykespain/ContenidoArrojo" target="_blank" 
                   style="color: #555; text-decoration: none; font-size: 0.75rem; font-family: 'Roboto', sans-serif; display: flex; align-items: center; justify-content: center; gap: 5px; transition: color 0.3s;">
                   <svg height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true" fill="#555"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
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
        visual_context = st.text_area("Contexto Visual (¿Qué se ve?)", placeholder="Ej: Foto de Guille haciendo splash, primer plano guitarra...")
    with col2:
        user_instructions = st.text_area("Instrucciones Extra", placeholder="Ej: Haz énfasis en que es gratis, o menciona a tal persona...")

    st.markdown("### 🎯 Datos Específicos")

    # Lógica condicional para mostrar solo los campos necesarios
    if reason == "1. Concierto":
        c1, c2 = st.columns(2)
        date = c1.text_input("Fecha", placeholder="DD/MM")
        city = c2.text_input("Ciudad")
        venue = c1.text_input("Lugar/Sala")
        
        link_type = c2.radio("Tipo de Enlace", ["Venta de Entradas", "Ubicación/Web Sala (Sin venta)"], horizontal=True)
        link_url = st.text_input("URL del enlace")
        
        specific_data = {"date": date, "city": city, "venue": venue, "link_type": link_type, "link_url": link_url}

    elif reason == "2. Anuncio de Novedad":
        news_desc = st.text_area("¿Qué ha pasado?", placeholder="Nuevo integrante, nuevo equipo, firma con agencia...")
        tags = st.text_input("Etiquetas / Menciones")
        specific_data = {"description": news_desc, "tags": tags}

    elif reason == "3. Engagement / Busqueda de Likes":
        hook = st.text_input("Gancho o Idea principal", placeholder="Broma interna, pregunta a fans...")
        specific_data = {"hook": hook}

    elif reason == "4. Próximo Lanzamiento (Pre-save)":
        c1, c2 = st.columns(2)
        title = c1.text_input("Título")
        date_release = c2.text_input("Fecha Lanzamiento")
        type_release = c1.selectbox("Tipo", ["Single", "Videoclip", "Álbum"])
        link_presave = c2.text_input("Link Pre-save")
        specific_data = {"title": title, "release_date": date_release, "type": type_release, "link": link_presave}

    elif reason == "5. Lanzamiento (Ya disponible)":
        c1, c2 = st.columns(2)
        title = c1.text_input("Título")
        type_release = c2.selectbox("Tipo", ["Single", "Videoclip", "Álbum"])
        link_listen = st.text_input("Link Escucha/Ver")
        specific_data = {"title": title, "type": type_release, "link": link_listen}

    elif reason == "6. Crónica de Concierto Pasado":
        c1, c2 = st.columns(2)
        city_past = c1.text_input("Ciudad/Sala")
        highlight = c2.text_input("Dato destacado", placeholder="Sold out, anécdota...")
        link_cronica = st.text_input("Link (si hay video/fotos)")
        specific_data = {"city": city_past, "highlight": highlight, "link": link_cronica}

    elif reason == "7. Merchandising / Tienda":
        c1, c2 = st.columns(2)
        product = c1.text_input("Producto", placeholder="Camiseta, CD...")
        price = c2.text_input("Precio / Oferta (Opcional)")
        link_shop = st.text_input("Link Tienda")
        specific_data = {"product": product, "price": price, "link": link_shop}

    elif reason == "8. Prensa / Entrevistas":
        media_name = st.text_input("Medio / Programa")
        link_media = st.text_input("Link a la entrevista/noticia")
        quote = st.text_area("Cita destacada (Opcional)")
        specific_data = {"media_name": media_name, "link": link_media, "quote": quote}

    # Botón de Acción Principal
    submitted = st.form_submit_button("🔥 Generar Copy Arrojer")

# --- 5. EJECUCIÓN Y VISUALIZACIÓN ---

if submitted:
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("❌ Falta la API Key en el archivo .env")
    else:
        with st.spinner("🎸 Afinando guitarras, leyendo la agenda y consultando la base de datos..."):
            try:
                # 1. Inicializar la cadena de LangChain
                chain = get_chain()
                
                # 2. Descargar datos de agenda en tiempo real
                agenda_text = fetch_agenda_data()
                
                # 3. Preparar datos para el prompt
                specific_data_str = str(specific_data)

                # 4. Obtener fecha actual en formato legible
                today_str = datetime.now().strftime("%d/%m/%Y")
                
                # 5. Invocar al Agente
                response = chain.invoke({
                    "platform": platform,
                    "media_type": media_type,
                    "reason": reason,
                    "specific_data": specific_data_str,
                    "visual_context": visual_context,
                    "user_instructions": user_instructions,
                    "tone_modifier": tone,
                    "agenda_context": agenda_text,
                    "current_date": today_str
                })
                
                # 6. Renderizar Resultados (Estilo Tarjeta)
                st.success("¡Copy Generado con éxito! 🤘")
                
                st.markdown("### 📋 Copy Final")
                
                with st.container():
                    st.code(response.copy_text, language="markdown")
                
                # Columnas para metadatos (Hashtags y Sugerencia visual)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("""
                    <div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; border: 1px solid #333;">
                        <h4 style="color: #e74c3c; margin: 0;">#️⃣ Hashtags</h4>
                        <p style="margin-top: 5px; font-size: 0.9em;">{}</p>
                    </div>
                    """.format(response.hashtags), unsafe_allow_html=True)
                    
                with c2:
                    st.markdown("""
                    <div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; border: 1px solid #333;">
                        <h4 style="color: #e74c3c; margin: 0;">💡 Idea Visual</h4>
                        <p style="margin-top: 5px; font-size: 0.9em;">{}</p>
                    </div>
                    """.format(response.visual_suggestion), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error al generar: {str(e)}")