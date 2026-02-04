import streamlit as st
import os
from dotenv import load_dotenv
import requests
from operator import itemgetter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- CORRECCIÓN DE IMPORTS (Solución al error anterior) ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# ----------------------------------------------------------

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# Leemos la agenda de conciertos desde la URL (CSV público de Google Sheets)
@st.cache_data(ttl=3600) # Cachear 1 hora para no saturar
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

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
load_dotenv()

# URLs de los logos
LOGO_URL_LARGE = "https://arrojorock.es/android-chrome-192x192.png"
LOGO_URL_SMALL = "https://arrojorock.es/favicon-32x32.png"

st.set_page_config(
    page_title="Arrojo Content Generator",
    page_icon=LOGO_URL_SMALL, # Icono en la pestaña del navegador
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS (THEME ARROJO) ---
st.markdown("""
<style>
    /* IMPORTAR FUENTES OFICIALES */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;900&family=Roboto:wght@300;400&display=swap');

    /* --- VARIABLES GLOBALES (Mapeo de tu styles.css) --- */
    :root {
        --color-primary: #e74c3c;
        --color-primary-dark: #c0392b;
        --bg-dark: #0e0e0e;
        --bg-card: #1a1a1a;
        --text-headings: #ffffff;
        --text-body: #cccccc;
    }

    /* --- FONDO Y TEXTO GENERAL --- */
    .stApp {
        background-color: var(--bg-dark);
        font-family: 'Roboto', sans-serif;
    }
    
    p, label, .stMarkdown {
        color: var(--text-body) !important;
    }

    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif !important;
        color: var(--text-headings) !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: -1px;
    }

    /* --- SIDEBAR (Barra Lateral) --- */
    section[data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }
    
    /* Títulos del sidebar */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: var(--color-primary) !important;
    }

    /* --- INPUTS (Cajas de texto y Selects) --- */
    /* Fondo oscuro para inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--bg-card) !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
    }
    
    /* Efecto Focus (Borde Rojo) */
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 5px rgba(231, 76, 60, 0.5) !important;
    }

    /* Etiquetas de los inputs */
    .stTextInput label, .stSelectbox label, .stTextArea label {
        color: #999 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    /* --- BOTONES (Estilo Rockero) --- */
    div.stButton > button {
        background: linear-gradient(45deg, var(--color-primary), var(--color-primary-dark)) !important;
        color: white !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        width: 100%;
    }

    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(231, 76, 60, 0.3);
    }
    
    div.stButton > button:active {
        transform: translateY(1px);
    }

    /* --- RESULTADOS (Estilo Tarjeta / Code Block) --- */
    .stCode {
        background-color: #111 !important;
        border: 1px solid #333;
        border-radius: 10px;
    }
    
    /* Separadores */
    hr {
        border-color: #333 !important;
    }
    
    /* Alertas y Mensajes */
    .stAlert {
        background-color: var(--bg-card) !important;
        color: white !important;
        border: 1px solid #333;
    }
    
    /* Slider (Tono) */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--color-primary) !important;
    }
    .stSlider div[data-baseweb="slider"] > div > div {
        background-color: #333 !important; /* Riel inactivo */
    }

</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND: LÓGICA LANGCHAIN Y RAG ---

@st.cache_resource
def get_chain():
    """
    Inicializa la conexión con LLM y VectorStore.
    Se cachea para no reconectar en cada interacción.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION")

    # A. Configurar Embeddings (Debe coincidir con n8n: Qwen3-8B vía OpenRouter)
    embeddings = OpenAIEmbeddings(
        model="qwen/qwen3-embedding-8b",
        openai_api_key=api_key,
        openai_api_base=base_url
    )

    # B. Conectar a Qdrant
    client = QdrantClient(
        url=qdrant_url,
        port=443,
        https=True,
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=20
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # C. Configurar LLM (Gemini 2.0 Flash vía OpenRouter)
    llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct",
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.7 # Un poco de creatividad para el copy
    )

    # D. Estructura de Salida (JSON)
    class SocialPost(BaseModel):
        platform: str = Field(description="Plataforma seleccionada")
        copy_text: str = Field(description="El texto del post listo para copiar, con emojis y estructura")
        hashtags: str = Field(description="Lista de hashtags optimizados separada por espacios")
        visual_suggestion: str = Field(description="Sugerencia breve para la imagen/video si no se provee")

    structured_llm = llm.with_structured_output(SocialPost)

    # E. Prompt del Sistema (MEGA-PROMPT ARROJO)
    system_prompt = """
    # Identidad
    Eres el Community Manager de la banda de rock "Arrojo". Tu voz es "Estilo Arrojer": canalla, pasional, pero sobre todo CERCANA.
    No eres un altavoz de noticias, eres un colega contándole una novedad a otro colega en la barra de un bar.

    # Reglas CRÍTICAS de Redacción (OBLIGATORIO CUMPLIR)
    1. **La Regla del TÚ:** PROHIBIDO hablar en plural ("os esperamos", "preparaos", "arrojers"). Habla SIEMPRE en segunda persona del singular ("te esperamos", "prepárate", "no te lo pierdas"). Le hablas a UNA sola persona que lee el móvil.
    2. **Uso de 'Arrojers':** PROHIBIDO usar "Arrojers" en el título o en la frase de apertura. Úsalo con mucha moderación, máximo 1 vez por post y preferiblemente en el cierre o cuerpo, nunca gritando.
    3. **Dieta de Emojis:** MÁXIMO 2 o 3 emojis en TODO el texto. Si pones más, pierdes credibilidad. No uses emojis de relleno (como poner una guitarra cada vez que dices música). Solo úsalos para enfatizar un golpe emocional real.
    4. **Cero Clichés de IA:** Prohibido usar frases como "Noche inolvidable", "Lo vamos a romper", "Prepárense". Sé específico y real, una banda auténtica de rock castizo y cañero que evita frases hechas.
    # Estrategia de Contenido y CTAs
    - **Si es CONCIERTO (reason 1):**
      * CTA 1 (Obligatorio): Link de venta de entradas.
      * CTA 2 (Creativo): Debes incitar a que escuchen los temas ANTES del concierto. Diles que vayan con las letras aprendidas para dejarse la voz. Enlaza a Spotify para "hacer los deberes".
    
    # Contexto RAG (Datos Reales)
    {context}

    # Agenda de Conciertos (Histórico y Futuro - CSV Oficial)
    Usa esta tabla para verificar fechas, ver si hemos tocado antes en esa ciudad o recordar hitos pasados:
    {agenda_context}

    # Instrucciones Específicas
    Genera un post para la plataforma {platform} con formato {media_type}.
    
    MOTIVO DEL POST: {reason}
    DETALLES: {specific_data}
    CONTEXTO VISUAL: {visual_context}
    INSTRUCCIONES EXTRA: {user_instructions}
    TONO: {tone_modifier}

    # Links Maestros (Si el usuario no provee uno específico)
    - Entradas/Web/Info oficial: https://arrojorock.es
    - Spotify: https://open.spotify.com/artist/4s0uEp9gcIcvU1ZEsDKQXv
    - YouTube: https://www.youtube.com/channel/UCJnAZC6v6OfKxNydcD6CFqQ

    Genera el objeto JSON final.
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)

    # F. Crear la cadena (CORREGIDA)
    # Definimos qué texto usamos para buscar en la BD (RAG)
    # Concatenamos motivo + datos + instrucciones para que la búsqueda sea precisa
    rag_query_generator = (
        lambda x: f"{x['reason']} {x['specific_data']} {x['user_instructions']}"
    )

    chain = (
        {
            "context": rag_query_generator | retriever,  # Ahora buscamos con texto, no con diccionarios
            "agenda_context": itemgetter("agenda_context"),
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

# --- 3. FRONTEND: STREAMLIT UI ---

# Título con Logo Oficial (Integrado con HTML para mejor alineación)
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="{LOGO_URL_LARGE}" alt="Logo Arrojo" style="width: 60px; height: 60px; border-radius: 10px;">
        <h1 style="margin: 0; padding: 0;">Arrojo Content Generator</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Generador de copys con **RAG** (Base de Conocimiento) y **Estilo Arrojer**.")

# --- Sidebar: Configuración Global ---
with st.sidebar:
    st.header("📢 Configuración")
    platform = st.selectbox("Plataforma", ["Instagram (Feed)", "Instagram (Stories)", "TikTok", "Facebook", "WhatsApp Channel", "YouTube (Video)", "YouTube (Shorts)"])
    media_type = st.selectbox("Formato Multimedia", ["Vídeo", "Foto", "Carrusel", "Solo Texto"])
    tone = st.select_slider("Tono del Mensaje", options=["Serio/Informativo", "Normal", "Canalla (Default)", "Urgente/Hype", "Emotivo"], value="Canalla (Default)")

    # --- Enlace a Documentación ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; margin-top: 10px;">
            <a href="https://github.com/funkykespain/ContenidoArrojo" target="_blank" 
               style="color: #555; text-decoration: none; font-size: 0.75rem; font-family: 'Roboto', sans-serif; display: flex; align-items: center; justify-content: center; gap: 5px; transition: color 0.3s;">
               <svg height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true" fill="#555"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
               Repositorio & Docs
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )
# --- Área Principal: Formulario Dinámico ---

# Selector de Motivo
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

# Formulario específico según motivo
specific_data = {} # Aquí guardaremos los datos limpios para el LLM

with st.form("main_form"):
    st.subheader("📝 Detalles del Contenido")
    
    col1, col2 = st.columns(2)
    
    # Campos comunes
    with col1:
        visual_context = st.text_area("Contexto Visual (¿Qué se ve?)", placeholder="Ej: Foto de Kyke saltando, primer plano guitarra...")
    with col2:
        user_instructions = st.text_area("Instrucciones Extra", placeholder="Ej: Haz énfasis en que es gratis, o menciona a tal persona...")

    st.markdown("### 🎯 Datos Específicos")

    # LÓGICA CONDICIONAL DE CAMPOS
    if reason == "1. Concierto":
        c1, c2 = st.columns(2)
        date = c1.text_input("Fecha", placeholder="DD/MM")
        city = c2.text_input("Ciudad")
        venue = c1.text_input("Lugar/Sala")
        
        # Lógica del Link corregida
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
        product = c1.text_input("Producto", placeholder="Camiseta, Vinilo...")
        price = c2.text_input("Precio / Oferta (Opcional)")
        link_shop = st.text_input("Link Tienda")
        specific_data = {"product": product, "price": price, "link": link_shop}

    elif reason == "8. Prensa / Entrevistas":
        media_name = st.text_input("Medio / Programa")
        link_media = st.text_input("Link a la entrevista/noticia")
        quote = st.text_area("Cita destacada (Opcional)")
        specific_data = {"media_name": media_name, "link": link_media, "quote": quote}

    # Botón de Generación
    submitted = st.form_submit_button("🔥 Generar Copy Arrojer")

# --- 4. EJECUCIÓN ---

if submitted:
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("❌ Falta la API Key en el archivo .env")
    else:
        with st.spinner("🎸 Afinando guitarras, leyendo la agenda y consultando la base de datos..."):
            try:
                # 1. Obtener la cadena
                chain = get_chain()
                
                # 1.5 OBTENER LA AGENDA (NUEVO)
                agenda_text = fetch_agenda_data()
                
                # 2. Convertir diccionario specific_data a string
                specific_data_str = str(specific_data)
                
                # 3. Invocar
                response = chain.invoke({
                    "platform": platform,
                    "media_type": media_type,
                    "reason": reason,
                    "specific_data": specific_data_str,
                    "visual_context": visual_context,
                    "user_instructions": user_instructions,
                    "tone_modifier": tone,
                    "agenda_context": agenda_text
                })
                
                # 4. Mostrar Resultados (ESTILO ARROJO WEB)
                st.success("¡Copy Generado con éxito! 🤘")
                
                st.markdown("### 📋 Copy Final")
                
                # Usamos un container para el texto principal
                with st.container():
                    st.code(response.copy_text, language="markdown")
                
                # Tarjetas inferiores para datos extra
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

            # --- ESTO ES LO QUE FALTABA ---
            except Exception as e:
                st.error(f"Error al generar: {str(e)}")