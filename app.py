"""
Sistema de Consulta RAG - Caso Penal
Versión Streamlit Cloud
"""

import streamlit as st
import json
import numpy as np
import requests
import faiss
import os
from pathlib import Path
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Trigger de actualizacion nube: 2026-02-06 10:48
UPDATE_TRIGGER = "force_redeploy_v3"

# --- RUTAS ---
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
INDEX_PATH = DATA_DIR / "caso_penal.index"
CHUNKS_PATH = DATA_DIR / "chunks_caso.json"
CONFIG_PATH = DATA_DIR / "config_caso.json"

# --- DEEPSEEK ---
DEEPSEEK_API_KEY = st.secrets["credentials"]["deepseek_api_key"]
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# --- CONFIG ---
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# --- PROMPT ---
SYSTEM_PROMPT_CASO = """Eres un asistente legal especializado en derecho penal peruano, trabajando para la DEFENSA del investigado Raul Antonio Oliva Guerrero.

CASO: Expediente 00203-2024-23-5001-JR-PE-01
DELITOS IMPUTADOS: Organizacion Criminal (Art. 317 CP) y Trafico de Influencias (Art. 400 CP)
JUZGADO: 1er Juzgado de Investigacion Preparatoria Nacional
JUEZ: Richard Augusto Concepcion Carhuancho
FISCALIA: EFICCOP - Equipo 5

SOBRE EL DEFENDIDO:
- Raul Antonio Oliva Guerrero fue Director de la Direccion de Autoridades Politicas del Ministerio del Interior
- Se le imputa ser "operador funcionarial" de una presunta organizacion criminal
- Designado el 01/03/2023 mediante R.M. n. 0298-2023-IN

TU ROL:
1. Responde basandote UNICAMENTE en los documentos del caso proporcionados como contexto
2. Identifica tanto los elementos de cargo como posibles argumentos de defensa
3. Cita siempre el documento fuente, pagina y seccion
4. Si detectas contradicciones o debilidades en la acusacion, senalalas
5. Se preciso con nombres, fechas y cargos
6. Si no encuentras informacion en el contexto, dilo claramente

FORMATO DE RESPUESTA:
- **Respuesta:** (resumen directo)
- **Detalle:** (analisis con citas del expediente)
- **Fuentes:** (documento, pagina)
- **Nota para la defensa:** (si aplica, observaciones estrategicas)

CONTEXTO DE DOCUMENTOS DEL CASO:
{contexto}

---
CONSULTA:
{consulta}"""


# --- FUNCIONES ---
@st.cache_resource
def cargar_modelo():
    return SentenceTransformer(CONFIG["modelo_embeddings"])


@st.cache_resource
def cargar_indice():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def buscar_documentos(consulta, modelo, index, chunks, top_k=8):
    query_emb = modelo.encode([consulta], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_emb, top_k)
    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            resultados.append(chunk)
    return resultados


def consultar_deepseek(consulta, resultados):
    contexto = "\n\n".join([
        f"[{i+1}] Documento: {r['archivo_original']} | Tipo: {r['tipo_documento']} | "
        f"Pagina: {r['pagina']} | Relevancia: {r['score']:.3f}\n"
        f"Personas: {', '.join(r.get('personas_mencionadas', [])) or 'N/A'}\n"
        f"{r['texto']}"
        for i, r in enumerate(resultados)
    ])

    prompt = SYSTEM_PROMPT_CASO.format(contexto=contexto, consulta=consulta)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 3000
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error al consultar IA: {str(e)}"


# --- LOGIN ---
def verificar_login():
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False
        st.session_state.usuario = None

    if st.session_state.autenticado:
        return True

    st.markdown("## Acceso al Sistema de Consulta")
    st.markdown("Expediente N. 00203-2024-23-5001-JR-PE-01")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        usuario = st.text_input("Usuario", key="login_user")
        clave = st.text_input("Clave", type="password", key="login_pass")

        if st.button("Ingresar", use_container_width=True):
            passwords = st.secrets.get("passwords", {})
            if usuario in passwords and passwords[usuario] == clave:
                st.session_state.autenticado = True
                st.session_state.usuario = usuario
                st.rerun()
            else:
                st.error("Credenciales incorrectas")
    return False


# --- INTERFAZ ---
def main():
    st.set_page_config(
        page_title="Consulta Expediente Penal",
        page_icon="LEGAL",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .chunk-source {
        background: #1a1a2e;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 3px solid #e94560;
        font-size: 0.85rem;
        color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)

    if not verificar_login():
        return

    modelo = cargar_modelo()
    index, chunks = cargar_indice()

    if index is None:
        st.error("Error: No se encontraron los datos del caso.")
        return

    # Sidebar
    with st.sidebar:
        st.markdown("### Caso Penal")
        st.markdown(f"**Exp:** {CONFIG['caso']['expediente']}")
        st.markdown(f"**Defendido:** {CONFIG['caso']['defendido']}")
        st.divider()
        st.metric("Vectores en indice", index.ntotal)
        st.metric("Chunks disponibles", len(chunks))

        # Contar documentos unicos
        docs_unicos = set(c.get("archivo_original", "") for c in chunks)
        st.metric("Documentos", len(docs_unicos))

        st.divider()
        st.markdown("**Documentos cargados:**")
        for doc in sorted(docs_unicos):
            st.caption(f"- {doc}")

        st.divider()
        st.markdown(f"*Usuario: {st.session_state.usuario}*")
        if st.button("Cerrar sesion"):
            st.session_state.autenticado = False
            st.rerun()

    # Contenido principal
    st.markdown("## Sistema de Consulta - Expediente Penal")

    tab_chat, tab_busqueda = st.tabs(["Chat con IA", "Busqueda directa"])

    # --- CHAT ---
    with tab_chat:
        st.markdown("Pregunte sobre el caso. La IA busca en los documentos y responde.")

        if "mensajes" not in st.session_state:
            st.session_state.mensajes = []

        for msg in st.session_state.mensajes:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        consulta = st.chat_input("Escriba su consulta sobre el caso...")

        if consulta:
            st.session_state.mensajes.append({"role": "user", "content": consulta})
            with st.chat_message("user"):
                st.markdown(consulta)

            with st.chat_message("assistant"):
                with st.spinner("Buscando en el expediente..."):
                    resultados = buscar_documentos(consulta, modelo, index, chunks)

                with st.spinner("Analizando con IA..."):
                    respuesta = consultar_deepseek(consulta, resultados)

                st.markdown(respuesta)

                with st.expander("Ver documentos fuente"):
                    for i, r in enumerate(resultados):
                        st.markdown(
                            f'<div class="chunk-source">'
                            f'<b>[{i+1}]</b> {r["archivo_original"]} | '
                            f'Pag. {r["pagina"]} | '
                            f'Relevancia: {r["score"]:.3f}<br>'
                            f'<small>{r["texto"][:300]}...</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.mensajes.append({"role": "assistant", "content": respuesta})

        if not st.session_state.mensajes:
            st.markdown("### Consultas sugeridas")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                - Que se le imputa a Oliva Guerrero?
                - Cual es el rol de Oliva en la organizacion?
                - Que pruebas hay contra Oliva Guerrero?
                - Que dijo Oliva en su declaracion?
                """)
            with col2:
                st.markdown("""
                - Quienes son los co-investigados?
                - Que es la Ley 32108 y como afecta el caso?
                - Cual es la estructura de la organizacion criminal?
                - Que contradicciones hay en las declaraciones?
                """)

    # --- BÚSQUEDA DIRECTA ---
    with tab_busqueda:
        st.markdown("Busqueda directa sin IA. Muestra los fragmentos mas relevantes.")

        busqueda = st.text_input("Buscar en el expediente:", key="busq")
        num_res = st.slider("Resultados", 3, 20, 10)

        if busqueda:
            resultados = buscar_documentos(busqueda, modelo, index, chunks, top_k=num_res)
            st.markdown(f"**{len(resultados)} resultados**")

            for i, r in enumerate(resultados):
                with st.expander(
                    f"[{r['score']:.3f}] {r['archivo_original']} - Pag. {r['pagina']}",
                    expanded=(i < 3)
                ):
                    personas = r.get("personas_mencionadas", [])
                    if personas:
                        st.markdown(f"**Personas:** {', '.join(personas)}")
                    st.markdown(r["texto"])


if __name__ == "__main__":
    main()
