"""
Sistema de Consulta RAG - Caso Penal
Versión Cloud Completa (Sincronizada con Local)
"""

import streamlit as st
import json
import numpy as np
import requests
import faiss
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Trigger de actualizacion nube: 2026-02-24 (Agentic RAG + Streaming + Memoria)
UPDATE_TRIGGER = "force_redeploy_v5_agentic"

# --- RUTAS CLOUD ---
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

INDEX_PATH = DATA_DIR / "caso_penal.index"
CHUNKS_PATH = DATA_DIR / "chunks_caso.json"
CONFIG_PATH = DATA_DIR / "config_caso.json"
META_PATH = DATA_DIR / "meta_embeddings.json"
PROCESADOS_DIR = DATA_DIR / "03_PARSER_EMBEDDINGS" / "procesados"

# --- DEEPSEEK ---
if "credentials" in st.secrets:
    DEEPSEEK_API_KEY = st.secrets["credentials"]["deepseek_api_key"]
else:
    DEEPSEEK_API_KEY = "sk-4e6b4c12e3e24d5c8296b6084aac4aac"

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# --- CONFIG ---
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# --- CREDENCIALES ---
if "passwords" in st.secrets:
    USUARIOS = st.secrets["passwords"]
else:
    USUARIOS = {
        "raul": "caso2024",
        "abogado": "defensa2024",
        "juan": "admin2024"
    }

# --- PROMPT DEL SISTEMA ---
SYSTEM_PROMPT_CASO = """Eres un asistente legal especializado en derecho penal peruano, trabajando para la DEFENSA del investigado Raúl Antonio Oliva Guerrero.

CASO: Expediente 00203-2024-23-5001-JR-PE-01
DELITOS IMPUTADOS: Organización Criminal (Art. 317 CP) y Tráfico de Influencias (Art. 400 CP)
JUZGADO: 1er Juzgado de Investigación Preparatoria Nacional
JUEZ: Richard Augusto Concepción Carhuancho
FISCALÍA: EFICCOP - Equipo 5

SOBRE EL DEFENDIDO:
- Raúl Antonio Oliva Guerrero fue Director de la Dirección de Autoridades Políticas del Ministerio del Interior
- Se le imputa ser "operador funcionarial" de una presunta organización criminal
- Designado el 01/03/2023 mediante R.M. n.° 0298-2023-IN

TU ROL:
1. Responde basándote ÚNICAMENTE en los documentos del caso proporcionados como contexto
2. Identifica tanto los elementos de cargo como posibles argumentos de defensa
3. Cita siempre el documento fuente, página y sección
4. Si detectas contradicciones o debilidades en la acusación, señálalas
5. Sé preciso con nombres, fechas y cargos
6. Si no encuentras información en el contexto, dilo claramente

FORMATO DE RESPUESTA:
- **Respuesta:** (resumen directo)
- **Detalle:** (análisis con citas del expediente)
- **Fuentes:** (documento, página)
- **Nota para la defensa:** (si aplica, observaciones estratégicas)

CONTEXTO DE DOCUMENTOS DEL CASO:
{contexto}

---
CONSULTA:
{consulta}"""


AGENTIC_SYSTEM_PROMPT_CASO = """Eres un asistente legal especializado en defensa jurídica penal peruana.

CASO: Expediente 00203-2024-23-5001-JR-PE-01
DEFENDIDO: Raúl Antonio Oliva Guerrero
DELITOS: Art. 317 CP (Org. Criminal), Art. 400 CP (Tráfico Influencias)
JUZGADO: 1er Juzgado de Investigación Preparatoria Nacional | JUEZ: Concepción Carhuancho

Tienes acceso a search_expediente para buscar en los documentos del caso (disposiciones fiscales, resoluciones, providencias, declaraciones, informes).

PROCESO OBLIGATORIO:
1. Analiza qué información del expediente necesitas para responder
2. Usa search_expediente con términos específicos (nombres, fechas, hechos, tipo de documento)
3. Busca múltiples veces para cubrir distintos ángulos de la pregunta
4. Identifica contradicciones entre testimonios o inconsistencias en la acusación
5. Solo cuando tengas suficiente información, detente

REGLAS DE BÚSQUEDA:
- Máximo 5 búsquedas por consulta
- Usa nombres exactos cuando busques personas (ej: "Oliva Guerrero declaración")
- Usa el tipo de documento si es relevante (ej: "Disposición 16 organización criminal")
- Para contradicciones, busca el mismo hecho en documentos distintos

REGLAS DE RESPUESTA (las aplica R1 al final, no tú):
- Solo cita lo que encontraste en el expediente
- Identifica contradicciones y debilidades de la acusación cuando las veas
- Sé preciso con nombres, fechas y fuentes"""


# --- FUNCIONES DE CARGA ---
@st.cache_resource
def cargar_modelo():
    """Carga el modelo de embeddings."""
    return SentenceTransformer(CONFIG["modelo_embeddings"])


@st.cache_resource
def cargar_indice():
    """Carga el índice FAISS y los chunks."""
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


@st.cache_data
def cargar_estadisticas():
    """Carga estadísticas del sistema."""
    stats = {
        "total_documentos": 0,
        "total_vectores": 0,
        "total_personas": set(),
        "tipos_documento": {},
        "documentos": []
    }

    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        stats["total_vectores"] = meta.get("total_vectores", 0)
        stats["total_documentos"] = len(meta.get("documentos_incluidos", {}))

        for doc_id, info in meta.get("documentos_incluidos", {}).items():
            tipo = info.get("tipo", "otro")
            stats["tipos_documento"][tipo] = stats["tipos_documento"].get(tipo, 0) + 1
            stats["documentos"].append({
                "id": doc_id,
                "archivo": info.get("archivo", ""),
                "tipo": tipo,
                "chunks": info.get("chunks", 0)
            })

    if PROCESADOS_DIR.exists():
        for json_path in PROCESADOS_DIR.glob("*.json"):
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            for nombre in doc.get("personas", {}).keys():
                stats["total_personas"].add(nombre)

    stats["total_personas"] = len(stats["total_personas"])
    return stats


# --- BÚSQUEDA ---
def buscar_documentos(consulta, modelo, index, chunks, top_k=8):
    """Busca los chunks más relevantes."""
    query_embedding = modelo.encode([consulta], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_embedding, top_k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            resultados.append(chunk)

    return resultados


def stream_consultar_caso(consulta, resultados, session_summary=""):
    """
    Versión streaming de consultar_deepseek.
    Usa deepseek-reasoner (R1) para la síntesis final.
    Generador que emite tokens uno a uno para st.write_stream().
    """
    contexto = "\n\n".join([
        f"[{i+1}] Documento: {r['archivo_original']} | Tipo: {r['tipo_documento']} | "
        f"Página: {r['pagina']} | Relevancia: {r['score']:.3f}\n"
        f"Personas mencionadas: {', '.join(r.get('personas_mencionadas', [])) or 'N/A'}\n"
        f"{r['texto']}"
        for i, r in enumerate(resultados)
    ])

    system_prompt = SYSTEM_PROMPT_CASO.split("CONTEXTO DE DOCUMENTOS")[0].strip()
    if session_summary:
        system_prompt += f"\n\nCONTEXTO DE LA SESIÓN ACTUAL (turnos previos):\n{session_summary}"

    prompt = f"{system_prompt}\n\nCONTEXTO DE DOCUMENTOS DEL CASO:\n{contexto}\n\n---\nCONSULTA:\n{consulta}"

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4000,
        "stream": True
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, stream=True, timeout=90)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        pass
    except Exception as e:
        yield f"\n\nError al consultar IA: {str(e)}"


def agentic_buscar_expediente(consulta, modelo, index, chunks, session_summary="", max_iter=5):
    """
    Fase 1 del Agentic RAG para el caso penal:
    - deepseek-chat con tools decide qué buscar en el expediente
    - Acumula chunks únicos de todas las búsquedas
    Retorna (top_chunks, trace)
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_expediente",
                "description": "Busca en los documentos del expediente penal (disposiciones, resoluciones, providencias, declaraciones). Úsala para encontrar hechos, testimonios, fechas, personas, argumentos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Texto a buscar. Usa nombres exactos, fechas, hechos o tipo de documento. Ejemplo: 'Oliva Guerrero declaración abril 2024' o 'Disposición 16 organización criminal'"
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 10,
                            "description": "Número de fragmentos a retornar. Default 5."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    gathered_chunks = []
    seen_hashes = set()

    def execute_search(query, top_k=5):
        resultados = buscar_documentos(query, modelo, index, chunks, top_k=top_k * 2)
        if not resultados:
            return "No se encontraron fragmentos relevantes para esa búsqueda."
        lines = []
        for i, r in enumerate(resultados[:top_k]):
            h = hash(r["texto"])
            if h not in seen_hashes:
                gathered_chunks.append(r)
                seen_hashes.add(h)
            lines.append(
                f"[{i+1}] {r['archivo_original']} | Pág. {r['pagina']} | "
                f"Tipo: {r['tipo_documento']} | Score: {r['score']:.3f}\n"
                f"Personas: {', '.join(r.get('personas_mencionadas', [])) or 'N/A'}\n"
                f"{r['texto']}"
            )
        return "\n\n".join(lines)

    def get_top_chunks():
        return sorted(gathered_chunks, key=lambda x: x.get("score", 0), reverse=True)[:12]

    system_content = AGENTIC_SYSTEM_PROMPT_CASO
    if session_summary:
        system_content += f"\n\nCONTEXTO DE LA SESIÓN ACTUAL:\n{session_summary}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": consulta}
    ]

    trace = []
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    for iteration in range(max_iter):
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.1,
            "max_tokens": 1000
        }
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            finish_reason = choice.get("finish_reason", "")
        except Exception:
            return get_top_chunks(), trace

        tool_calls = msg.get("tool_calls") or []

        if not tool_calls or finish_reason == "stop":
            return get_top_chunks(), trace

        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})

        for tc in tool_calls:
            fn_args = json.loads(tc["function"]["arguments"])
            query = fn_args.get("query", "")
            top_k = fn_args.get("top_k", 5)

            trace.append({"iteracion": iteration + 1, "query": query, "top_k": top_k})
            resultado = execute_search(query, top_k)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": resultado
            })

    return get_top_chunks(), trace


def actualizar_resumen_sesion(resumen_anterior, consulta_usuario, respuesta_asistente):
    """Actualiza el resumen acumulativo de la sesión. Fallo silencioso."""
    prompt = f"""Eres un sintetizador de conversaciones sobre un caso penal.
Actualiza el resumen de la sesión incorporando el nuevo turno.
Captura: hechos clave discutidos, personas mencionadas, documentos revisados, conclusiones de defensa.
Máximo 6 líneas. Muy conciso. No repitas lo ya resumido.

RESUMEN ANTERIOR: {resumen_anterior if resumen_anterior else "(sesión nueva)"}

NUEVO TURNO:
Usuario: {consulta_usuario[:400]}
Asistente: {respuesta_asistente[:400]}

RESUMEN ACTUALIZADO:"""

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 300
    }
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        nuevo = resp.json()["choices"][0]["message"]["content"].strip()
        return nuevo if nuevo else resumen_anterior
    except Exception:
        return resumen_anterior


def verificar_referencias_caso(respuesta, resultados):
    """Verifica referencias a artículos CP y documentos del expediente."""
    patron_arts = r'[Aa]rt(?:ículo|iculo)?s?\.?\s+(\d+(?:\.\d+)*)\s*(?:CP|del\s+CP|Código\s+Penal)?'
    citas_arts = set(f"Art. {n}" for n in re.findall(patron_arts, respuesta))
    citas_docs = set()
    for tipo, num in re.findall(r'(Disposici[oó]n|Resoluci[oó]n|Providencia|Informe)\s+(?:N[°º]?\s*)?(\d+)', respuesta):
        citas_docs.add(f"{tipo} {num}")

    todas_citas = citas_arts | citas_docs
    if not todas_citas:
        return set(), set()

    texto_total = " ".join(r["texto"] for r in resultados) + " " + " ".join(r.get("archivo_original", "") for r in resultados)

    verificadas, no_verificadas = set(), set()
    for cita in todas_citas:
        num = re.search(r'\d+', cita)
        if num and re.search(rf'\b{re.escape(num.group())}\b', texto_total):
            verificadas.add(cita)
        else:
            no_verificadas.add(cita)

    return verificadas, no_verificadas


def generar_reporte_word(consulta, respuesta, resultados):
    """Genera un contenido HTML compatible con Word (.doc)."""
    fecha = datetime.now().strftime('%d/%m/%Y %H:%M')
    respuesta_html = respuesta.replace("\n", "<br>")

    html = f"""
    <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
    <head>
        <meta charset='utf-8'>
        <title>Reporte Caso Penal</title>
        <style>
            body {{ font-family: 'Calibri', Arial, sans-serif; line-height: 1.5; }}
            h1 {{ color: #2E74B5; border-bottom: 2px solid #2E74B5; padding-bottom: 10px; }}
            h2 {{ color: #1F4D78; margin-top: 25px; border-bottom: 1px solid #ddd; }}
            .info-box {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; }}
            .respuesta-box {{ background-color: #e8f4f8; padding: 15px; border-left: 5px solid #2E74B5; margin-bottom: 20px; }}
            .fuente-box {{ border: 1px solid #eee; padding: 10px; margin-bottom: 15px; background-color: #fff; }}
            .fuente-header {{ font-weight: bold; color: #555; font-size: 0.9em; background-color: #f0f0f0; padding: 5px; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #888; text-align: center; border-top: 1px solid #eee; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Consulta Legal - Caso Penal</h1>
        <div class="info-box">
            <p><strong>Fecha:</strong> {fecha}</p>
            <p><strong>Consulta Realizada:</strong> {consulta}</p>
        </div>
        <h2>Análisis de Inteligencia Artificial</h2>
        <div class="respuesta-box">{respuesta_html}</div>
        <h2>Documentos Fuente Consultados</h2>
        <p>Fragmentos del expediente utilizados para generar la respuesta:</p>
    """

    for i, r in enumerate(resultados):
        texto_limpio = r['texto'].replace("\n", " ")
        html += f"""
        <div class="fuente-box">
            <div class="fuente-header">
                [{i+1}] {r['archivo_original']} (Pág. {r['pagina']}) | Tipo: {r['tipo_documento']} | Relevancia: {r['score']:.3f}
            </div>
            <p>{texto_limpio}</p>
        </div>
        """

    html += """
        <div class="footer">Generado por Sistema de Consulta Legal RAG - JNJ</div>
    </body>
    </html>
    """
    return html.encode('utf-8')


# --- AUTENTICACIÓN ---
def verificar_login():
    """Sistema de login simple."""
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False
        st.session_state.usuario = None

    if st.session_state.autenticado:
        return True

    st.markdown("## Acceso al Sistema")
    st.markdown("Ingrese sus credenciales para acceder al expediente.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        usuario = st.text_input("Usuario", key="login_user")
        clave = st.text_input("Clave", type="password", key="login_pass")

        if st.button("Ingresar", use_container_width=True):
            if usuario in USUARIOS and USUARIOS[usuario] == clave:
                st.session_state.autenticado = True
                st.session_state.usuario = usuario
                st.rerun()
            else:
                st.error("Credenciales incorrectas")

    return False


# --- INTERFAZ PRINCIPAL ---
def main():
    st.set_page_config(
        page_title="Caso Penal - Consulta de Expediente",
        page_icon="LEGAL",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 1.5rem; font-weight: bold; color: #1a1a2e;
        padding: 0.5rem 0; border-bottom: 2px solid #e94560; margin-bottom: 1rem;
    }
    .stat-card {
        background: #f8f9fa; border-radius: 8px; padding: 1rem;
        text-align: center; border-left: 4px solid #0f3460;
    }
    .chunk-source {
        background: #f0f2f6; border-radius: 6px; padding: 0.8rem;
        margin: 0.5rem 0; border-left: 3px solid #e94560; font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

    if not verificar_login():
        return

    modelo = cargar_modelo()
    index, chunks = cargar_indice()
    stats = cargar_estadisticas()

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### Caso Penal")
        st.markdown(f"**Exp:** {CONFIG['caso']['expediente']}")
        st.markdown(f"**Defendido:** {CONFIG['caso']['defendido']}")
        st.markdown(f"**Juzgado:** {CONFIG['caso']['juzgado']}")
        st.divider()

        st.markdown("### Estado del Sistema")
        fecha_indice = "No encontrado"
        if INDEX_PATH.exists():
            fecha_mod = datetime.fromtimestamp(INDEX_PATH.stat().st_mtime)
            fecha_indice = fecha_mod.strftime("%d/%m %H:%M:%S")

        st.metric("Datos actualizados", fecha_indice)
        st.metric("Documentos", stats["total_documentos"])
        st.metric("Vectores", stats["total_vectores"])
        st.metric("Personas detectadas", stats["total_personas"])

        st.divider()
        st.markdown("### Documentos cargados")
        for doc in stats.get("documentos", []):
            st.markdown(f"- **{doc['tipo']}**: {doc['archivo']} ({doc['chunks']} chunks)")

        st.divider()
        st.markdown(f"*Usuario: {st.session_state.usuario}*")

        col_buttons = st.columns(2)
        with col_buttons[0]:
            if st.button("🔄 Recargar"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()
        with col_buttons[1]:
            if st.button("Cerrar sesión"):
                st.session_state.autenticado = False
                st.session_state.usuario = None
                st.session_state.session_summary = ""
                st.rerun()

        st.divider()
        if st.button("🧹 Limpiar Chat"):
            st.session_state.mensajes = []
            st.session_state.session_summary = ""
            st.rerun()
        debug_mode = st.toggle("🛠️ Modo Debug")

    # --- CONTENIDO PRINCIPAL ---
    st.markdown('<div class="main-header">Sistema de Consulta - Expediente Penal</div>', unsafe_allow_html=True)

    if index is None or chunks is None:
        st.error("No se han generado embeddings. Ejecute primero PROCESAR_CASO.bat localmente y suba los datos.")
        return

    tab_chat, tab_busqueda, tab_personas = st.tabs(["Chat con IA", "Busqueda directa", "Personas del caso"])

    # --- TAB: CHAT CON IA ---
    with tab_chat:
        st.markdown("Haga preguntas sobre el caso. La IA buscara en los documentos y respondera.")

        if "mensajes" not in st.session_state:
            st.session_state.mensajes = []
        if "session_summary" not in st.session_state:
            st.session_state.session_summary = ""
        if "last_chunks" not in st.session_state:
            st.session_state.last_chunks = []

        for msg in st.session_state.mensajes:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        consulta = st.chat_input("Escriba su consulta sobre el caso...")

        if consulta:
            st.session_state.mensajes.append({"role": "user", "content": consulta})
            with st.chat_message("user"):
                st.markdown(consulta)

            with st.chat_message("assistant"):
                # Fase 1: Agente busca en el expediente
                with st.spinner("🔍 Analizando y buscando en el expediente..."):
                    top_chunks, trace = agentic_buscar_expediente(
                        consulta, modelo, index, chunks,
                        session_summary=st.session_state.session_summary
                    )

                # Fase 2: R1 sintetiza en streaming
                if top_chunks:
                    respuesta = st.write_stream(stream_consultar_caso(
                        consulta, top_chunks,
                        session_summary=st.session_state.session_summary
                    ))
                    st.session_state.last_chunks = top_chunks
                else:
                    respuesta = "No se encontraron fragmentos relevantes en el expediente."
                    st.warning(respuesta)

                # Fase 3: Verificar referencias
                if top_chunks:
                    verificadas, no_verificadas = verificar_referencias_caso(respuesta, top_chunks)
                    if no_verificadas:
                        st.warning(
                            f"⚠️ **Referencias sin respaldo en contexto** — verificar: "
                            f"{', '.join(sorted(no_verificadas))}"
                        )
                    if verificadas:
                        st.caption(f"✅ Referencias verificadas: {', '.join(sorted(verificadas))}")

                # Botón de descarga Word
                reporte_bytes = generar_reporte_word(consulta, respuesta, top_chunks)
                st.download_button(
                    label="📄 Descargar Reporte en Word",
                    data=reporte_bytes,
                    file_name=f"Reporte_Caso_{datetime.now().strftime('%Y%m%d_%H%M')}.doc",
                    mime="application/msword",
                    key=f"download_{len(st.session_state.mensajes)}"
                )

                # Fuentes consultadas
                with st.expander(f"📂 Documentos consultados ({len(top_chunks)})"):
                    for i, r in enumerate(top_chunks):
                        st.markdown(
                            f'<div class="chunk-source">'
                            f'<b>[{i+1}]</b> {r["archivo_original"]} | '
                            f'Pag. {r["pagina"]} | '
                            f'Tipo: {r["tipo_documento"]} | '
                            f'Relevancia: {r["score"]:.3f}<br>'
                            f'<small>{r["texto"][:300]}...</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # Debug
                if debug_mode:
                    if trace:
                        with st.expander(f"🔍 Búsquedas realizadas ({len(trace)})"):
                            for t in trace:
                                st.markdown(f"**Búsqueda {t['iteracion']}:** `{t['query']}` (top_k={t['top_k']})")
                    if st.session_state.session_summary:
                        with st.expander("🧠 Resumen de sesión"):
                            st.markdown(st.session_state.session_summary)

            st.session_state.mensajes.append({"role": "assistant", "content": respuesta})

            # Fase 4: Actualizar resumen de sesión (silencioso)
            st.session_state.session_summary = actualizar_resumen_sesion(
                st.session_state.session_summary, consulta, respuesta
            )

        if not st.session_state.mensajes:
            st.markdown("### Consultas sugeridas")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                - Que se le imputa a Oliva Guerrero?
                - Cual es el rol de Oliva en la organizacion?
                - Que pruebas hay contra Oliva Guerrero?
                - Que dijo Oliva en su declaracion del 09/04/2024?
                """)
            with col2:
                st.markdown("""
                - Quienes son los co-investigados?
                - Que es la Ley 32108 y como afecta el caso?
                - Que hizo el juez con la excepcion de Herrera Vasquez?
                - Cual es la estructura de la organizacion criminal?
                """)

    # --- TAB: BÚSQUEDA DIRECTA ---
    with tab_busqueda:
        st.markdown("Busqueda directa en los documentos sin IA. Muestra los fragmentos mas relevantes.")

        busqueda = st.text_input("Buscar en el expediente:", key="busqueda_directa")

        col_filtro1, col_filtro2 = st.columns(2)
        with col_filtro1:
            num_resultados = st.slider("Resultados a mostrar", 3, 20, 10)
        with col_filtro2:
            filtro_tipo = st.selectbox("Filtrar por tipo", ["Todos"] + list(stats["tipos_documento"].keys()))

        if busqueda:
            resultados = buscar_documentos(busqueda, modelo, index, chunks, top_k=num_resultados * 2)
            if filtro_tipo != "Todos":
                resultados = [r for r in resultados if r.get("tipo_documento") == filtro_tipo]
            resultados = resultados[:num_resultados]

            st.markdown(f"**{len(resultados)} resultados encontrados**")

            for i, r in enumerate(resultados):
                with st.expander(
                    f"[{r['score']:.3f}] {r['archivo_original']} - Pag. {r['pagina']} ({r['tipo_documento']})",
                    expanded=(i < 3)
                ):
                    personas = r.get("personas_mencionadas", [])
                    if personas:
                        st.markdown(f"**Personas:** {', '.join(personas)}")
                    st.markdown(r["texto"])
                    st.caption(f"Chunk: {r['chunk_id']} | Documento: {r['documento_id']}")

    # --- TAB: PERSONAS ---
    with tab_personas:
        st.markdown("Todas las personas detectadas en los documentos del caso.")

        todas_personas = {}
        if PROCESADOS_DIR.exists():
            for json_path in sorted(PROCESADOS_DIR.glob("*.json")):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                    for nombre, info in doc.get("personas", {}).items():
                        if nombre not in todas_personas:
                            todas_personas[nombre] = {"dni": info.get("dni"), "frecuencia_total": 0, "documentos": []}
                        todas_personas[nombre]["frecuencia_total"] += info.get("frecuencia", 0)
                        todas_personas[nombre]["documentos"].append(doc["archivo_original"])
                        if info.get("dni") and not todas_personas[nombre]["dni"]:
                            todas_personas[nombre]["dni"] = info["dni"]
                except Exception:
                    continue

        if not todas_personas:
            st.info("No hay datos detallados de personas disponibles en esta versión (los archivos procesados no están sincronizados).")
        else:
            personas_ordenadas = sorted(todas_personas.items(), key=lambda x: x[1]["frecuencia_total"], reverse=True)
            filtro_persona = st.text_input("Filtrar por nombre:", key="filtro_persona")

            for nombre, info in personas_ordenadas:
                if filtro_persona and filtro_persona.lower() not in nombre.lower():
                    continue
                es_defendido = "oliva" in nombre.lower()
                prefijo = "**[DEFENDIDO]** " if es_defendido else ""
                dni_str = f" (DNI: {info['dni']})" if info.get("dni") else ""
                docs_str = ", ".join(set(info["documentos"]))
                st.markdown(
                    f"{prefijo}**{nombre}**{dni_str} - "
                    f"{info['frecuencia_total']} menciones en {len(set(info['documentos']))} documentos"
                )
                st.caption(f"Documentos: {docs_str}")


if __name__ == "__main__":
    main()
