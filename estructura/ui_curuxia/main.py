import streamlit as st
import pandas as pd
import plotly.express as px
from modules.sql_queries import get_alerts
from modules.audio_conversion import base64_to_audio

st.set_page_config(
    page_title="CuruxIA",
    page_icon="./assets/img/favicon.png",
    layout="wide"
)

# Cargar CSS
def aplicar_css(ruta_css: str):
    with open(ruta_css) as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

aplicar_css("assets/styles.css")

# Encabezado principal
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.markdown('<h1 class="titulo-principal">Curux<span>IA</span></h1>', unsafe_allow_html=True)
with col2:
    st.image("assets/img/favicon.png", width=150)

st.markdown("#### O aparello que supervisa todas as túas máquinas")
st.markdown('<h2 class="titulo-principal seccion-titulo">Xestión de alertas</h2>', unsafe_allow_html=True)

# Cargar datos iniciales
alertas = get_alerts()
df = pd.DataFrame(alertas)

# Obtener lista de tipos de máquinas únicas para el filtro
tipos_disponibles = ["Todos"] + sorted(df["machine_type"].unique())

# Filtros dinámicos
colf1, colf2, colf3 = st.columns([2, 2, 2])
with colf1:
    mes_filtro = st.selectbox("Mes", ["Todos"] + list(range(1, 13)))
with colf2:
    estado_filtro = st.selectbox("Estado", ["Activas", "Pendiente", "En revisión", "Arreglada", "Todas"], index=0)
with colf3:
    tipo_filtro = st.selectbox("Tipo de máquina", tipos_disponibles)

# Aplicar filtros
alertas_filtradas = get_alerts(estado_filtro)
df_filtrado = pd.DataFrame(alertas_filtradas)
print(df_filtrado.columns)
if mes_filtro != "Todos":
    df_filtrado = df_filtrado[pd.to_datetime(df_filtrado["date_time"]).dt.month == int(mes_filtro)]
if tipo_filtro != "Todos":
    df_filtrado = df_filtrado[df_filtrado["machine_type"] == tipo_filtro]

# Renombrar columnas
df_filtrado = df_filtrado.rename(columns={
    "machine_id": "ID Máquina",
    "public_id": "Máquina",
    "machine_type": "Tipo",
    "date_time": "Fecha y hora",
    "place": "Ubicación",
    "alert_type": "Tipo de avería",
    "estado": "Estado"
})

# **Construcción de la tabla con formato mejorado**
st.markdown('<div class="tabla-container">', unsafe_allow_html=True)

# **Encabezado de la tabla**
st.markdown("""
<table class="styled-table">
    <thead>
        <tr>
            <th>ID Máquina</th>
            <th>Máquina</th>
            <th>Tipo</th>
            <th>Fecha y hora</th>
            <th>Ubicación</th>
            <th>Tipo de avería</th>
            <th>Estado</th>
            <th>Acciones</th>
        </tr>
    </thead>
    <tbody>
""", unsafe_allow_html=True)

# **Filas de la tabla**
for index, row in df_filtrado.iterrows():
    st.markdown("<tr>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1: st.markdown(f"<td>{row['ID Máquina']}</td>", unsafe_allow_html=True)
    with col2: st.markdown(f"<td>{row['Máquina']}</td>", unsafe_allow_html=True)
    with col3: st.markdown(f"<td>{row['Tipo']}</td>", unsafe_allow_html=True)
    with col4: st.markdown(f"<td>{row['Fecha y hora']}</td>", unsafe_allow_html=True)
    with col5: st.markdown(f"<td>{row['Ubicación']}</td>", unsafe_allow_html=True)
    with col6: st.markdown(f"<td>{row['Tipo de avería']}</td>", unsafe_allow_html=True)
    with col7: st.markdown(f"<td>{row['Estado']}</td>", unsafe_allow_html=True)

    with col8:
        col_audio, col_editar = st.columns(2)
        col_audio.button("🔊", key=f"audio_{row['ID Máquina']}_{index}")
        col_editar.button("✏️", key=f"edit_{row['ID Máquina']}_{index}")
        st.audio(base64_to_audio(row['audio_record']), format="wav")


    st.markdown("</tr>", unsafe_allow_html=True)

st.markdown("</tbody></table>", unsafe_allow_html=True)

# **Dashboard restaurado**
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
st.markdown('<h2 class="titulo-principal seccion-titulo">Dashboard</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Gráfico de pastel
with col1:
    tipo_counts = df_filtrado["Tipo de avería"].value_counts()
    fig1 = px.pie(tipo_counts, values=tipo_counts.values, names=tipo_counts.index, 
                  title="Distribución de tipos de avería", color_discrete_sequence=["#FB8500"])
    st.plotly_chart(fig1, use_container_width=True)

# Gráfico de línea
with col2:
    monthly = df_filtrado.groupby(pd.to_datetime(df_filtrado["Fecha y hora"]).dt.month).size().reindex(range(1, 13), fill_value=0)
    fig2 = px.line(monthly, labels={"value": "Cantidad"}, title="Evolución de averías por mes", color_discrete_sequence=["#FB8500"])
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.area(df_filtrado.groupby("ID Máquina").size(), labels={'index': 'ID Máquina', 'value': 'Histórico'}, 
               title="Histórico de alertas por máquina", color_discrete_sequence=["#FB8500"])
st.plotly_chart(fig3, use_container_width=True)
