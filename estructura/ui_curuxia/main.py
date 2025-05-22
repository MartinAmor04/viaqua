import streamlit as st
import pandas as pd
from modules.sql_queries import get_alerts, edit_alert
from modules.audio_conversion import base64_to_audio
from streamlit_extras.card import card
from streamlit_extras.badges import badge 
from streamlit_extras.great_tables import great_tables 
from streamlit_extras.bottom_container import bottom 
from streamlit_extras.row import row 
import numpy as np
st.set_page_config(
    page_title="CuruxIA",  
    page_icon="./assets/img/favicon.png", 
    layout="centered",  
    initial_sidebar_state="auto"
)

def aplicar_css(ruta_css: str):
    with open(ruta_css) as f:
        css = f.read()
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Zen+Dots&display=swap');

            html, body, .stApp, [class^="css"] {{
                font-family: 'Roboto', sans-serif !important;
            }}

            {css}
        </style>
    """, unsafe_allow_html=True)
aplicar_css("assets/styles.css")


# ID de la fila que se está editando
edit_id = st.session_state.get("edit_id", None)

# Lista de tipos de avería
tipos_averia = ["Fallo eléctrico", "Fallo mecánico", "Sobrecalentamiento", "Otro"]


col1, col2= st.columns([0.7,0.3])
with col1:
    st.markdown('<h1 class="titulo-principal">Curux<span>IA</span></h1>', unsafe_allow_html=True)
with col2:
    st.image("assets/img/favicon.png", width=150)
st.markdown("#### O aparello que supervisa todas as túas máquinas")


st.divider()
st.subheader("Xestión de alertas")

with st.container():

    col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 2])
    with col1:
        st.markdown('<div class="header-row">Máquina</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="header-row">Día y hora</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="header-row">Lugar</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="header-row">Grabación</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="header-row">Tipo de avería</div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="header-row">Acción</div>', unsafe_allow_html=True)


    for idx, row in enumerate(get_alerts()):
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 2])
        with col1:
            st.text(row["public_id"])
        with col2:
            st.text(row["date_time"])
        with col3:
            st.text(row["place"])
        with col4:
            st.audio(base64_to_audio(row["audio_record"]), format="audio/wav", loop=False)
        with col5:
            if edit_id == row['id']:
                new_value = st.selectbox("Tipo avería", tipos_averia, key=f"select_{row['id']}")
            else:
                st.text(row["alert_type"])
        with col6:
            if edit_id == row['id']:
                if st.button("Guardar", key=f"save_{row['id']}"):
                    edit_alert(row['id'], new_value)
                    st.session_state.edit_id = None
                    st.rerun()
            else:
                if st.button("Editar", key=f"edit_{row['id']}"):
                    st.session_state.edit_id = row['id']
                    st.rerun()

st.divider()
card(text='olaa', title="Tarjeta con Contenedor")
badge(type="streamlit", url="https://plost.streamlitapp.com")


st.write("This is the main container")

with bottom():
    st.write("This is the bottom container")
    st.text_input("This is a text input in the bottom container")

