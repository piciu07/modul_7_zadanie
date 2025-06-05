import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from pycaret.clustering import load_model, predict_model
import plotly.express as px

st.set_page_config(page_title="Znajdź swoich ludzi", layout="wide")

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_DESC = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_data():
    model = get_model()
    df = pd.read_csv(DATA, sep=';')
    return predict_model(model, data=df)

@st.cache_data
def get_cluster_descriptions():
    with open(CLUSTER_DESC, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_unique(df, column):
    return ["Wybierz"] + sorted(df[column].dropna().astype(str).unique())

model = get_model()
df = get_all_data()
descriptions = get_cluster_descriptions()

required_features = list(model.feature_names_in_)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Powiedz coś o sobie")
    st.markdown("Wybierz dowolne z poniższych cech. Im więcej podasz, tym trafniejsze dopasowanie!")

    user_input = {}
    for col in required_features:
        options = get_unique(df, col)
        selected = st.selectbox(col.replace('_', ' ').capitalize(), options, key=col)

        # Podpowiedź dla użytkownika
        if selected != "Wybierz":
            user_input[col] = selected

        # Obrazki — pod kategorią
        if selected != "Wybierz":
            image_name = selected if col != "fav_animals" or selected != "Inne" else "other_pets"
            image_path = f"jpg/{image_name}.jpg"
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)

# Walidacja: czy użytkownik wybrał cokolwiek
if not user_input:
    st.warning("⚠️ Wybierz przynajmniej jedną cechę z lewej strony, aby kontynuować.")
else:
    st.title("🔍 Znajdź swoich ludzi")

    # Uzupełnij brakujące cechy NaN-ami, by model mógł zadziałać
    full_input = {col: user_input.get(col, np.nan) for col in required_features}
    person_df = pd.DataFrame([full_input])

    cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    same_cluster_df = df[df["Cluster"] == cluster_id]
    description = descriptions[str(cluster_id)]

    st.markdown(f"🧠 **Opis grupy:** {description['description']}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("👥 Liczba osób w Twojej grupie", len(same_cluster_df))
    with col2:
        st.info("Porównaj się z innymi osobami o podobnych zainteresowaniach!")

    st.divider()
    st.subheader("📊 Statystyki grupy")

    tabs = st.tabs(["Wiek", "Wykształcenie", "Zwierzęta", "Miejsca", "Płeć"])

    if "age" in same_cluster_df.columns:
        with tabs[0]:
            fig = px.histogram(same_cluster_df, x="age", color="age", title="Rozkład wieku")
            st.plotly_chart(fig, use_container_width=True)
    if "edu_level" in same_cluster_df.columns:
        with tabs[1]:
            fig = px.histogram(same_cluster_df, x="edu_level", color="edu_level", title="Rozkład wykształcenia")
            st.plotly_chart(fig, use_container_width=True)
    if "fav_animals" in same_cluster_df.columns:
        with tabs[2]:
            fig = px.histogram(same_cluster_df, x="fav_animals", color="fav_animals", title="Ulubione zwierzęta")
            st.plotly_chart(fig, use_container_width=True)
    if "fav_place" in same_cluster_df.columns:
        with tabs[3]:
            fig = px.histogram(same_cluster_df, x="fav_place", color="fav_place", title="Ulubione miejsca")
            st.plotly_chart(fig, use_container_width=True)
    if "gender" in same_cluster_df.columns:
        with tabs[4]:
            fig = px.pie(same_cluster_df, names="gender", title="Płeć w grupie", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
