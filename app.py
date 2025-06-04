import streamlit as st
import pandas as pd
import json
import os
st.title('Znajdź znajomych')

from pycaret.clustering import load_model, predict_model
import plotly.express as px  # type: ignore

# Przypisanie modelu do zmiennej
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)
     

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())


@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters
    
@st.cache_data
def get_animals(all_df):
    unique_animals = sorted(all_df["fav_animals"].dropna().astype(str).unique())
    return unique_animals

@st.cache_data
def get_place(all_df):
    unique_place = sorted(all_df["fav_place"].dropna().astype(str).unique())
    return unique_place

@st.cache_data
def get_age(all_df):
    unique_age = sorted(all_df["age"].dropna().astype(str).unique())
    return unique_age

@st.cache_data
def get_edu(all_df):
    unique_edu = sorted(all_df["edu_level"].dropna().astype(str).unique())
    return unique_edu




model = get_model()

all_df = get_all_participants()
print(all_df)
unique_animals=get_animals(all_df)
unique_place=get_place(all_df)
unique_age=get_age(all_df)
unique_edu=get_edu(all_df)
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")

    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])
    # Sprawdzamy, czy istnieje plik JPG o nazwie zwierzęcia
    image_path = f"jpg/{gender}.jpg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    age = st.selectbox("Wiek", unique_age)
    edu_level = st.selectbox("Wykształcenie", "Wybierz wykstałcenie", unique_edu)
    fav_animals = st.selectbox("Ulubione zwierzęta", unique_animals)

    for animal in unique_animals:
        if animal == 'Inne':
            image_path = f"jpg/other_pets.jpg"
        else:
            image_path = f"jpg/{animal}.jpg"
        if fav_animals == animal and os.path.exists(image_path):
            st.image(image_path, use_container_width=True)

    fav_place = st.selectbox("Ulubione miejsce", unique_place)

    for place in unique_place:
    # Sprawdzamy, czy istnieje plik JPG o nazwie zwierzęcia
        image_path = f"jpg/{place}.jpg"
        if fav_place == place and os.path.exists(image_path):
            st.image(image_path, use_container_width=True)




    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])


cluster_names_and_descriptions = get_cluster_names_and_descriptions()


predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))





st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)
