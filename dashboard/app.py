import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import ee
import requests
import json
import pickle

from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stress Hydrique Sentinel-2",
    page_icon="🌿",
    layout="wide"
)

# ─────────────────────────────────────────────
# INITIALISATION GEE
# ─────────────────────────────────────────────
@st.cache_resource
def init_gee():
    try:
        key_data = dict(st.secrets["gee"])

        credentials = ee.ServiceAccountCredentials(
            email=key_data["client_email"],
            key_data=json.dumps(key_data)
        )

        ee.Initialize(credentials)
        return True

    except Exception as e:
        st.error(f"Erreur GEE : {e}")
        return False

gee_ok = init_gee()

# ─────────────────────────────────────────────
# CHARGEMENT MODELES IA
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        model_dir = "dashboard/models"

        with open(f"{model_dir}/random_forest_v1.pkl", "rb") as f:
            rf_model = pickle.load(f)

        with open(f"{model_dir}/xgboost_v1.pkl", "rb") as f:
            xgb_model = pickle.load(f)

        with open(f"{model_dir}/scaler_v1.pkl", "rb") as f:
            scaler = pickle.load(f)

        return rf_model, xgb_model, scaler

    except Exception as e:
        st.error(f"Erreur chargement modèles : {e}")
        return None, None, None

rf_model, xgb_model, scaler = load_models()

# ─────────────────────────────────────────────
# TITRE
# ─────────────────────────────────────────────
st.title("🌿 Détection du Stress Hydrique")
st.markdown("### Irrigation intelligente via Sentinel-2 & Données Météo")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.header("⚙️ Paramètres")

    st.subheader("📅 Période")

    date_debut = st.date_input(
        "Date début",
        value=datetime(2024, 1, 1),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    )

    date_fin = st.date_input(
        "Date fin",
        value=datetime(2024, 3, 31),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    )

    st.subheader("🛰️ Indice spectral")

    indice = st.selectbox(
        "Indice à afficher",
        [
            "RGB (couleurs naturelles)",
            "NIR (fausses couleurs)",
            "NDVI",
            "NDWI",
            "MSI"
        ]
    )

    st.subheader("📍 Zone d'étude")

    lat_centre = st.number_input(
        "Latitude centre",
        value=34.50,
        format="%.4f"
    )

    lon_centre = st.number_input(
        "Longitude centre",
        value=-6.275,
        format="%.4f"
    )

    buffer_km = st.slider(
        "Rayon (km)",
        5,
        50,
        15
    )

    analyser = st.button(
        "🔍 Analyser la zone",
        type="primary",
        use_container_width=True
    )

# ─────────────────────────────────────────────
# CARTE
# ─────────────────────────────────────────────
st.subheader("🗺️ Carte interactive")

col_carte, col_info = st.columns([2, 1])

with col_carte:

    m = folium.Map(
        location=[lat_centre, lon_centre],
        zoom_start=10,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    if gee_ok and analyser:

        with st.spinner("Chargement Sentinel-2..."):

            try:

                zone = ee.Geometry.Point(
                    [lon_centre, lat_centre]
                ).buffer(buffer_km * 1000)

                def masquer_nuages(img):

                    scl = img.select("SCL")

                    mask = (
                        scl.eq(4)
                        .Or(scl.eq(5))
                        .Or(scl.eq(6))
                        .Or(scl.eq(7))
                    )

                    return img.updateMask(mask).divide(10000)

                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(zone)
                    .filterDate(str(date_debut), str(date_fin))
                    .filter(
                        ee.Filter.lt(
                            "CLOUDY_PIXEL_PERCENTAGE",
                            20
                        )
                    )
                    .map(masquer_nuages)
                )

                image = collection.median()

                if indice == "RGB (couleurs naturelles)":

                    viz = {
                        "bands": ["B4", "B3", "B2"],
                        "min": 0,
                        "max": 0.3,
                        "gamma": 1.4
                    }

                elif indice == "NIR (fausses couleurs)":

                    viz = {
                        "bands": ["B8", "B4", "B3"],
                        "min": 0,
                        "max": 0.4,
                        "gamma": 1.4
                    }

                elif indice == "NDVI":

                    image = image.normalizedDifference(
                        ["B8", "B4"]
                    )

                    viz = {
                        "min": -0.2,
                        "max": 0.8,
                        "palette": [
                            "d73027",
                            "f46d43",
                            "fdae61",
                            "ffffbf",
                            "a6d96a",
                            "1a9850"
                        ]
                    }

                elif indice == "NDWI":

                    image = image.normalizedDifference(
                        ["B3", "B8"]
                    )

                    viz = {
                        "min": -0.8,
                        "max": 0.2,
                        "palette": [
                            "d73027",
                            "f46d43",
                            "ffffbf",
                            "74add1",
                            "4575b4"
                        ]
                    }

                elif indice == "MSI":

                    image = image.select("B11").divide(
                        image.select("B8")
                    )

                    viz = {
                        "min": 0.4,
                        "max": 1.5,
                        "palette": [
                            "1a9850",
                            "a6d96a",
                            "ffffbf",
                            "f46d43",
                            "d73027"
                        ]
                    }

                map_id = image.getMapId(viz)

                folium.TileLayer(
                    tiles=map_id["tile_fetcher"].url_format,
                    attr="Google Earth Engine",
                    overlay=True,
                    name=indice
                ).add_to(m)

                folium.Circle(
                    location=[lat_centre, lon_centre],
                    radius=buffer_km * 1000,
                    color="green",
                    fill=True,
                    fill_opacity=0.1
                ).add_to(m)

                folium.LayerControl().add_to(m)

                st.success("Image Sentinel-2 chargée")

            except Exception as e:
                st.error(f"Erreur carte : {e}")

    st_folium(m, width=700, height=500)

with col_info:

    st.subheader("📊 Informations")

    st.info(f"""
    Latitude : {lat_centre}

    Longitude : {lon_centre}

    Rayon : {buffer_km} km

    Période :
    {date_debut} → {date_fin}

    Indice :
    {indice}
    """)

    if gee_ok:
        st.success("GEE connecté")
    else:
        st.error("GEE non connecté")

# ─────────────────────────────────────────────
# METEO
# ─────────────────────────────────────────────
st.divider()

st.subheader("🌦️ Données météo")

@st.cache_data(ttl=3600)
def get_meteo(lat, lon, debut, fin):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(debut),
        "end_date": str(fin),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "et0_fao_evapotranspiration"
        ],
        "timezone": "Africa/Casablanca"
    }

    r = requests.get(url, params=params)

    if r.status_code == 200:

        data = r.json()["daily"]

        df = pd.DataFrame(data)

        df["time"] = pd.to_datetime(df["time"])

        df["deficit"] = (
            df["precipitation_sum"]
            - df["et0_fao_evapotranspiration"]
        )

        return df

    return None

df_meteo = None

if analyser:

    with st.spinner("Chargement météo..."):

        df_meteo = get_meteo(
            lat_centre,
            lon_centre,
            date_debut,
            date_fin
        )

    if df_meteo is not None:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Température max moyenne",
                f"{df_meteo['temperature_2m_max'].mean():.1f} °C"
            )

        with col2:
            st.metric(
                "Précipitations",
                f"{df_meteo['precipitation_sum'].sum():.1f} mm"
            )

        with col3:
            st.metric(
                "ETo moyenne",
                f"{df_meteo['et0_fao_evapotranspiration'].mean():.2f}"
            )

        with col4:
            st.metric(
                "Déficit moyen",
                f"{df_meteo['deficit'].mean():.2f}"
            )

        # GRAPH TEMP
        fig_temp = go.Figure()

        fig_temp.add_trace(
            go.Scatter(
                x=df_meteo["time"],
                y=df_meteo["temperature_2m_max"],
                name="Température max"
            )
        )

        fig_temp.add_trace(
            go.Scatter(
                x=df_meteo["time"],
                y=df_meteo["temperature_2m_min"],
                name="Température min"
            )
        )

        fig_temp.update_layout(
            title="Températures"
        )

        st.plotly_chart(
            fig_temp,
            use_container_width=True
        )

# ─────────────────────────────────────────────
# IA
# ─────────────────────────────────────────────
st.divider()

st.subheader("🤖 Intelligence Artificielle")

if analyser and df_meteo is not None:

    if rf_model is not None and xgb_model is not None:

        try:

            deficit_moy = df_meteo["deficit"].mean()

            temp_moy = df_meteo[
                "temperature_2m_max"
            ].mean()

            precip_total = df_meteo[
                "precipitation_sum"
            ].sum()

            et0_moy = df_meteo[
                "et0_fao_evapotranspiration"
            ].mean()

            nb_jours = len(df_meteo)

            # FEATURES
            features = pd.DataFrame([{
                "temperature_max": temp_moy,
                "precipitation": precip_total / nb_jours,
                "et0": et0_moy,
                "deficit": deficit_moy
            }])

            # SCALE
            X_scaled = scaler.transform(features)

            # PREDICTIONS
            rf_pred = rf_model.predict(X_scaled)[0]
            xgb_pred = xgb_model.predict(X_scaled)[0]

            # SCORE FINAL
            score = int(
                ((rf_pred + xgb_pred) / 2) * 100
            )

            score = max(0, min(score, 100))

            # LABEL
            if score < 25:

                label = "✅ Pas de stress"
                conseil = (
                    "Aucune irrigation nécessaire."
                )

            elif score < 50:

                label = "⚠️ Risque faible"
                conseil = (
                    "Surveillance recommandée."
                )

            elif score < 75:

                label = "🟠 Stress modéré"
                conseil = (
                    "Irrigation recommandée."
                )

            else:

                label = "🔴 Stress sévère"
                conseil = (
                    "Irrigation urgente."
                )

            col_s1, col_s2, col_s3 = st.columns(3)

            with col_s1:

                st.metric(
                    "Score IA",
                    f"{score}/100"
                )

                st.progress(score / 100)

            with col_s2:

                st.markdown("### Niveau")

                st.markdown(f"## {label}")

            with col_s3:

                st.markdown("### Recommandation")

                st.info(conseil)

                eau = abs(deficit_moy) * 0.7

                st.warning(
                    f"💧 Eau estimée : {eau:.1f} mm/j"
                )

            # DETAILS MODELES
            st.divider()

            st.subheader("📈 Résultats modèles")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Random Forest",
                    f"{rf_pred:.2f}"
                )

            with c2:
                st.metric(
                    "XGBoost",
                    f"{xgb_pred:.2f}"
                )

        except Exception as e:

            st.error(f"Erreur IA : {e}")

else:

    st.info(
        "Cliquez sur 'Analyser la zone'"
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()

st.markdown(
    """
    PFE Agriculture 4.0 —
    Détection précoce du stress hydrique
    via Sentinel-2 et Intelligence Artificielle
    """
)
