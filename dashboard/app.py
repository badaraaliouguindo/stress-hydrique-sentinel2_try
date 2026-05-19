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
st.markdown("### Irrigation intelligente via Sentinel-2 & IA")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    st.header("⚙️ Paramètres")

    st.subheader("📅 Période")

    date_debut = st.date_input(
        "Date début",
        value=datetime(2024, 1, 1)
    )

    date_fin = st.date_input(
        "Date fin",
        value=datetime(2024, 3, 31)
    )

    st.subheader("🛰️ Indice")

    indice = st.selectbox(
        "Indice à afficher",
        [
            "RGB",
            "NIR",
            "NDVI",
            "NDWI",
            "MSI"
        ]
    )

    st.subheader("📍 Zone d'étude")

    lat_centre = st.number_input(
        "Latitude",
        value=34.50,
        format="%.4f"
    )

    lon_centre = st.number_input(
        "Longitude",
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
        "🔍 Analyser",
        type="primary",
        width="stretch"
    )

# ─────────────────────────────────────────────
# CARTE
# ─────────────────────────────────────────────
st.subheader("🗺️ Carte interactive")

col_map, col_info = st.columns([2, 1])

image = None
zone = None

with col_map:

    m = folium.Map(
        location=[lat_centre, lon_centre],
        zoom_start=10,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    if gee_ok and analyser:

        try:

            zone = ee.Geometry.Point(
                [lon_centre, lat_centre]
            ).buffer(buffer_km * 1000)

            def mask_clouds(img):

                scl = img.select("SCL")

                mask = (
                    scl.eq(4)
                    .Or(scl.eq(5))
                    .Or(scl.eq(6))
                    .Or(scl.eq(7))
                )

                return img.updateMask(mask).divide(10000)

            collection = (
                ee.ImageCollection(
                    "COPERNICUS/S2_SR_HARMONIZED"
                )
                .filterBounds(zone)
                .filterDate(
                    str(date_debut),
                    str(date_fin)
                )
                .filter(
                    ee.Filter.lt(
                        "CLOUDY_PIXEL_PERCENTAGE",
                        20
                    )
                )
                .map(mask_clouds)
            )

            image = collection.median()

            if indice == "RGB":

                viz = {
                    "bands": ["B4", "B3", "B2"],
                    "min": 0,
                    "max": 0.3
                }

            elif indice == "NIR":

                viz = {
                    "bands": ["B8", "B4", "B3"],
                    "min": 0,
                    "max": 0.4
                }

            elif indice == "NDVI":

                image = image.normalizedDifference(
                    ["B8", "B4"]
                )

                viz = {
                    "min": -0.2,
                    "max": 0.8,
                    "palette": [
                        "red",
                        "yellow",
                        "green"
                    ]
                }

            elif indice == "NDWI":

                image = image.normalizedDifference(
                    ["B3", "B8"]
                )

                viz = {
                    "min": -0.5,
                    "max": 0.5,
                    "palette": [
                        "brown",
                        "yellow",
                        "blue"
                    ]
                }

            elif indice == "MSI":

                image = image.select(
                    "B11"
                ).divide(
                    image.select("B8")
                )

                viz = {
                    "min": 0.4,
                    "max": 1.5,
                    "palette": [
                        "green",
                        "yellow",
                        "red"
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

            st.error(f"Erreur GEE : {e}")

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

    df_meteo = get_meteo(
        lat_centre,
        lon_centre,
        date_debut,
        date_fin
    )

    if df_meteo is not None:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Température max",
                f"{df_meteo['temperature_2m_max'].mean():.1f} °C"
            )

        with c2:
            st.metric(
                "Précipitations",
                f"{df_meteo['precipitation_sum'].sum():.1f} mm"
            )

        with c3:
            st.metric(
                "ETo",
                f"{df_meteo['et0_fao_evapotranspiration'].mean():.2f}"
            )

        with c4:
            st.metric(
                "Déficit",
                f"{df_meteo['deficit'].mean():.2f}"
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_meteo["time"],
                y=df_meteo["temperature_2m_max"],
                name="Température max"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_meteo["time"],
                y=df_meteo["temperature_2m_min"],
                name="Température min"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

# ─────────────────────────────────────────────
# IA
# ─────────────────────────────────────────────
st.divider()

st.subheader("🤖 Analyse IA")

if (
    analyser
    and df_meteo is not None
    and image is not None
):

    try:

        # ─────────────────────────────
        # VARIABLES METEO
        # ─────────────────────────────
        temp_max = df_meteo[
            "temperature_2m_max"
        ].mean()

        eto = df_meteo[
            "et0_fao_evapotranspiration"
        ].mean()

        amplitude = (
            df_meteo["temperature_2m_max"]
            - df_meteo["temperature_2m_min"]
        ).mean()

        # ─────────────────────────────
        # NDVI
        # ─────────────────────────────
        try:

            ndvi_img = image.normalizedDifference(
                ["B8", "B4"]
            )

            ndvi_val = ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=zone,
                scale=20
            ).getInfo()

            ndvi = list(
                ndvi_val.values()
            )[0]

        except:
            ndvi = 0

        # ─────────────────────────────
        # NDWI
        # ─────────────────────────────
        try:

            ndwi_img = image.normalizedDifference(
                ["B3", "B8"]
            )

            ndwi_val = ndwi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=zone,
                scale=20
            ).getInfo()

            ndwi = list(
                ndwi_val.values()
            )[0]

        except:
            ndwi = 0

        # ─────────────────────────────
        # MSI
        # ─────────────────────────────
        try:

            msi_img = image.select(
                "B11"
            ).divide(
                image.select("B8")
            )

            msi_val = msi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=zone,
                scale=20
            ).getInfo()

            msi = list(
                msi_val.values()
            )[0]

        except:
            msi = 0

        # ─────────────────────────────
        # FEATURES APPROXIMATIVES
        # ─────────────────────────────
        ndre = ndvi * 0.85
        cri = (1 - ndvi) * 0.5
        lai = ndvi * 3

        # ─────────────────────────────
        # FEATURES FINALES
        # ─────────────────────────────
        features = pd.DataFrame([{
            "NDVI": ndvi,
            "NDWI": ndwi,
            "MSI": msi,
            "NDRE": ndre,
            "CRI": cri,
            "LAI": lai,
            "SPI_30j": 0,
            "deficit_cum_30j": 0,
            "HSI_cum_30j": 0,
            "temp_max": temp_max,
            "ETo": eto,
            "amplitude_thermique": amplitude
        }])

        # ─────────────────────────────
        # SCALE
        # ─────────────────────────────
        X_scaled = scaler.transform(features)

        # ─────────────────────────────
        # PREDICTIONS
        # ─────────────────────────────
        rf_pred = rf_model.predict(X_scaled)[0]

        xgb_pred = xgb_model.predict(X_scaled)[0]

        score = int(
            ((rf_pred + xgb_pred) / 2) * 33
        )

        score = max(0, min(score, 100))

        # ─────────────────────────────
        # LABELS
        # ─────────────────────────────
        if score < 25:

            label = "✅ Pas de stress"

            conseil = (
                "Aucune irrigation nécessaire."
            )

        elif score < 50:

            label = "⚠️ Risque"

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

        # ─────────────────────────────
        # AFFICHAGE
        # ─────────────────────────────
        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "Score IA",
                f"{score}/100"
            )

            st.progress(score / 100)

        with s2:

            st.markdown(f"## {label}")

        with s3:

            st.info(conseil)

        # ─────────────────────────────
        # DETAILS
        # ─────────────────────────────
        st.divider()

        st.subheader("📈 Détails IA")

        d1, d2 = st.columns(2)

        with d1:

            st.metric(
                "Random Forest",
                f"{rf_pred}"
            )

            st.metric(
                "XGBoost",
                f"{xgb_pred}"
            )

        with d2:

            st.dataframe(features)

    except Exception as e:

        st.error(f"Erreur IA : {e}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()

st.markdown("""
PFE Agriculture 4.0 —
Détection précoce du stress hydrique
via Sentinel-2, météo et Intelligence Artificielle
""")
