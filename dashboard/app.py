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
import os
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
# SESSION STATE
# ─────────────────────────────────────────────
if "analyse_lancee" not in st.session_state:
    st.session_state.analyse_lancee = False

# ─────────────────────────────────────────────
# CONNEXION GEE
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


# ─────────────────────────────────────────────
# CHARGEMENT MODÈLES
# ─────────────────────────────────────────────
@st.cache_resource
def charger_modeles():

    base = os.path.dirname(__file__)

    with open(os.path.join(base, "models", "random_forest_v1.pkl"), "rb") as f:
        rf = pickle.load(f)

    with open(os.path.join(base, "models", "xgboost_v1.pkl"), "rb") as f:
        xgb = pickle.load(f)

    with open(os.path.join(base, "models", "scaler_v1.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return rf, xgb, scaler


gee_ok = init_gee()

try:
    rf, xgb, scaler = charger_modeles()
except Exception as e:
    st.error(f"Erreur chargement modèles : {e}")
    st.stop()

# ─────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────
NOMS_LABELS = {
    0: "✅ Pas de stress",
    1: "⚠️ Risque",
    2: "🟠 Stress modéré",
    3: "🔴 Stress sévère"
}

COULEURS_LABELS = {
    0: "#27AE60",
    1: "#F39C12",
    2: "#E67E22",
    3: "#C0392B"
}

# ─────────────────────────────────────────────
# TITRE
# ─────────────────────────────────────────────
st.title("🌿 Détection du Stress Hydrique")
st.markdown("### Irrigation intelligente via Sentinel-2 & Données météo")
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
        max_value=datetime(2024, 12, 31)
    )

    date_fin = st.date_input(
        "Date fin",
        value=datetime(2024, 3, 31),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 12, 31)
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

    st.subheader("🤖 Modèle IA")

    modele_choisi = st.radio(
        "Choisir le modèle",
        [
            "Random Forest",
            "XGBoost",
            "Comparer les deux"
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

    # BOUTON ANALYSER
    if st.button(
        "🔍 Analyser",
        type="primary",
        width="stretch"
    ):
        st.session_state.analyse_lancee = True

    # RESET
    if st.button(
        "♻️ Réinitialiser",
        width="stretch"
    ):
        st.session_state.analyse_lancee = False

analyser = st.session_state.analyse_lancee

# ─────────────────────────────────────────────
# CARTE
# ─────────────────────────────────────────────
st.subheader("🗺️ Carte Sentinel-2 Interactive")

col_carte, col_info = st.columns([2, 1])

with col_carte:

    m = folium.Map(
        location=[lat_centre, lon_centre],
        zoom_start=11,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    if gee_ok and analyser:

        with st.spinner("⏳ Chargement Sentinel-2..."):

            try:

                zone = ee.Geometry.Point(
                    [lon_centre, lat_centre]
                ).buffer(buffer_km * 1000)

                def masquer(img):

                    scl = img.select("SCL")

                    mk = (
                        scl.eq(4)
                        .Or(scl.eq(5))
                        .Or(scl.eq(6))
                        .Or(scl.eq(7))
                    )

                    return img.updateMask(mk).divide(10000)

                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(zone)
                    .filterDate(str(date_debut), str(date_fin))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                    .map(masquer)
                )

                img = collection.median()

                # RGB
                if indice == "RGB (couleurs naturelles)":

                    viz = {
                        "bands": ["B4", "B3", "B2"],
                        "min": 0,
                        "max": 0.3,
                        "gamma": 1.4
                    }

                    img2 = img

                # NIR
                elif indice == "NIR (fausses couleurs)":

                    viz = {
                        "bands": ["B8", "B4", "B3"],
                        "min": 0,
                        "max": 0.4,
                        "gamma": 1.4
                    }

                    img2 = img

                # NDVI
                elif indice == "NDVI":

                    img2 = img.normalizedDifference(["B8", "B4"])

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

                # NDWI
                elif indice == "NDWI":

                    img2 = img.normalizedDifference(["B3", "B8"])

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

                # MSI
                else:

                    img2 = img.select("B11").divide(img.select("B8"))

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

                tile_url = img2.getMapId(viz)["tile_fetcher"].url_format

                folium.TileLayer(
                    tiles=tile_url,
                    attr="Google Earth Engine",
                    overlay=True,
                    name=indice
                ).add_to(m)

                folium.Circle(
                    location=[lat_centre, lon_centre],
                    radius=buffer_km * 1000,
                    color="#97BC62",
                    fill=True,
                    fill_opacity=0.15
                ).add_to(m)

                folium.LayerControl().add_to(m)

                st.success(f"✅ {indice} chargé avec succès")

            except Exception as e:
                st.error(f"Erreur Sentinel-2 : {e}")

    st_folium(
        m,
        width=700,
        height=500
    )

# ─────────────────────────────────────────────
# INFOS
# ─────────────────────────────────────────────
with col_info:

    st.subheader("ℹ️ Informations")

    st.info(f"""
    **Zone analysée**

    - Latitude : {lat_centre}
    - Longitude : {lon_centre}
    - Rayon : {buffer_km} km

    **Période**

    - Début : {date_debut}
    - Fin : {date_fin}

    **Indice :** {indice}

    **Modèle :** {modele_choisi}
    """)

    if gee_ok:
        st.success("🛰️ GEE connecté")
    else:
        st.error("❌ GEE non connecté")

# ─────────────────────────────────────────────
# MÉTÉO
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
            "et0_fao_evapotranspiration",
            "windspeed_10m_max"
        ],
        "timezone": "Africa/Casablanca"
    }

    r = requests.get(url, params=params)

    if r.status_code == 200:

        d = r.json()["daily"]

        df = pd.DataFrame(d)

        df["time"] = pd.to_datetime(df["time"])

        df["deficit"] = (
            df["precipitation_sum"]
            - df["et0_fao_evapotranspiration"]
        )

        df["amplitude"] = (
            df["temperature_2m_max"]
            - df["temperature_2m_min"]
        )

        return df

    return None

# ─────────────────────────────────────────────
# ANALYSE
# ─────────────────────────────────────────────
if analyser:

    with st.spinner("⏳ Chargement météo..."):

        df_meteo = get_meteo(
            lat_centre,
            lon_centre,
            date_debut,
            date_fin
        )

    if df_meteo is not None:

        # KPIs
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Temp. max moyenne",
            f"{df_meteo['temperature_2m_max'].mean():.1f}°C"
        )

        c2.metric(
            "Précipitations",
            f"{df_meteo['precipitation_sum'].sum():.0f} mm"
        )

        c3.metric(
            "ETo moyenne",
            f"{df_meteo['et0_fao_evapotranspiration'].mean():.2f} mm/j"
        )

        deficit_moy = df_meteo["deficit"].mean()

        c4.metric(
            "Déficit moyen",
            f"{deficit_moy:.2f} mm/j"
        )

        # ─────────────────────────
        # TEMPÉRATURES
        # ─────────────────────────
        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(
            x=df_meteo["time"],
            y=df_meteo["temperature_2m_max"],
            name="Temp max"
        ))

        fig1.add_trace(go.Scatter(
            x=df_meteo["time"],
            y=df_meteo["temperature_2m_min"],
            name="Temp min"
        ))

        fig1.update_layout(
            title="Évolution des températures"
        )

        st.plotly_chart(
            fig1,
            width="stretch"
        )

        # ─────────────────────────
        # FEATURES IA
        # ─────────────────────────
        temp_max = df_meteo["temperature_2m_max"].mean()
        eto = df_meteo["et0_fao_evapotranspiration"].mean()
        amplitude = df_meteo["amplitude"].mean()
        deficit_j = df_meteo["deficit"].mean()

        deficit_cum = df_meteo["deficit"].sum()

        hsi = (
            df_meteo["temperature_2m_max"] > 35
        ).sum()

        ndvi_est = max(
            0.1,
            0.6 - max(0, -deficit_j) * 0.05
        )

        ndwi_est = -0.3 - max(0, -deficit_j) * 0.03

        msi_est = min(
            1.5,
            0.7 + max(0, -deficit_j) * 0.04
        )

        ndre_est = max(0.05, ndvi_est * 0.65)

        cri_est = max(1.0, 8.0 - ndvi_est * 10)

        lai_est = max(0.1, ndvi_est * 4.5)

        spi_30j = max(
            -2,
            min(2, deficit_cum / 50)
        )

        X_pred = pd.DataFrame(
            [[
                ndvi_est,
                ndwi_est,
                msi_est,
                ndre_est,
                cri_est,
                lai_est,
                spi_30j,
                deficit_cum,
                float(hsi),
                temp_max,
                eto,
                amplitude
            ]],
            columns=[
                "NDVI",
                "NDWI",
                "MSI",
                "NDRE",
                "CRI",
                "LAI",
                "SPI_30j",
                "deficit_cum_30j",
                "HSI_cum_30j",
                "temp_max",
                "ETo",
                "amplitude_thermique"
            ]
        )

        X_pred_sc = scaler.transform(X_pred)

        # ─────────────────────────
        # PREDICTIONS
        # ─────────────────────────
        pred_rf = rf.predict(X_pred_sc)[0]
        prob_rf = rf.predict_proba(X_pred_sc)[0]

        pred_xgb = xgb.predict(X_pred_sc)[0]
        prob_xgb = xgb.predict_proba(X_pred_sc)[0]

        if modele_choisi == "Random Forest":

            predictions = [
                ("Random Forest", pred_rf, prob_rf)
            ]

        elif modele_choisi == "XGBoost":

            predictions = [
                ("XGBoost", pred_xgb, prob_xgb)
            ]

        else:

            predictions = [
                ("Random Forest", pred_rf, prob_rf),
                ("XGBoost", pred_xgb, prob_xgb)
            ]

        # ─────────────────────────
        # AFFICHAGE
        # ─────────────────────────
        st.divider()

        st.subheader("🤖 Prédiction IA")

        for nom_modele, pred, prob in predictions:

            st.markdown(f"## {nom_modele}")

            col1, col2 = st.columns([1, 2])

            with col1:

                confiance = int(prob[min(pred, 3)] * 100)

                st.metric(
                    "Confiance",
                    f"{confiance}%"
                )

                st.progress(confiance / 100)

            with col2:

                st.markdown(
                    f"### {NOMS_LABELS[pred]}"
                )

            fig_prob = go.Figure(go.Bar(
                x=[NOMS_LABELS[i] for i in range(4)],
                y=[p * 100 for p in prob],
                marker_color=[
                    COULEURS_LABELS[i]
                    for i in range(4)
                ]
            ))

            fig_prob.update_layout(
                title="Probabilités par classe",
                height=300
            )

            st.plotly_chart(
                fig_prob,
                width="stretch"
            )

            # RECOMMANDATIONS
            if pred == 0:

                st.success(
                    "✅ Aucune irrigation nécessaire."
                )

            elif pred == 1:

                eau = abs(deficit_j) * 0.7 * 7

                st.warning(
                    f"""
                    ⚠️ Irrigation préventive recommandée.

                    💧 Quantité estimée :
                    {eau:.1f} mm
                    """
                )

            elif pred == 2:

                eau = abs(deficit_j) * 0.8 * 3

                st.warning(
                    f"""
                    🟠 Irrigation recommandée sous 48h.

                    💧 Quantité estimée :
                    {eau:.1f} mm
                    """
                )

            else:

                eau = abs(deficit_j)

                st.error(
                    f"""
                    🔴 IRRIGATION URGENTE

                    💧 Quantité estimée :
                    {eau:.1f} mm/jour
                    """
                )

            st.divider()

else:

    st.info(
        "👈 Configurez les paramètres puis cliquez sur Analyser."
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "---\n"
    "### PFE Agriculture 4.0\n"
    "Badara Aliou Guindo — 2026"
)
