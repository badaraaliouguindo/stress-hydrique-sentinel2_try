import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import ee
import requests
import json
from datetime import datetime, timedelta

# ── Configuration page
st.set_page_config(
    page_title="Stress Hydrique Sentinel-2",
    page_icon="🌿",
    layout="wide"
)

# ── Connexion GEE via Service Account
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

# ── Titre
st.title("🌿 Détection du Stress Hydrique")
st.markdown("**Irrigation intelligente via Sentinel-2 & Données Météo**")
st.divider()

# ── Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")

    # Sélection période
    st.subheader("📅 Période")
    date_debut = st.date_input("Date début",
                                value=datetime(2024,1,1),
                                min_value=datetime(2020,1,1),
                                max_value=datetime(2024,12,31))
    date_fin = st.date_input("Date fin",
                              value=datetime(2024,3,31),
                              min_value=datetime(2020,1,1),
                              max_value=datetime(2024,12,31))

    # Sélection indice
    st.subheader("🛰️ Indice spectral")
    indice = st.selectbox("Indice à afficher",
                          ["RGB (couleurs naturelles)",
                           "NIR (fausses couleurs)",
                           "NDVI",
                           "NDWI",
                           "MSI"])

    # Coordonnées zone
    st.subheader("📍 Zone d'étude")
    lat_centre = st.number_input("Latitude centre",  value=34.50, format="%.4f")
    lon_centre = st.number_input("Longitude centre", value=-6.275, format="%.4f")
    buffer_km  = st.slider("Rayon (km)", 5, 50, 15)

    analyser = st.button("🔍 Analyser la zone", type="primary", use_container_width=True)

# ── Carte principale
st.subheader("🗺️ Carte interactive de la zone")

col_carte, col_info = st.columns([2, 1])

with col_carte:
    # Carte Folium de base
    m = folium.Map(
        location=[lat_centre, lon_centre],
        zoom_start=11,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )

    # Ajouter couche GEE si connecté
    if gee_ok and analyser:
        with st.spinner("⏳ Chargement images Sentinel-2..."):
            try:
                zone = ee.Geometry.Point([lon_centre, lat_centre]).buffer(buffer_km * 1000)

                def masquer_nuages(img):
                    scl = img.select("SCL")
                    m_scl = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
                    return img.updateMask(m_scl).divide(10000)

                collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(zone)
                    .filterDate(str(date_debut), str(date_fin))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                    .map(masquer_nuages))

                image = collection.median()

                # Paramètres selon indice choisi
                if indice == "RGB (couleurs naturelles)":
                    viz = {"bands":["B4","B3","B2"], "min":0, "max":0.3, "gamma":1.4}
                elif indice == "NIR (fausses couleurs)":
                    viz = {"bands":["B8","B4","B3"], "min":0, "max":0.4, "gamma":1.4}
                elif indice == "NDVI":
                    ndvi = image.normalizedDifference(["B8","B4"])
                    image = ndvi
                    viz = {"min":-0.2, "max":0.8,
                           "palette":["d73027","f46d43","fdae61",
                                     "ffffbf","a6d96a","1a9850"]}
                elif indice == "NDWI":
                    ndwi = image.normalizedDifference(["B3","B8"])
                    image = ndwi
                    viz = {"min":-0.8, "max":0.2,
                           "palette":["d73027","f46d43","ffffbf",
                                     "74add1","4575b4"]}
                elif indice == "MSI":
                    msi = image.select("B11").divide(image.select("B8"))
                    image = msi
                    viz = {"min":0.4, "max":1.5,
                           "palette":["1a9850","a6d96a","ffffbf",
                                     "f46d43","d73027"]}

                # Générer URL tuiles
                map_id = image.getMapId(viz)
                tile_url = map_id["tile_fetcher"].url_format

                folium.TileLayer(
                    tiles=tile_url,
                    attr="Google Earth Engine",
                    name=f"Sentinel-2 — {indice}",
                    overlay=True
                ).add_to(m)

                # Cercle zone
                folium.Circle(
                    location=[lat_centre, lon_centre],
                    radius=buffer_km * 1000,
                    color="#97BC62",
                    fill=True,
                    fill_opacity=0.1,
                    popup=f"Zone analysée ({buffer_km}km)"
                ).add_to(m)

                folium.LayerControl().add_to(m)
                st.success(f"✅ Image {indice} chargée !")

            except Exception as e:
                st.error(f"Erreur GEE : {e}")

    # Afficher la carte
    map_result = st_folium(m, width=700, height=500)

with col_info:
    st.subheader("📊 Informations zone")
    st.info(f"""
    **Localisation**
    - Latitude  : {lat_centre}°N
    - Longitude : {lon_centre}°W
    - Rayon     : {buffer_km} km

    **Période analysée**
    - Du : {date_debut}
    - Au : {date_fin}

    **Indice sélectionné**
    - {indice}
    """)

    if gee_ok:
        st.success("🛰️ GEE connecté")
    else:
        st.error("❌ GEE non connecté")

# ── Données météo Open-Meteo
st.divider()
st.subheader("🌦️ Données Météo — Open-Meteo")

@st.cache_data(ttl=3600)
def get_meteo(lat, lon, debut, fin):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(debut), "end_date": str(fin),
        "daily": ["temperature_2m_max","temperature_2m_min",
                  "precipitation_sum","et0_fao_evapotranspiration"],
        "timezone": "Africa/Casablanca"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()["daily"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df["deficit"] = df["precipitation_sum"] - df["et0_fao_evapotranspiration"]
        return df
    return None

if analyser:
    with st.spinner("⏳ Chargement données météo..."):
        df_meteo = get_meteo(lat_centre, lon_centre, date_debut, date_fin)

    if df_meteo is not None:
        # KPIs météo
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Temp. max moy",
                      f"{df_meteo['temperature_2m_max'].mean():.1f}°C")
        with col_m2:
            st.metric("Précip. totale",
                      f"{df_meteo['precipitation_sum'].sum():.1f} mm")
        with col_m3:
            st.metric("ETo moy",
                      f"{df_meteo['et0_fao_evapotranspiration'].mean():.2f} mm/j")
        with col_m4:
            deficit_moy = df_meteo["deficit"].mean()
            st.metric("Déficit moy",
                      f"{deficit_moy:.2f} mm/j",
                      delta="Stress ⚠️" if deficit_moy < -3 else "Normal ✅")

        # Graphiques météo
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=df_meteo["time"], y=df_meteo["temperature_2m_max"],
                name="Temp. max", line=dict(color="#E74C3C")))
            fig_temp.add_trace(go.Scatter(
                x=df_meteo["time"], y=df_meteo["temperature_2m_min"],
                name="Temp. min", line=dict(color="#3498DB")))
            fig_temp.add_hline(y=35, line_dash="dash",
                               line_color="orange",
                               annotation_text="Seuil stress 35°C")
            fig_temp.update_layout(
                title="Températures",
                plot_bgcolor="#0D1F35",
                paper_bgcolor="#0A1628",
                font_color="white"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with col_g2:
            fig_precip = go.Figure()
            fig_precip.add_trace(go.Bar(
                x=df_meteo["time"], y=df_meteo["precipitation_sum"],
                name="Précipitations", marker_color="#3498DB"))
            fig_precip.add_trace(go.Scatter(
                x=df_meteo["time"],
                y=df_meteo["et0_fao_evapotranspiration"],
                name="ETo", line=dict(color="#E74C3C")))
            fig_precip.update_layout(
                title="Précipitations vs ETo",
                plot_bgcolor="#0D1F35",
                paper_bgcolor="#0A1628",
                font_color="white"
            )
            st.plotly_chart(fig_precip, use_container_width=True)

        # Déficit hydrique
        fig_def = go.Figure()
        fig_def.add_trace(go.Bar(
            x=df_meteo["time"], y=df_meteo["deficit"],
            marker_color=["#27AE60" if v >= 0 else "#E74C3C"
                         for v in df_meteo["deficit"]],
            name="Déficit hydrique"))
        fig_def.add_hline(y=0, line_color="white", line_width=1)
        fig_def.update_layout(
            title="Déficit Hydrique Journalier (Précip - ETo)",
            plot_bgcolor="#0D1F35",
            paper_bgcolor="#0A1628",
            font_color="white"
        )
        st.plotly_chart(fig_def, use_container_width=True)

# ── Score IA + Recommandations
st.divider()
st.subheader("🤖 Score IA & Recommandations d'Irrigation")

if analyser and df_meteo is not None:
    # Calcul score simplifié basé sur les données météo
    deficit_moy  = df_meteo["deficit"].mean()
    temp_moy     = df_meteo["temperature_2m_max"].mean()
    precip_total = df_meteo["precipitation_sum"].sum()
    nb_jours     = len(df_meteo)

    # Score 0-100 (100 = stress maximal)
    score = 0
    if deficit_moy < -8:   score += 35
    elif deficit_moy < -4: score += 20
    elif deficit_moy < 0:  score += 10

    if temp_moy > 35:      score += 30
    elif temp_moy > 30:    score += 20
    elif temp_moy > 25:    score += 10

    if precip_total/nb_jours < 0.5: score += 25
    elif precip_total/nb_jours < 1: score += 15
    elif precip_total/nb_jours < 2: score += 5

    score = min(score, 100)

    # Label
    if score < 25:
        label    = "✅ Pas de stress"
        couleur  = "green"
        conseil  = "Aucune irrigation nécessaire. Surveiller dans 15 jours."
    elif score < 50:
        label    = "⚠️ Risque de stress"
        couleur  = "orange"
        conseil  = "Irrigation préventive recommandée dans les 7 jours."
    elif score < 75:
        label    = "🟠 Stress modéré"
        couleur  = "orange"
        conseil  = "Irrigation recommandée dans les 48 heures."
    else:
        label    = "🔴 Stress sévère"
        couleur  = "red"
        conseil  = "Irrigation URGENTE. Intervenir dans les 24 heures."

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.metric("Score de stress", f"{score}/100")
        st.progress(score/100)

    with col_s2:
        st.markdown(f"### Niveau de stress")
        st.markdown(f"## {label}")

    with col_s3:
        st.markdown("### Recommandation")
        st.info(conseil)

        # Quantité d'eau estimée
        if score >= 50:
            eau = abs(deficit_moy) * 0.7
            st.warning(f"💧 Quantité estimée : **{eau:.1f} mm/j**")

else:
    st.info("👆 Définissez une zone et cliquez sur **Analyser la zone** pour obtenir le score IA.")

# ── Footer
st.divider()
st.markdown("*PFE Agriculture 4.0 — Détection stress hydrique Sentinel-2 | Avril 2026*")
