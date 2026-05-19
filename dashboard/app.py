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
import pickle
import os
from datetime import datetime

# ── Configuration page
st.set_page_config(
    page_title="Stress Hydrique Sentinel-2",
    page_icon="🌿",
    layout="wide"
)

# ── Connexion GEE
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

# ── Charger les modèles
@st.cache_resource
def charger_modeles():
    base = os.path.dirname(__file__)
    with open(os.path.join(base,"models","random_forest_v1.pkl"),"rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(base,"models","xgboost_v1.pkl"),"rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(base,"models","scaler_v1.pkl"),"rb") as f:
        scaler = pickle.load(f)
    return rf, xgb, scaler

gee_ok     = init_gee()
rf, xgb, scaler = charger_modeles()

# ── Features attendues par les modèles
FEATURES = ["NDVI","NDWI","MSI","NDRE","CRI","LAI",
            "SPI_30j","deficit_cum_30j","HSI_cum_30j",
            "temp_max","ETo","amplitude_thermique"]

NOMS_LABELS = {
    0:"✅ Pas de stress",
    1:"⚠️ Risque",
    2:"🟠 Stress modéré",
    3:"🔴 Stress sévère"
}
COULEURS_LABELS = {
    0:"#27AE60", 1:"#F39C12",
    2:"#E67E22", 3:"#C0392B"
}

# ── Titre
st.title("🌿 Détection du Stress Hydrique")
st.markdown("**Irrigation intelligente via Sentinel-2 & Données Météo**")
st.divider()

# ── Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")

    st.subheader("📅 Période")
    date_debut = st.date_input("Date début",
                                value=datetime(2024,1,1),
                                min_value=datetime(2020,1,1),
                                max_value=datetime(2024,12,31))
    date_fin = st.date_input("Date fin",
                              value=datetime(2024,3,31),
                              min_value=datetime(2020,1,1),
                              max_value=datetime(2024,12,31))

    st.subheader("🛰️ Indice spectral")
    indice = st.selectbox("Indice à afficher",
                          ["RGB (couleurs naturelles)",
                           "NIR (fausses couleurs)",
                           "NDVI","NDWI","MSI"])

    st.subheader("🤖 Modèle IA")
    modele_choisi = st.radio("Choisir le modèle",
                              ["Random Forest","XGBoost","Comparer les deux"])

    st.subheader("📍 Zone d'étude")
    lat_centre = st.number_input("Latitude",  value=34.50, format="%.4f")
    lon_centre = st.number_input("Longitude", value=-6.275, format="%.4f")
    buffer_km  = st.slider("Rayon (km)", 5, 50, 15)

    analyser = st.button("🔍 Analyser",
                         type="primary",
                         use_container_width=True)

# ── Carte
st.subheader("🗺️ Carte Sentinel-2 Interactive")
col_carte, col_info = st.columns([2,1])

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
                zone = ee.Geometry.Point([lon_centre, lat_centre]).buffer(buffer_km*1000)

                def masquer(img):
                    scl = img.select("SCL")
                    mk = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
                    return img.updateMask(mk).divide(10000)

                col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                       .filterBounds(zone)
                       .filterDate(str(date_debut), str(date_fin))
                       .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))
                       .map(masquer))

                img = col.median()

                if indice == "RGB (couleurs naturelles)":
                    viz  = {"bands":["B4","B3","B2"],"min":0,"max":0.3,"gamma":1.4}
                    img2 = img
                elif indice == "NIR (fausses couleurs)":
                    viz  = {"bands":["B8","B4","B3"],"min":0,"max":0.4,"gamma":1.4}
                    img2 = img
                elif indice == "NDVI":
                    img2 = img.normalizedDifference(["B8","B4"])
                    viz  = {"min":-0.2,"max":0.8,
                            "palette":["d73027","f46d43","fdae61","ffffbf","a6d96a","1a9850"]}
                elif indice == "NDWI":
                    img2 = img.normalizedDifference(["B3","B8"])
                    viz  = {"min":-0.8,"max":0.2,
                            "palette":["d73027","f46d43","ffffbf","74add1","4575b4"]}
                elif indice == "MSI":
                    img2 = img.select("B11").divide(img.select("B8"))
                    viz  = {"min":0.4,"max":1.5,
                            "palette":["1a9850","a6d96a","ffffbf","f46d43","d73027"]}

                tile_url = img2.getMapId(viz)["tile_fetcher"].url_format
                folium.TileLayer(
                    tiles=tile_url,
                    attr="GEE",
                    name=f"Sentinel-2 — {indice}",
                    overlay=True
                ).add_to(m)

                folium.Circle(
                    location=[lat_centre, lon_centre],
                    radius=buffer_km*1000,
                    color="#97BC62",
                    fill=True, fill_opacity=0.1
                ).add_to(m)

                folium.LayerControl().add_to(m)
                st.success(f"✅ {indice} chargé !")

            except Exception as e:
                st.error(f"Erreur : {e}")

    st_folium(m, width=700, height=500)

with col_info:
    st.subheader("ℹ️ Informations")
    st.info(f"""
    **Zone analysée**
    - Lat : {lat_centre}°N
    - Lon : {lon_centre}°W
    - Rayon : {buffer_km} km

    **Période**
    - Du : {date_debut}
    - Au : {date_fin}

    **Indice :** {indice}
    **Modèle :** {modele_choisi}
    """)
    if gee_ok:
        st.success("🛰️ GEE connecté")
    else:
        st.error("❌ GEE non connecté")

# ── Météo
st.divider()
st.subheader("🌦️ Données Météo")

@st.cache_data(ttl=3600)
def get_meteo(lat, lon, debut, fin):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":lat, "longitude":lon,
        "start_date":str(debut), "end_date":str(fin),
        "daily":["temperature_2m_max","temperature_2m_min",
                 "precipitation_sum","et0_fao_evapotranspiration",
                 "windspeed_10m_max"],
        "timezone":"Africa/Casablanca"
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        d = r.json()["daily"]
        df = pd.DataFrame(d)
        df["time"] = pd.to_datetime(df["time"])
        df["deficit"] = df["precipitation_sum"] - df["et0_fao_evapotranspiration"]
        df["amplitude"] = df["temperature_2m_max"] - df["temperature_2m_min"]
        return df
    return None

if analyser:
    with st.spinner("⏳ Chargement météo..."):
        df_meteo = get_meteo(lat_centre, lon_centre, date_debut, date_fin)

    if df_meteo is not None:
        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Temp. max moy",   f"{df_meteo['temperature_2m_max'].mean():.1f}°C")
        c2.metric("Précip. totale",  f"{df_meteo['precipitation_sum'].sum():.0f} mm")
        c3.metric("ETo moy",         f"{df_meteo['et0_fao_evapotranspiration'].mean():.2f} mm/j")
        deficit_moy = df_meteo["deficit"].mean()
        c4.metric("Déficit moy",     f"{deficit_moy:.2f} mm/j",
                  delta="Stress ⚠️" if deficit_moy < -3 else "Normal ✅")

        # Graphiques
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_meteo["time"], y=df_meteo["temperature_2m_max"],
                name="Temp. max", line=dict(color="#E74C3C")))
            fig1.add_trace(go.Scatter(
                x=df_meteo["time"], y=df_meteo["temperature_2m_min"],
                name="Temp. min", line=dict(color="#3498DB")))
            fig1.add_hline(y=35, line_dash="dash",
                           line_color="orange",
                           annotation_text="Seuil 35°C")
            fig1.update_layout(
                title="Températures (°C)",
                plot_bgcolor="#0D1F35",
                paper_bgcolor="#0A1628",
                font_color="white")
            st.plotly_chart(fig1, use_container_width=True)

        with col_g2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_meteo["time"], y=df_meteo["precipitation_sum"],
                name="Précipitations", marker_color="#3498DB"))
            fig2.add_trace(go.Scatter(
                x=df_meteo["time"],
                y=df_meteo["et0_fao_evapotranspiration"],
                name="ETo", line=dict(color="#E74C3C")))
            fig2.update_layout(
                title="Précipitations vs ETo",
                plot_bgcolor="#0D1F35",
                paper_bgcolor="#0A1628",
                font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

        # Déficit
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df_meteo["time"], y=df_meteo["deficit"],
            marker_color=["#27AE60" if v>=0 else "#E74C3C"
                         for v in df_meteo["deficit"]],
            name="Déficit hydrique"))
        fig3.add_hline(y=0, line_color="white", line_width=1)
        fig3.update_layout(
            title="Déficit Hydrique (Précip - ETo)",
            plot_bgcolor="#0D1F35",
            paper_bgcolor="#0A1628",
            font_color="white")
        st.plotly_chart(fig3, use_container_width=True)

        # ── Score IA avec vrais modèles
        st.divider()
        st.subheader("🤖 Score IA — Prédiction du Stress Hydrique")

        # Calculer les features pour le modèle
        temp_max    = df_meteo["temperature_2m_max"].mean()
        precip      = df_meteo["precipitation_sum"].mean()
        eto         = df_meteo["et0_fao_evapotranspiration"].mean()
        amplitude   = df_meteo["amplitude"].mean()
        deficit_j   = df_meteo["deficit"].mean()

        # Cumul 30j
        deficit_cum = df_meteo["deficit"].sum()
        hsi         = (df_meteo["temperature_2m_max"] > 35).sum()

        # Valeurs approximées pour features satellite
        # (calculées depuis la période météo)
        ndvi_est = max(0.1, 0.6 - max(0, -deficit_j)*0.05)
        ndwi_est = -0.3 - max(0, -deficit_j)*0.03
        msi_est  = min(1.5, 0.7 + max(0, -deficit_j)*0.04)
        ndre_est = max(0.05, ndvi_est * 0.65)
        cri_est  = max(1.0, 8.0 - ndvi_est * 10)
        lai_est  = max(0.1, ndvi_est * 4.5)

        # SPI simplifié
        spi_30j = max(-2, min(2, deficit_cum / 50))

        # Vecteur features
        X_pred = np.array([[ndvi_est, ndwi_est, msi_est, ndre_est,
                            cri_est, lai_est, spi_30j, deficit_cum,
                            float(hsi), temp_max, eto, amplitude]])

        X_pred_sc = scaler.transform(X_pred)

        # Prédictions
        pred_rf  = rf.predict(X_pred_sc)[0]
        prob_rf  = rf.predict_proba(X_pred_sc)[0]
        pred_xgb = xgb.predict(X_pred_sc)[0]
        prob_xgb = xgb.predict_proba(X_pred_sc)[0]

        # Affichage selon choix modèle
        if modele_choisi == "Random Forest":
            predictions = [("Random Forest", pred_rf, prob_rf)]
        elif modele_choisi == "XGBoost":
            predictions = [("XGBoost", pred_xgb, prob_xgb)]
        else:
            predictions = [
                ("Random Forest", pred_rf,  prob_rf),
                ("XGBoost",       pred_xgb, prob_xgb)
            ]

        for nom_modele, pred, prob in predictions:
            st.markdown(f"### 📊 {nom_modele}")
            col_pred1, col_pred2, col_pred3 = st.columns(3)

            with col_pred1:
                score = int(prob[min(pred, 3)] * 100)
                st.metric("Confiance", f"{score}%")
                st.progress(score/100)

            with col_pred2:
                label_nom = NOMS_LABELS[pred]
                st.markdown("**Niveau de stress :**")
                st.markdown(f"## {label_nom}")

            with col_pred3:
                # Probabilités par classe
                fig_prob = go.Figure(go.Bar(
                    x=[NOMS_LABELS[i] for i in range(4)],
                    y=[p*100 for p in prob],
                    marker_color=[COULEURS_LABELS[i] for i in range(4)]
                ))
                fig_prob.update_layout(
                    title="Probabilités par classe",
                    plot_bgcolor="#0D1F35",
                    paper_bgcolor="#0A1628",
                    font_color="white",
                    height=250,
                    margin=dict(t=30,b=0,l=0,r=0)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            # Recommandation
            if pred == 0:
                st.success("✅ Aucune irrigation nécessaire. Prochaine vérification dans 15 jours.")
            elif pred == 1:
                eau = abs(deficit_j) * 0.7 * 7
                st.warning(f"⚠️ Irrigation préventive recommandée dans les 7 jours.\n💧 Quantité estimée : {eau:.1f} mm")
            elif pred == 2:
                eau = abs(deficit_j) * 0.8 * 3
                st.warning(f"🟠 Irrigation recommandée dans les 48h.\n💧 Quantité estimée : {eau:.1f} mm")
            else:
                eau = abs(deficit_j) * 1.0
                st.error(f"🔴 IRRIGATION URGENTE — Intervenir dans les 24h !\n💧 Quantité estimée : {eau:.1f} mm/jour")

            st.divider()

else:
    st.info("👆 Définissez une zone et cliquez sur **Analyser** pour obtenir les prédictions.")

# ── Footer
st.markdown("*PFE Agriculture 4.0 — Badaraa Liou Guindo — Avril 2026*")
