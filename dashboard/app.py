import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Configuration page
st.set_page_config(
    page_title="Stress Hydrique — Sentinel-2",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style CSS
st.markdown("""
<style>
    .main { background-color: #0A1628; }
    .stMetric { background-color: #0D1F35; border-radius: 8px; padding: 10px; }
    h1, h2, h3 { color: #97BC62; }
</style>
""", unsafe_allow_html=True)

# ── Titre
st.title("🌿 Détection du Stress Hydrique")
st.markdown("**Plaine du Gharb, Maroc — Sentinel-2 & Données Météo (2020–2024)**")
st.divider()

# ── Données simulées (sans Drive)
@st.cache_data
def charger_donnees():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    n = len(dates)

    # Cycle saisonnier réaliste
    jour = np.arange(n)
    ndvi = 0.4 + 0.3*np.sin(2*np.pi*jour/365 - np.pi/2) + np.random.normal(0,0.05,n)
    ndvi = ndvi.clip(0.05, 0.85)

    temp = 27 + 13*np.sin(2*np.pi*jour/365) + np.random.normal(0,3,n)
    precip = np.where(np.sin(2*np.pi*jour/365) > 0.3,
                      np.random.exponential(3, n), 0)

    df = pd.DataFrame({
        "date": dates,
        "NDVI": ndvi.round(3),
        "MSI":  (1.2 - ndvi*0.5 + np.random.normal(0,0.05,n)).clip(0.4,1.5).round(3),
        "temp_max": temp.round(1),
        "precipitation": precip.round(1),
        "ETo": (2 + 4*np.sin(2*np.pi*jour/365) + np.random.normal(0,0.5,n)).clip(0.5,12).round(2),
    })
    df["annee"] = df["date"].dt.year
    df["mois"]  = df["date"].dt.month

    # Labels
    def label(row):
        if row["NDVI"] > 0.45 and row["temp_max"] < 30: return 0
        elif row["NDVI"] > 0.25: return 1
        elif row["NDVI"] > 0.15: return 2
        else: return 3
    df["label_stress"] = df.apply(label, axis=1)
    df["label_nom"] = df["label_stress"].map({
        0:"Pas de stress", 1:"Risque",
        2:"Stress modéré", 3:"Stress sévère"
    })
    return df

df = charger_donnees()

# ── Sidebar
st.sidebar.title("⚙️ Filtres")
annee_sel = st.sidebar.selectbox("Année", [2020,2021,2022,2023,2024], index=4)
df_an = df[df["annee"] == annee_sel]

# ── KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("NDVI moyen", f"{df_an['NDVI'].mean():.3f}",
              delta=f"{df_an['NDVI'].mean()-0.405:.3f} vs moy globale")
with col2:
    st.metric("Temp. max moy", f"{df_an['temp_max'].mean():.1f}°C")
with col3:
    jours_stress = (df_an["label_stress"] >= 2).sum()
    st.metric("Jours de stress", f"{jours_stress}",
              delta=f"{jours_stress/len(df_an)*100:.0f}% de l'année")
with col4:
    st.metric("Précip. totale", f"{df_an['precipitation'].sum():.0f} mm")

st.divider()

# ── Graphiques
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("📈 NDVI — Evolution temporelle")
    couleurs_label = {
        "Pas de stress":"#27AE60",
        "Risque":"#F39C12",
        "Stress modéré":"#E67E22",
        "Stress sévère":"#C0392B"
    }
    fig1 = px.scatter(df_an, x="date", y="NDVI",
                      color="label_nom",
                      color_discrete_map=couleurs_label,
                      title=f"NDVI {annee_sel}")
    fig1.add_hline(y=0.30, line_dash="dash",
                   line_color="orange", annotation_text="Seuil stress (0.30)")
    fig1.add_hline(y=0.15, line_dash="dash",
                   line_color="red", annotation_text="Stress sévère (0.15)")
    fig1.update_layout(
        plot_bgcolor="#0D1F35",
        paper_bgcolor="#0A1628",
        font_color="white",
        legend_title="Niveau de stress"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_g2:
    st.subheader("🌡️ Température & ETo")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_an["date"], y=df_an["temp_max"],
                              name="Temp. max (°C)", line=dict(color="#E74C3C")))
    fig2.add_trace(go.Scatter(x=df_an["date"], y=df_an["ETo"],
                              name="ETo (mm/j)", line=dict(color="#F39C12")))
    fig2.add_hline(y=35, line_dash="dash", line_color="orange",
                   annotation_text="Seuil chaleur 35°C")
    fig2.update_layout(
        plot_bgcolor="#0D1F35",
        paper_bgcolor="#0A1628",
        font_color="white",
        title=f"Température & ETo {annee_sel}"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Distribution labels
st.subheader("📊 Distribution des niveaux de stress")
col_d1, col_d2 = st.columns(2)

with col_d1:
    dist = df_an["label_nom"].value_counts()
    fig3 = px.bar(x=dist.index, y=dist.values,
                  color=dist.index,
                  color_discrete_map=couleurs_label,
                  title=f"Jours par niveau — {annee_sel}",
                  labels={"x":"Niveau","y":"Nombre de jours"})
    fig3.update_layout(
        plot_bgcolor="#0D1F35",
        paper_bgcolor="#0A1628",
        font_color="white",
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_d2:
    fig4 = px.pie(values=dist.values, names=dist.index,
                  color=dist.index,
                  color_discrete_map=couleurs_label,
                  title=f"Répartition — {annee_sel}")
    fig4.update_layout(
        paper_bgcolor="#0A1628",
        font_color="white"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Précipitations
st.subheader("🌧️ Précipitations journalières")
fig5 = px.bar(df_an, x="date", y="precipitation",
              title=f"Précipitations {annee_sel}",
              color_discrete_sequence=["#3498DB"])
fig5.update_layout(
    plot_bgcolor="#0D1F35",
    paper_bgcolor="#0A1628",
    font_color="white"
)
st.plotly_chart(fig5, use_container_width=True)

# ── Footer
st.divider()
st.markdown("*PFE Agriculture 4.0 — Détection stress hydrique Sentinel-2 | Avril 2026*")
