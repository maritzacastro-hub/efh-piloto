import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.express as px

st.title("Dashboard EFH – Piloto (Métricas por fold)")

@st.cache_data
def load_metrics():
    return pd.read_csv("public_results/metrics_by_fold.csv")

df = load_metrics()

# (opcional) filtra solo los 2 modelos
df = df[df["Modelo"].str.contains("LR ridge|MLP seleccionadas", case=False, regex=True)].copy()
df["Fold"] = df["Fold"].astype(int)

def plot_fold(metric, title):
    piv = df.pivot(index="Fold", columns="Modelo", values=metric).reset_index()
    long = piv.melt(id_vars="Fold", var_name="Modelo", value_name=metric)
    fig = px.line(long, x="Fold", y=metric, color="Modelo", markers=True, title=title)
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    plot_fold("Precision", "Precisión por fold")
    plot_fold("Sensibilidad", "Sensibilidad por fold")
with col2:
    plot_fold("Especificidad", "Especificidad por fold")
    plot_fold("AUC", "AUC por fold")

# conclusión simple
mean_auc = df.groupby("Modelo")["AUC"].mean()
best_model = mean_auc.idxmax()
st.success(f"Modelo recomendado por AUC promedio: **{best_model}**")