# ================================================================
#   DASHBOARD EFH 2021 – Público (MISMO ESTILO)
#   Fuente: public_results/ (sin EFH cruda)
#   Cambio solicitado: quitar p-valor y Chi²
# ================================================================

import streamlit as st
st.set_page_config(layout="wide")  # debe ir primero

import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------------
#   PALETA WOW
# ---------------------------------------------------------------
PRIMARY = "#4A90E2"
PURPLE = "#9B59B6"
TEAL = "#1ABC9C"
YELLOW = "#F1C40F"
DARK = "#2C3E50"
LIGHT = "#ECF0F1"

# ---------------------------------------------------------------
#   CSS WOW
# ---------------------------------------------------------------
st.markdown(f"""
    <style>

        .main {{
            background-color: {LIGHT};
        }}

        h1 {{
            text-align: center;
            background: linear-gradient(90deg, {PRIMARY}, {PURPLE});
            color: white !important;
            padding: 18px;
            border-radius: 12px;
            font-size: 40px !important;
            margin-bottom: 25px !important;
        }}

        h2 {{
            color: {DARK} !important;
            border-left: 10px solid {PRIMARY};
            padding-left: 12px;
            font-weight: 900 !important;
            margin-top: 15px !important;
            margin-bottom: 18px !important;
        }}

        .metric h3 {{
        font-size: 25px !important;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: #1B2B65 !important;
            padding: 20px;
            border-radius: 10px;
        }}

        .metric {{
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 2px solid {PRIMARY};
            box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
#   CARGA DE RESULTADOS (public_results)
# ---------------------------------------------------------------
@st.cache_data
def load_public_eda():
    summary = pd.read_csv("public_results/eda_summary.csv")
    meta = pd.read_csv("public_results/eda_meta.csv")
    num_stats = pd.read_csv("public_results/eda_num_stats.csv")
    num_hist = pd.read_csv("public_results/eda_num_hist.csv.gz", compression="gzip")
    cat_counts = pd.read_csv("public_results/eda_cat_counts.csv.gz", compression="gzip")
    biv_num = pd.read_csv("public_results/eda_biv_num.csv")
    biv_cat = pd.read_csv("public_results/eda_biv_cat.csv.gz", compression="gzip")
    corr_long = pd.read_csv("public_results/eda_corr.csv.gz", compression="gzip")
    return summary, meta, num_stats, num_hist, cat_counts, biv_num, biv_cat, corr_long

@st.cache_data
def load_metrics_by_fold(path="public_results/metrics_by_fold.csv"):
    return pd.read_csv(path)

try:
    summary, meta, num_stats, num_hist, cat_counts, biv_num, biv_cat, corr_long = load_public_eda()
except Exception as e:
    st.error("No pude cargar EDA desde public_results/. Revisa que existan los archivos eda_*.csv.")
    st.caption(str(e))
    st.stop()

try:
    df_metrics_fold = load_metrics_by_fold()
    have_fold_metrics = True
except Exception as e:
    have_fold_metrics = False
    st.warning("No pude cargar public_results/metrics_by_fold.csv (métricas por fold).")
    st.caption(str(e))

# Usamos All (sin filtro macrozona, igual que tu dashboard)
MACRO = "All"

# ---------------------------------------------------------------
#   DEFINICIÓN DE VARIABLES (las 18)
# ---------------------------------------------------------------
vars_rep = [
    # Numéricas
    'act_fijo', 'act_var', 'cap_pen_ent', 'edad_pr', 'hr_trabajadas_pr', 'yoprinm_pr', 'ypenh', 'ysubh',
    # Binarias
    't_cc', 't_tbco', 'u_cheq', 'u_pac', 'u_pat', 'u_tbco', 'u_tprepago',
    # Categóricas
    'est_civil_pr', 'numh', 'ocuph',
]

available_vars = set(meta["var"].unique())
vars_rep = [v for v in vars_rep if v in available_vars]

num_vars = meta.loc[meta["type"] == "numeric", "var"].tolist()
cat_vars = meta.loc[meta["type"] != "numeric", "var"].tolist()

# ---------------------------------------------------------------
#   SIDEBAR DE VARIABLES (igual)
# ---------------------------------------------------------------
st.sidebar.title("Variables 🔍")

search = st.sidebar.text_input("Buscar:", placeholder="Escriba para filtrar…")
filtered = [v for v in vars_rep if search.lower() in v.lower()] if search else vars_rep

if "var_selected" not in st.session_state:
    st.session_state["var_selected"] = filtered[0]

for v in filtered:
    if st.sidebar.button(v, key=f"btn_{v}"):
        st.session_state["var_selected"] = v

var = st.session_state["var_selected"]

# ---------------------------------------------------------------
#   TÍTULO PRINCIPAL
# ---------------------------------------------------------------
st.markdown("<h1>Dashboard Encuesta Financiera de Hogares 2021</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------
#   SECCIÓN 1 — MÉTRICAS (igual, pero desde eda_summary)
# ---------------------------------------------------------------
st.header("📌 1. Resumen General")

n_reg = int(summary.loc[0, "n_registros"])
pct_vp = float(summary.loc[0, "pct_viv_propia"])

TOTAL_VARS = 18
NUMERICAS = 8
CATEGORICAS = 3
BINARIAS = 7

row1 = st.columns(6)
row1[0].markdown(f"<div class='metric'><h3>Registros</h3><h2>{n_reg:,}</h2></div>", unsafe_allow_html=True)
row1[1].markdown(f"<div class='metric'><h3>Vivienda Propia (%)</h3><h2>{pct_vp:.2f}%</h2></div>", unsafe_allow_html=True)
row1[2].markdown(f"<div class='metric'><h3>Total de Variables</h3><h2>{TOTAL_VARS}</h2></div>", unsafe_allow_html=True)
row1[3].markdown(f"<div class='metric'><h3>Variables Numéricas</h3><h2>{NUMERICAS}</h2></div>", unsafe_allow_html=True)
row1[4].markdown(f"<div class='metric'><h3>Variables Categóricas</h3><h2>{CATEGORICAS}</h2></div>", unsafe_allow_html=True)
row1[5].markdown(f"<div class='metric'><h3>Variables Binarias</h3><h2>{BINARIAS}</h2></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
#   helpers: reconstruir "df" sintético para gráficas sin EFH cruda
# ---------------------------------------------------------------
@st.cache_data
def pseudo_series_from_hist(var_name: str) -> pd.Series:
    h = num_hist[(num_hist["macrozona"] == MACRO) & (num_hist["var"] == var_name)].copy()
    if h.empty:
        return pd.Series([], dtype=float)
    h["bin_mid"] = (h["bin_left"] + h["bin_right"]) / 2.0
    vals = np.repeat(h["bin_mid"].to_numpy(), h["count"].to_numpy().astype(int))
    return pd.Series(vals, name=var_name)

@st.cache_data
def pseudo_biv_numeric(var_name: str, n_each: int = 2000) -> pd.DataFrame:
    # genera datos sintéticos por grupo usando cuantiles Q25/Q50/Q75
    row = biv_num[(biv_num["macrozona"] == MACRO) & (biv_num["var"] == var_name)]
    if row.empty:
        return pd.DataFrame({"viv_propia": [], var_name: []})
    row = row.iloc[0]

    def gen(q25, q50, q75, n, seed):
        rng = np.random.default_rng(seed)
        if any(pd.isna([q25, q50, q75])) or (q25 == q75):
            return np.full(n, q50 if not pd.isna(q50) else 0.0)
        u = rng.random(n)
        x = np.empty(n)
        left = u < 0.5
        x[left] = q25 + (q50 - q25) * (u[left] / 0.5)
        x[~left] = q50 + (q75 - q50) * ((u[~left] - 0.5) / 0.5)
        # muchos montos son >=0
        return np.clip(x, 0, None)

    seed0 = abs(hash(var_name)) % (2**32)
    x0 = gen(row["q25_0"], row["q50_0"], row["q75_0"], n_each, seed0 + 0)
    x1 = gen(row["q25_1"], row["q50_1"], row["q75_1"], n_each, seed0 + 1)

    dfb = pd.DataFrame({
        "viv_propia": np.r_[np.zeros(n_each, dtype=int), np.ones(n_each, dtype=int)],
        var_name: np.r_[x0, x1]
    })
    return dfb

# ---------------------------------------------------------------
#   SECCIÓN 2 — UNIVARIADO (igual en estructura)
# ---------------------------------------------------------------
st.header("📊 2. Análisis Univariadas")

VAR_LABELS = {
    "yoprinm_pr": "ingreso mensual de la persona de referencia del hogar",
    "act_fijo": "monto total de activos financieros de renta fija",
    "act_var": "monto total que el hogar tiene invertido en los diferentes instrumentos de renta variable.",
    "cap_pen_ent": "saldo en cuenta de capitalización individual ",
    "edad_pr": "edad de la persona de referencia",
    "hr_trabajadas_pr": "horas trabajadas por la persona de referencia",
    "ypenh": "ingreso mensual del hogar por pensiones",
    "ysubh": "ingreso mensual del hogar por subsidios",
    "numh": "número de miembros en el hogar",
    "ocuph": "número de miembros del hogar que se encuentra trabajando",
    "u_pac": "si algún miembro del hogar utiliza pago automático a las cuentas corrientes",
    "t_tbco": "si algún miembro del hogar posee tarjetas de crédito bancarias",
    "u_pat": "si algún miembro del hogar utiliza pago automático a tarjetas de crédito",
    "u_tbco": "si algún miembro del hogar utiliza como medio de pago las tarjetas de crédito bancarias",
    "u_cheq": "si algún miembro del hogar utiliza como medio de pago los cheques",
    "t_cc": "si algún miembro del hogar posee cuenta corriente",
    "u_prepago": "si algún miembro del hogar utiliza como medio de pago los instrumentos de prepago",
    "est_civil_pr": "estado civil de la persona de referencia",
    
}

def label_var(v: str) -> str:
    return VAR_LABELS.get(v, v)  # si no está, muestra el nombre

if var in num_vars and var not in ["numh", "ocuph"]:

    st.markdown("### Distribución")
    colD, colE = st.columns([2,2])

    srow = num_stats[(num_stats["macrozona"] == MACRO) & (num_stats["var"] == var)].iloc[0]
    total = int(srow["n"])
    cero = int(srow["cero"])
    pos = int(srow["pos"])
    porc_cero = float(srow["porc_cero"])
    porc_pos = float(srow["porc_pos"])

    with colD:
        st.subheader(" ")
        sig = label_var(var)
        cat0 = f"Hogares sin monto en {sig}"
        cat1 = f"Hogares con monto en {sig}"
        donut_df = pd.DataFrame({"categoria": [cat0, cat1], "valor": [cero, pos]})

        fig_donut = px.pie(
            donut_df, names="categoria", values="valor", hole=0.55,
            color="categoria",
            color_discrete_map={cat0: PRIMARY, cat1: TEAL}
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent", textfont_size=22)
        st.plotly_chart(fig_donut, use_container_width=True)

    with colE:
        st.subheader("🧠 Interpretación")
        st.markdown(f"""
        **Distribución general de `{sig}`:**
        - **{porc_cero:.2f}%** de los hogares no presentan **{sig}**.
        - **{porc_pos:.2f}%** tienen **{sig}**.

        """)
        

    st.markdown("---")
    colA, colB, colC = st.columns([2, 2, 1])

    serie = pseudo_series_from_hist(var)

    with colA:
        st.subheader("Histograma")
        fig = px.histogram(serie.to_frame(), x=var, nbins=30, color_discrete_sequence=[PRIMARY])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("BoxPlot")
        fig = px.box(serie.to_frame(), y=var, color_discrete_sequence=[TEAL])
        st.plotly_chart(fig, use_container_width=True)

    with colC:
        st.subheader("Resumen")
        st.write(serie.describe(percentiles=[0.25, 0.5, 0.75]))

else:
    colA, colC = st.columns([3, 1.5])

    raw = cat_counts[(cat_counts["macrozona"] == MACRO) & (cat_counts["var"] == var)].copy()

    # ✅ 1) Normalizar categorías numéricas: "2.0" -> "2"
    raw["category"] = raw["category"].astype(str).str.strip()
    raw["category"] = raw["category"].str.replace(r"\.0$", "", regex=True)

    # ✅ 2) Agrupar (por si venían duplicadas)
    raw = raw.groupby("category", as_index=False)["n"].sum()

    # ✅ 3) (Opcional recomendado) ordenar numéricamente si aplica
    raw["category_num"] = pd.to_numeric(raw["category"], errors="coerce")
    raw = raw.sort_values(["category_num", "n"], ascending=[True, False]).drop(columns=["category_num"])

    # ✅ 4) porcentaje
    raw["porcentaje"] = (raw["n"] / raw["n"].sum() * 100).round(2).astype(str) + "%"

    # ✅ tabla final (solo 3 columnas)
    table = raw[["category", "n", "porcentaje"]].copy()
    table.columns = [var, "n", "porcentaje"]

    with colA:
        st.subheader("Categorías")
        fig = px.bar(table.head(25), x=var, y="n", text="porcentaje", color=var)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with colC:
        st.subheader("Frecuencias")
        st.dataframe(table.head(25), use_container_width=True)


# ---------------------------------------------------------------
#   SECCIÓN 3 — CORRELACIONES (desde eda_corr)
# ---------------------------------------------------------------
st.header("📈 3. Matriz de Correlación")

corr_mat = corr_long.pivot(index="var1", columns="var2", values="corr")
fig = px.imshow(corr_mat, text_auto=True, color_continuous_scale="Blues")
fig.update_layout(width=900, height=900)
st.plotly_chart(fig)

# ---------------------------------------------------------------
#   MODELOS (igual al piloto que ya te funcionó)
# ---------------------------------------------------------------
st.subheader("📌 Métricas por fold (LR Ridge vs MLP seleccionadas)")

if have_fold_metrics:
    metric_cols = ["Precision", "Sensibilidad", "Especificidad", "AUC"]
    metric_cols = [m for m in metric_cols if m in df_metrics_fold.columns]

    if len(metric_cols) == 0:
        st.error("No encuentro columnas de métricas en metrics_by_fold.csv.")
    else:
        dfp = df_metrics_fold[df_metrics_fold["Modelo"].str.contains("LR Ridge|MLP seleccionadas", case=False, regex=True)].copy()
        dfp["Fold"] = dfp["Fold"].astype(int)

        def plot_fold(metric_name, title):
            piv = dfp.pivot(index="Fold", columns="Modelo", values=metric_name).reset_index()
            long = piv.melt(id_vars="Fold", var_name="Modelo", value_name=metric_name)
            fig = px.line(long, x="Fold", y=metric_name, color="Modelo", markers=True, title=title)
            fig.update_layout(xaxis_title="Fold", yaxis_title=metric_name)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Precision" in metric_cols:
                plot_fold("Precision", "Distribución por fold - Precisión")
            if "Sensibilidad" in metric_cols:
                plot_fold("Sensibilidad", "Distribución por fold - Sensibilidad")

        with col2:
            if "Especificidad" in metric_cols:
                plot_fold("Especificidad", "Distribución por fold - Especificidad")
            if "AUC" in metric_cols:
                plot_fold("AUC", "Distribución por fold - AUC")

    st.subheader("✅ Conclusión automática (criterio: promedio por fold)")
    metrics_use = ["AUC", "Sensibilidad", "Especificidad", "Precision"]
    metrics_use = [m for m in metrics_use if m in df_metrics_fold.columns]

    summary_models = df_metrics_fold.groupby("Modelo")[metrics_use].agg(["mean", "std"])
    st.dataframe(summary_models, use_container_width=True)

    best_model = df_metrics_fold.groupby("Modelo")["AUC"].mean().idxmax()
    st.success(f"📌 Según el **AUC promedio (OOF)**, el modelo recomendado es: **{best_model}**.")
else:
    st.info("No hay métricas por fold cargadas. (Falta public_results/metrics_by_fold.csv)")
