import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "modelo")

modelo   = None
scaler   = None
columnas = None

def cargar_modelo():
    global modelo, scaler, columnas
    try:
        modelo   = joblib.load(os.path.join(MODEL_DIR, "modelo_obesidad.pkl"))
        scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        columnas = joblib.load(os.path.join(MODEL_DIR, "columnas_modelo.pkl"))
    except FileNotFoundError:
        pass  # los .pkl se generan desde el notebook


CLASES_ES = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight":        "Peso normal",
    "Overweight_Level_I":   "Sobrepeso I",
    "Overweight_Level_II":  "Sobrepeso II",
    "Obesity_Type_I":       "Obesidad tipo I",
    "Obesity_Type_II":      "Obesidad tipo II",
    "Obesity_Type_III":     "Obesidad tipo III",
}

COLORES_CLASE = {
    "Insufficient_Weight": "#3b82f6",
    "Normal_Weight":        "#10b981",
    "Overweight_Level_I":   "#f59e0b",
    "Overweight_Level_II":  "#f97316",
    "Obesity_Type_I":       "#ef4444",
    "Obesity_Type_II":      "#dc2626",
    "Obesity_Type_III":     "#991b1b",
}


def preprocesar(form):
    """Replica exacta del pipeline del notebook."""
    # --- 1. Recolectar datos crudos ---
    datos = {
        "Gender":                         form["Gender"],
        "Age":                            float(form["Age"]),
        "family_history_with_overweight": form["family_history_with_overweight"],
        "FAVC":   form["FAVC"],
        "FCVC":   float(form["FCVC"]),
        "NCP":    float(form["NCP"]),
        "CAEC":   form["CAEC"],
        "SMOKE":  form["SMOKE"],
        "CH2O":   float(form["CH2O"]),
        "SCC":    form["SCC"],
        "FAF":    float(form["FAF"]),
        "TUE":    float(form["TUE"]),
        "CALC":   form["CALC"],
        "MTRANS": form["MTRANS"],
    }
    df = pd.DataFrame([datos])

    # --- 2. Redondear variables ordinales (Cell 22 del notebook) ---
    for col in ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        df[col] = df[col].round(0).astype(int)

    # --- 3. Codificación binaria (Cell 23 del notebook) ---
    # Gender: Female=0, Male=1
    df["Gender"] = 1 if df["Gender"].iloc[0] == "Male" else 0
    # Columnas yes/no: no=0, yes=1
    for col in ["FAVC", "SMOKE", "SCC", "family_history_with_overweight"]:
        df[col] = 1 if df[col].iloc[0] == "yes" else 0

    # --- 4. One-hot encoding de categóricas (Cell 23) ---
    df = pd.get_dummies(df, columns=["MTRANS", "CAEC", "CALC"],
                        prefix=["MTRANS", "CAEC", "CALC"], dtype=int)

    # Eliminar categorías de referencia
    for ref in ["CAEC_no", "CALC_no", "MTRANS_Public_Transportation"]:
        if ref in df.columns:
            df.drop(columns=[ref], inplace=True)

    # --- 5. Términos de interacción (Cell 24) ---
    faf = df["FAF"].iloc[0]
    gen = df["Gender"].iloc[0]
    df["FAF_Female"] = (1 - gen) * faf
    df["FAF_Male"]   = gen * faf

    # --- 6. Separar columnas OHE vs numéricas ---
    cols_onehot = [c for c in df.columns
                   if c.startswith(("MTRANS_", "CAEC_", "CALC_"))]
    # sorted() replica el comportamiento de pandas Index.difference() que usó el notebook
    cols_num    = sorted([c for c in df.columns if c not in set(cols_onehot)])

    # --- 7. Escalar solo columnas numéricas (Cell 26) ---
    df_num_scaled = pd.DataFrame(
        scaler.transform(df[cols_num]),
        columns=cols_num,
        index=df.index,
    )
    df_prep = pd.concat([df_num_scaled, df[cols_onehot]], axis=1)

    # --- 8. Alinear con las columnas del modelo ---
    df_prep = df_prep.reindex(columns=columnas, fill_value=0)
    return df_prep.values


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    if request.method == "GET":
        return render_template("predictor.html")

    if modelo is None:
        return render_template("predictor.html",
                               error="El modelo no está cargado. Ejecutá el notebook y copiá los .pkl a app/modelo/.")

    try:
        X = preprocesar(request.form)
    except (KeyError, ValueError) as e:
        return render_template("predictor.html", error=f"Error en los datos: {e}")

    pred_clase = modelo.predict(X)[0]
    probas     = modelo.predict_proba(X)[0]
    clases     = modelo.classes_

    confianza = round(float(probas.max()) * 100, 1)

    proba_data = sorted([
        {
            "clase_es": CLASES_ES.get(c, c),
            "prob":     round(float(p) * 100, 1),
            "color":    COLORES_CLASE.get(c, "#6366f1"),
        }
        for c, p in zip(clases, probas)
    ], key=lambda x: x["prob"], reverse=True)

    return render_template(
        "resultado.html",
        prediccion=CLASES_ES.get(pred_clase, pred_clase),
        confianza=confianza,
        color=COLORES_CLASE.get(pred_clase, "#6366f1"),
        proba_data=proba_data,
    )


@app.route("/nosotros")
def nosotros():
    return render_template("nosotros.html")


@app.route("/conclusion")
def conclusion():
    return render_template("conclusion.html")


@app.route("/analisis")
def analisis():
    img_dir  = os.path.join(BASE_DIR, "static", "img")
    imagenes = {
        "importancias":    os.path.isfile(os.path.join(img_dir, "importancias.png")),
        "comparativa":     os.path.isfile(os.path.join(img_dir, "comparativa.png")),
        "matriz_confusion": os.path.isfile(os.path.join(img_dir, "matriz_confusion.png")),
        "heatmap":         os.path.isfile(os.path.join(img_dir, "heatmap.png")),
    }
    return render_template("analisis.html", imagenes=imagenes)


if __name__ == "__main__":
    cargar_modelo()
    app.run(debug=True)
