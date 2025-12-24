from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

from messaging.consumer import JsonKafkaConsumer
from messaging.producer import JsonKafkaProducer, send_with_retry
from model.churn_model import (
    TARGET_COL,
    build_pipeline,
    build_preprocessor,
    create_lr_from_weights,
    evaluate_binary_classifier,
    extract_lr_weights_from_pipeline,
    load_telco_csv,
)

TOPIC_CLIENT_DATA = "client_data"
TOPIC_GLOBAL_MODEL = "global_model"
TOPIC_PREDICTIONS = "predictions"


@st.cache_data(show_spinner=False)
def load_df(csv_path: str) -> pd.DataFrame:
    return load_telco_csv(csv_path)


@st.cache_resource(show_spinner=False)
def build_schema_pipeline(df_full: pd.DataFrame) -> Pipeline:
    X = df_full.drop(columns=[TARGET_COL])
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in numeric_cols]
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    pipe = build_pipeline(preprocessor)
    pipe.named_steps["preprocess"].fit(X, df_full[TARGET_COL].to_numpy())
    return pipe


def local_train_for_demo(df_full: pd.DataFrame, which: str) -> Pipeline:
    """
    Train a *local* model (client_1 or client_2) only for comparison/visualization in Streamlit.
    """
    n = len(df_full)
    mid = n // 2
    if which == "client_1":
        df_part = df_full.iloc[:mid].copy()
    else:
        df_part = df_full.iloc[mid:].copy()

    pipe = build_schema_pipeline(df_full)
    X_local = df_part.drop(columns=[TARGET_COL])
    y_local = df_part[TARGET_COL].to_numpy()
    X_local_t = pipe.named_steps["preprocess"].transform(X_local)
    pipe.named_steps["clf"].fit(X_local_t, y_local)
    return pipe


def pipeline_from_global_weights(schema_pipe: Pipeline, coef: np.ndarray, intercept: np.ndarray) -> Pipeline:
    """
    Clone-ish: keep preprocess fitted, set classifier weights.
    """
    new_pipe = Pipeline(steps=[("preprocess", schema_pipe.named_steps["preprocess"]), ("clf", create_lr_from_weights(coef, intercept))])
    return new_pipe


def evaluate_pipe(pipe: Pipeline, df_full: pd.DataFrame) -> Dict[str, Any]:
    X = df_full.drop(columns=[TARGET_COL])
    y = df_full[TARGET_COL].to_numpy()
    X_t = pipe.named_steps["preprocess"].transform(X)
    probs = pipe.named_steps["clf"].predict_proba(X_t)[:, 1]
    m = evaluate_binary_classifier(y_true=y, y_prob=probs)
    return {"accuracy": m["accuracy"], "roc_auc": m["roc_auc"], "fpr": m["fpr"], "tpr": m["tpr"]}


def consume_latest_global_model(kafka_bootstrap: str, timeout_s: float = 3.0) -> Optional[Dict[str, Any]]:
    """
    Reads recent messages from topic global_model and returns the latest one seen within timeout_s.
    Uses auto_offset_reset='latest' to avoid replaying the full log.
    """
    consumer = JsonKafkaConsumer(
        topics=[TOPIC_GLOBAL_MODEL],
        bootstrap_servers=kafka_bootstrap,
        group_id=f"streamlit-global-{uuid.uuid4().hex[:8]}",
        client_id=f"streamlit-global-consumer-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=int(timeout_s * 1000),
    )

    latest = None
    for msg in consumer:
        val = msg["value"]
        if val and val.get("type") == "global_model":
            latest = val

    consumer.close()
    return latest


def request_prediction_and_wait(
    producer: JsonKafkaProducer,
    kafka_bootstrap: str,
    payload: Dict[str, Any],
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    request_id = uuid.uuid4().hex
    req = {
        "type": "predict_request",
        "request_id": request_id,
        "source": "streamlit",
        "payload": payload,
        "ts": time.time(),
    }
    send_with_retry(producer, TOPIC_CLIENT_DATA, req, key=request_id)

    consumer = JsonKafkaConsumer(
        topics=[TOPIC_PREDICTIONS],
        bootstrap_servers=kafka_bootstrap,
        group_id=f"streamlit-preds-{uuid.uuid4().hex[:8]}",
        client_id=f"streamlit-preds-consumer-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=int(timeout_s * 1000),
    )

    result = {"status": "timeout", "request_id": request_id}
    for msg in consumer:
        val = msg["value"]
        if not val:
            continue
        if val.get("type") == "prediction_result" and val.get("request_id") == request_id:
            result = val
            break

    consumer.close()
    return result


def main() -> None:
    st.set_page_config(page_title="Federated Churn (Kafka + Streamlit)", layout="wide")

    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    csv_path = os.getenv("CSV_PATH", "./data/telco_churn.csv")

    st.title("Federated Machine Learning — Customer Churn (Kafka + Streamlit)")

    df = load_df(csv_path)
    schema_pipe = build_schema_pipeline(df)

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Choix du modèle", ["Modèle fédéré (global)", "Modèle local (client_1)", "Modèle local (client_2)"])
        st.caption("Le modèle fédéré est consommé depuis Kafka (topic `global_model`).")
        st.caption("Les modèles locaux sont entraînés dans Streamlit uniquement pour comparaison.")

    tab1, tab2, tab3 = st.tabs(["Dataset", "Prédiction", "Évaluation & ROC"])

    # --- Tab 1: Dataset
    with tab1:
        st.subheader("Chargement & visualisation")
        st.write(f"Dimensions: **{df.shape[0]} lignes** × **{df.shape[1]} colonnes**")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### Distribution de la cible (Churn)")
        vc = df[TARGET_COL].value_counts().rename(index={0: "No", 1: "Yes"})
        st.bar_chart(vc)

    # Prepare model
    global_msg = None
    global_pipe = None

    if model_choice.startswith("Modèle fédéré"):
        with st.spinner("Lecture du modèle global depuis Kafka..."):
            global_msg = consume_latest_global_model(kafka_bootstrap=kafka_bootstrap, timeout_s=3.0)

        if not global_msg:
            st.warning("Aucun modèle global trouvé dans Kafka. Lance d'abord `docker compose up` et attends la fin de l'entraînement fédéré.")
        else:
            coef = np.array(global_msg["coef"], dtype=float)
            intercept = np.array(global_msg["intercept"], dtype=float)
            global_pipe = pipeline_from_global_weights(schema_pipe, coef, intercept)

    elif model_choice.endswith("client_1"):
        with st.spinner("Entraînement modèle local client_1 (comparaison)..."):
            global_pipe = local_train_for_demo(df, "client_1")
    else:
        with st.spinner("Entraînement modèle local client_2 (comparaison)..."):
            global_pipe = local_train_for_demo(df, "client_2")

    # --- Tab 2: Prediction
    with tab2:
        st.subheader("Formulaire de prédiction")
        st.caption("Pour le modèle fédéré, la prédiction est demandée au serveur via Kafka (`client_data` -> `predictions`).")

        feature_cols = [c for c in df.columns if c != TARGET_COL]

        # Provide a UI-friendly form with a mix of selectboxes for categorical and number_input for numeric
        sample_row = df.drop(columns=[TARGET_COL]).iloc[0].to_dict()
        payload: Dict[str, Any] = {}

        with st.form("predict_form"):
            cols = st.columns(3)
            for i, col_name in enumerate(feature_cols):
                col_ui = cols[i % 3]
                if df[col_name].dtype == "object":
                    options = sorted([x for x in df[col_name].dropna().unique().tolist()])
                    default = sample_row.get(col_name, options[0] if options else "")
                    payload[col_name] = col_ui.selectbox(col_name, options=options, index=(options.index(default) if default in options else 0))
                else:
                    default_val = float(sample_row.get(col_name, 0.0) if sample_row.get(col_name) is not None else 0.0)
                    payload[col_name] = col_ui.number_input(col_name, value=default_val)
            submitted = st.form_submit_button("Prédire")

        if submitted:
            if model_choice.startswith("Modèle fédéré"):
                producer = JsonKafkaProducer(bootstrap_servers=kafka_bootstrap, client_id=f"streamlit-producer-{uuid.uuid4().hex[:8]}")
                with st.spinner("Envoi requête au serveur fédéré via Kafka..."):
                    result = request_prediction_and_wait(producer, kafka_bootstrap, payload, timeout_s=12.0)
                producer.close()

                if result.get("status") != "ok":
                    st.error(f"Erreur: {result}")
                else:
                    st.success(f"Prédiction churn: **{result['prediction']}**")
                    st.metric("Probabilité churn=Yes", f"{result['probability_churn_yes']:.3f}")
                    st.caption(f"Round global: {result.get('round')}")
            else:
                if global_pipe is None:
                    st.error("Modèle indisponible.")
                else:
                    x_row = pd.DataFrame([payload])
                    X_t = global_pipe.named_steps["preprocess"].transform(x_row)
                    prob = float(global_pipe.named_steps["clf"].predict_proba(X_t)[:, 1][0])
                    pred = "Yes" if prob >= 0.5 else "No"
                    st.success(f"Prédiction churn: **{pred}**")
                    st.metric("Probabilité churn=Yes", f"{prob:.3f}")

    # --- Tab 3: Evaluation
    with tab3:
        st.subheader("Accuracy, ROC-AUC, courbe ROC")
        if global_pipe is None:
            st.warning("Modèle indisponible (fédéré non trouvé ou entraînement local non prêt).")
        else:
            metrics_main = evaluate_pipe(global_pipe, df)

            # Compute both locals for comparison
            with st.spinner("Calcul comparaison (locaux vs fédéré)..."):
                local1 = local_train_for_demo(df, "client_1")
                local2 = local_train_for_demo(df, "client_2")
                m1 = evaluate_pipe(local1, df)
                m2 = evaluate_pipe(local2, df)

            # Display metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Modèle sélectionné — Accuracy", f"{metrics_main['accuracy']:.3f}")
            c2.metric("Modèle sélectionné — ROC-AUC", f"{metrics_main['roc_auc']:.3f}")
            if model_choice.startswith("Modèle fédéré") and global_msg and "metrics" in global_msg:
                c3.metric("Metrics publiées serveur (ROC-AUC)", f"{global_msg['metrics'].get('roc_auc', float('nan')):.3f}")

            st.markdown("### Comparaison (sur dataset complet)")
            comp = pd.DataFrame(
                [
                    {"model": "local_client_1", "accuracy": m1["accuracy"], "roc_auc": m1["roc_auc"]},
                    {"model": "local_client_2", "accuracy": m2["accuracy"], "roc_auc": m2["roc_auc"]},
                    {"model": "selected_model", "accuracy": metrics_main["accuracy"], "roc_auc": metrics_main["roc_auc"]},
                ]
            )
            st.dataframe(comp, use_container_width=True)

            # ROC plot
            fig = plt.figure()
            plt.plot(metrics_main["fpr"], metrics_main["tpr"], label="selected_model")
            plt.plot(m1["fpr"], m1["tpr"], label="local_client_1")
            plt.plot(m2["fpr"], m2["tpr"], label="local_client_2")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves")
            plt.legend()
            st.pyplot(fig, clear_figure=True)

            st.caption("NB: Les modèles locaux sont entraînés dans Streamlit uniquement pour comparaison. Le modèle fédéré est consommé depuis Kafka.")


if __name__ == "__main__":
    main()
