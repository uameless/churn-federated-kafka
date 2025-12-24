from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fl.aggregation import fedavg
from messaging.consumer import JsonKafkaConsumer
from messaging.producer import JsonKafkaProducer, send_with_retry
from model.churn_model import (
    TARGET_COL,
    build_pipeline,
    build_preprocessor,
    create_lr_from_weights,
    evaluate_binary_classifier,
    load_telco_csv,
)

TOPIC_CLIENT_DATA = "client_data"
TOPIC_LOCAL_UPDATES = "local_model_updates"
TOPIC_GLOBAL_MODEL = "global_model"
TOPIC_PREDICTIONS = "predictions"


def build_global_pipeline_from_weights(df_schema: pd.DataFrame, coef: np.ndarray, intercept: np.ndarray) -> Pipeline:
    X_full = df_schema.drop(columns=[TARGET_COL])

    categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_full.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in numeric_cols]

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    pipe = build_pipeline(preprocessor)

    # Fit preprocess to align OHE feature space
    pipe.named_steps["preprocess"].fit(X_full, df_schema[TARGET_COL].to_numpy())

    # Set LR weights
    lr = create_lr_from_weights(coef=coef, intercept=intercept)
    pipe.named_steps["clf"] = lr
    return pipe


def evaluate_global_model(pipe: Pipeline, df_eval: pd.DataFrame) -> Dict:
    X = df_eval.drop(columns=[TARGET_COL])
    y = df_eval[TARGET_COL].to_numpy()

    X_t = pipe.named_steps["preprocess"].transform(X)
    probs = pipe.named_steps["clf"].predict_proba(X_t)[:, 1]
    metrics = evaluate_binary_classifier(y_true=y, y_prob=probs)
    return {"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]}


def main() -> None:
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    csv_path = os.getenv("CSV_PATH", "/app/data/telco_churn.csv")
    expected_clients = int(os.getenv("EXPECTED_CLIENTS", "2"))
    rounds = int(os.getenv("ROUNDS", "1"))

    producer = JsonKafkaProducer(bootstrap_servers=kafka_bootstrap, client_id="fl-server-producer")

    # Server loads dataset locally (no raw sharing)
    df_full = load_telco_csv(csv_path)

    consumer_updates = JsonKafkaConsumer(
        topics=[TOPIC_LOCAL_UPDATES],
        bootstrap_servers=kafka_bootstrap,
        group_id="fl-server-updates",
        client_id="fl-server-updates-consumer",
        auto_offset_reset="earliest",
        consumer_timeout_ms=2000,
    )

    consumer_client_data = JsonKafkaConsumer(
        topics=[TOPIC_CLIENT_DATA],
        bootstrap_servers=kafka_bootstrap,
        group_id="fl-server-client-data",
        client_id="fl-server-client-data-consumer",
        auto_offset_reset="earliest",
        consumer_timeout_ms=2000,
    )

    latest_global: Optional[Tuple[np.ndarray, np.ndarray, int]] = None  # (coef, intercept, round)
    buffer_by_round: Dict[int, List[Dict]] = {r: [] for r in range(1, rounds + 1)}
    done_rounds = set()

    print("[server] Running. Will train federated rounds then keep serving predictions forever.")

    # ✅ IMPORTANT: do NOT stop after training; keep serving prediction requests.
    while True:
        # 1) Consume local updates and aggregate (training phase)
        for msg in consumer_updates:
            val = msg["value"]
            if not val or val.get("type") != "local_update":
                continue

            r = int(val["round"])
            if r < 1 or r > rounds:
                continue

            buffer_by_round[r].append(val)
            got = len(buffer_by_round[r])
            print(f"[server] Received local_update round={r} ({got}/{expected_clients}) from {val.get('client_id')}")

            if got >= expected_clients and r not in done_rounds:
                coef_g, intercept_g = fedavg(buffer_by_round[r])
                latest_global = (coef_g, intercept_g, r)

                pipe_g = build_global_pipeline_from_weights(df_schema=df_full, coef=coef_g, intercept=intercept_g)
                metrics = evaluate_global_model(pipe_g, df_full)

                global_msg: Dict = {
                    "type": "global_model",
                    "round": r,
                    "model": "logistic_regression",
                    "coef": coef_g.tolist(),
                    "intercept": intercept_g.tolist(),
                    "metrics": metrics,
                    "ts": time.time(),
                }
                send_with_retry(producer, TOPIC_GLOBAL_MODEL, global_msg, key="global")
                print(f"[server] ✅ Published global model for round={r} metrics={metrics}")
                done_rounds.add(r)

        # 2) Serve predictions (always)
        for msg in consumer_client_data:
            val = msg["value"]
            if not val:
                continue

            if val.get("type") != "predict_request":
                continue

            request_id = val.get("request_id")
            payload = val.get("payload", {})
            source = val.get("source", "unknown")

            if latest_global is None:
                pred_msg = {
                    "type": "prediction_result",
                    "request_id": request_id,
                    "source": source,
                    "status": "no_global_model",
                    "message": "Global model not available yet. Train federated rounds first.",
                    "ts": time.time(),
                }
                send_with_retry(producer, TOPIC_PREDICTIONS, pred_msg, key=str(request_id))
                continue

            coef_g, intercept_g, r_g = latest_global
            pipe_g = build_global_pipeline_from_weights(df_schema=df_full, coef=coef_g, intercept=intercept_g)

            x_row = pd.DataFrame([payload])
            expected_cols = [c for c in df_full.columns if c != TARGET_COL]
            for c in expected_cols:
                if c not in x_row.columns:
                    x_row[c] = np.nan
            x_row = x_row[expected_cols]

            x_t = pipe_g.named_steps["preprocess"].transform(x_row)
            prob = float(pipe_g.named_steps["clf"].predict_proba(x_t)[:, 1][0])
            pred = "Yes" if prob >= 0.5 else "No"

            pred_msg = {
                "type": "prediction_result",
                "request_id": request_id,
                "source": source,
                "status": "ok",
                "round": r_g,
                "prediction": pred,
                "probability_churn_yes": prob,
                "ts": time.time(),
            }
            send_with_retry(producer, TOPIC_PREDICTIONS, pred_msg, key=str(request_id))
            print(f"[server] ✅ prediction_result sent request_id={request_id} pred={pred} prob={prob:.3f}")

        time.sleep(0.3)


if __name__ == "__main__":
    main()
