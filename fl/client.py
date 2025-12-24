from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from messaging.consumer import JsonKafkaConsumer
from messaging.producer import JsonKafkaProducer, send_with_retry
from model.churn_model import (
    TARGET_COL,
    build_pipeline,
    build_preprocessor,
    extract_lr_weights_from_pipeline,
    load_telco_csv,
)


TOPIC_CLIENT_DATA = "client_data"
TOPIC_LOCAL_UPDATES = "local_model_updates"
TOPIC_GLOBAL_MODEL = "global_model"


def partition_dataframe(df: pd.DataFrame, client_id: str, n_clients: int = 2) -> pd.DataFrame:
    """
    Split by rows deterministically to simulate federated partitions.
    client_1 -> first half, client_2 -> second half
    """
    if client_id not in {"client_1", "client_2"} and n_clients == 2:
        raise ValueError("For n_clients=2, client_id must be 'client_1' or 'client_2'.")

    n = len(df)
    mid = n // 2
    if client_id == "client_1":
        return df.iloc[:mid].copy()
    return df.iloc[mid:].copy()


def fit_local_model(
    df_full_for_schema: pd.DataFrame,
    df_partition: pd.DataFrame,
) -> Tuple[Pipeline, int]:
    """
    IMPORTANT: Preprocessor is fitted on df_full_for_schema to ensure consistent feature space
    across clients without sharing raw client data via Kafka.

    Local training uses only df_partition.
    """
    # Build consistent preprocessor from full dataset schema
    X_full = df_full_for_schema.drop(columns=[TARGET_COL])
    y_full = df_full_for_schema[TARGET_COL].to_numpy()

    # Infer columns
    categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_full.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in numeric_cols]

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    # Fit preprocessor on full data (schema alignment), but classifier on local partition only
    pipe = build_pipeline(preprocessor)
    # Fit preprocess step only:
    pipe.named_steps["preprocess"].fit(X_full, y_full)

    # Now transform local data and fit classifier
    X_local = df_partition.drop(columns=[TARGET_COL])
    y_local = df_partition[TARGET_COL].to_numpy()

    X_local_t = pipe.named_steps["preprocess"].transform(X_local)
    pipe.named_steps["clf"].fit(X_local_t, y_local)

    return pipe, int(len(df_partition))


def main() -> None:
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    client_id = os.getenv("CLIENT_ID", "client_1")
    csv_path = os.getenv("CSV_PATH", "/app/data/telco_churn.csv")
    rounds = int(os.getenv("ROUNDS", "1"))

    producer = JsonKafkaProducer(bootstrap_servers=kafka_bootstrap, client_id=f"{client_id}-producer")

    # Optional: send a "data streaming" sample row to client_data to satisfy streaming requirement.
    # This is a *simulation*; we do not send full raw dataset.
    df_full = load_telco_csv(csv_path)
    df_part = partition_dataframe(df_full, client_id=client_id, n_clients=2)

    # Stream a few sanitized samples (no raw bulk data)
    for i in range(min(3, len(df_part))):
        sample = df_part.drop(columns=[TARGET_COL]).iloc[i].to_dict()
        msg = {"type": "sample", "client_id": client_id, "payload": sample, "ts": time.time()}
        send_with_retry(producer, TOPIC_CLIENT_DATA, msg, key=client_id)

    # Local training + sending weights
    for r in range(1, rounds + 1):
        pipe, n_samples = fit_local_model(df_full_for_schema=df_full, df_partition=df_part)
        coef, intercept = extract_lr_weights_from_pipeline(pipe)

        update_msg: Dict = {
            "type": "local_update",
            "round": r,
            "client_id": client_id,
            "n_samples": n_samples,
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
            "model": "logistic_regression",
            "ts": time.time(),
        }
        send_with_retry(producer, TOPIC_LOCAL_UPDATES, update_msg, key=client_id)
        print(f"[{client_id}] Sent local update for round={r} (n_samples={n_samples})")

        # Optionally listen for global model to confirm reception
        consumer = JsonKafkaConsumer(
            topics=[TOPIC_GLOBAL_MODEL],
            bootstrap_servers=kafka_bootstrap,
            group_id=f"{client_id}-global-listener",
            client_id=f"{client_id}-consumer",
            auto_offset_reset="latest",
            consumer_timeout_ms=3000,
        )
        got = False
        for msg in consumer:
            val = msg["value"]
            if val and val.get("type") == "global_model" and val.get("round") == r:
                print(f"[{client_id}] Received global model for round={r}")
                got = True
                break
        consumer.close()

        if not got:
            print(f"[{client_id}] No global model received yet for round={r} (continuing).")

        time.sleep(2)

    producer.close()


if __name__ == "__main__":
    main()
