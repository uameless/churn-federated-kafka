import json
import time
from typing import Dict, Any, Optional

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable


class JsonKafkaProducer:
    def __init__(self, bootstrap_servers: str, client_id: str):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._producer = None

        for attempt in range(30):  # ⏳ attendre Kafka ~30s
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    client_id=self.client_id,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    retries=5,
                    acks="all",
                )
                print("[KafkaProducer] Connected to Kafka")
                break
            except NoBrokersAvailable:
                print(f"[KafkaProducer] Kafka not ready, retry {attempt+1}/30")
                time.sleep(2)

        if self._producer is None:
            raise RuntimeError("Kafka broker not available after retries")

    def send(self, topic: str, message: dict, key: str | None = None):
        self._producer.send(topic, value=message, key=key.encode() if key else None)
        self._producer.flush()


    def flush(self) -> None:
        self._producer.flush()

    def close(self) -> None:
        try:
            self._producer.flush()
        finally:
            self._producer.close()


def send_with_retry(
    producer: JsonKafkaProducer,
    topic: str,
    message: Dict[str, Any],
    key: Optional[str] = None,
    max_attempts: int = 10,
    sleep_s: float = 2.0,
) -> None:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            producer.send(topic=topic, message=message, key=key)
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to send message to topic={topic} after {max_attempts} attempts") from last_err
