import json
from typing import Any, Dict, Iterable, List, Optional

from kafka import KafkaConsumer


class JsonKafkaConsumer:
    """
    Simple Kafka JSON consumer using kafka-python.

    Notes:
    - value_deserializer expects bytes -> dict
    - set group_id to enable consumer groups
    """

    def __init__(
        self,
        topics: List[str],
        bootstrap_servers: str,
        group_id: Optional[str],
        client_id: str,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = True,
        consumer_timeout_ms: int = 1000,
    ) -> None:
        self.topics = topics
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.client_id = client_id

        self._consumer = KafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            client_id=self.client_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v else None,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            consumer_timeout_ms=consumer_timeout_ms,
        )

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for msg in self._consumer:
            yield {
                "topic": msg.topic,
                "partition": msg.partition,
                "offset": msg.offset,
                "timestamp": msg.timestamp,
                "key": msg.key,
                "value": msg.value,
            }

    def close(self) -> None:
        self._consumer.close()
