import numpy
from threading import Lock
from numpy.typing import NDArray
from models import Logistic
from typing import Dict, Iterable, List, Set


class GlobalModelState:
    def __init__(self, feature_weight: int = 12) -> None:
        self.model: Logistic = Logistic(feature_weight)
        self.round: int = 0
        self.registered: List[str] = []
        self.expected: Set[str] = set()
        self.updates: Dict[str, NDArray[numpy.float64]] = {}
        self.lock: Lock = Lock()

    def register(self, client_id: str) -> None:
        with self.lock:
            if client_id not in self.registered:
                self.registered.append(client_id)

    def configure_training_round(self, participants: Iterable[str]) -> None:
        with self.lock:
            self.expected = set(participants)
            self.updates = {}

    def add_client_data_to_current_model(
        self,
        client_id: str,
        delta: NDArray[numpy.float64]
    ) -> None:
        with self.lock:
            self.updates[client_id] = delta

    def check_all_data_received(self) -> bool:
        with self.lock:
            return set(self.updates.keys()) == self.expected

    def process_and_update_to_global_model(self) -> int:
        with self.lock:
            mats_array: NDArray[numpy.float64] = numpy.stack(
                list(self.updates.values()),
                axis=0
            )
            aggregate: NDArray[numpy.float64] = mats_array.mean(axis=0)
            self.model.set_model_weight(
                self.model.set_model_weight() + aggregate
            )
            self.round += 1
            self.expected.clear()
            self.updates.clear()
            return self.round
