import time
import numpy
import click
import requests
from numpy.typing import NDArray
from models.models import Logistic
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score
from client.data import generate_dataset_local
from models.crypto import pseudo_random_generator, derive_pair_seed

SECRET = b"shared_secret"


def client(
    server: str,
    client_id: str,
    samples: int,
    rounds: int,
    learning_rate: float,
    seed: int
):
    base = server.rstrip("/")
    requests.post(
        f"{base}/register",
        json={
            "client_id": client_id
        }
    )

    feature_weights = requests.get(f"{base}/model").json()["feature_weight"]
    X_matrix, y = generate_dataset_local(
        samples,
        feature_weights,
        seed + hash(client_id) % 1000
    )
    model: Logistic = Logistic(feature_weights)

    for _ in range(int(rounds)):
        model_info: Dict[str, Any] = requests.get(f"{base}/model").json()
        weights: NDArray[numpy.float64] = numpy.array(
            model_info["feature_weight"],
            dtype=numpy.float64
        )
        model.set_model_weight(weights)

        delta: NDArray[numpy.float64] = model.update_local(
            feature_matrix=X_matrix,
            binary_targets=y,
            epochs=1,
            learning_rate=float(learning_rate)
        )
        roster_response: Dict[str, Any] = requests.get(f"{base}/roster").json()
        roster: List[str] = list(roster_response["clients"])
        dimensions: int = int(delta.shape[0])
        mask: NDArray[numpy.float64] = numpy.zeroes(
            dimensions,
            dtype=numpy.float64
        )

        for peer in roster:
            if peer == client_id:
                continue
            seed_bytes: bytes = derive_pair_seed(
                client_secret=SECRET,
                identifier_a=client_id,
                identifier_b=peer
            )
            vector: NDArray[numpy.float64] = pseudo_random_generator(
                seed=seed_bytes,
                length=dimensions
            )
            mask = mask + vector if client_id < peer else mask - vector

        masked: List[float] = (delta + mask).astype(float).tolist()
        send_body: Dict[str, Any] = {
            "client_id": client_id,
            "round": int(model_info["round"]),
            "masked_update": masked
        }
        response: Dict[str, Any] = requests.post(
            url=f"{base}/submit-update",
            json=send_body
        ).json()

        print(f"[{client_id}] round={model_info['round']}\
        received={response.get('received')}")

        time.sleep(0.5)

        while True:
            status_response: Dict[str, Any] = requests.get(
                f"{base}/status"
            ).json()
            if not list(status_response["expected"]):
                break
            time.sleep(0.5)

        accuracy: float = float(accuracy_score(y, model.predict(X_matrix)))
        print(f"[{client_id}] local accuracy \
        after round {model_info['round']}: {accuracy: .3f}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--server", default="http://127.0.0.1:8000", show_default=True, help="Base URL for the server.")
@click.option("--client-id", "client_id", required=True, help="Unique client identifier.")
@click.option("--samples", type=int, default=300, show_default=True, help="Number of local samples to generate.")
@click.option("--rounds", type=int, default=5, show_default=True, help="Number of federated rounds to participate in.")
@click.option("--lr", type=float, default=0.5, show_default=True, help="Learning rate for local update.")
@click.option("--seed", type=int, default=1234, show_default=True, help="Base RNG seed for local data generation.")
def skynet_cli(server: str, client_id: str, samples: int, rounds: int, lr: float, seed: int) -> None:
    client(
        server=server,
        client_id=client_id,
        samples=samples,
        rounds=rounds,
        learning_rate=lr,
        seed=seed
    )


if __name__ == "__main__":
    skynet_cli()
