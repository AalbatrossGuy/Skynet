import time
import click
import requests
from typing import Any, Dict, List


def coordinator(
    server: str,
    rounds: int,
    minimum_clients_registered: int
):
    base: str = server.rstrip("/")
    per_round_time: List[float] = []
    time_total_0 = time.perf_counter()

    while True:
        roster_response: Dict[str, Any] = requests.get(f"{base}/roster").json()
        client_roster: List[str] = list(roster_response["clients"])
        if len(client_roster) >= minimum_clients_registered:
            print("Roster:", client_roster)
            break
        print("💤 Waiting for clients...")
        time.sleep(1)

    for round in range(int(rounds)):
        time_0 = time.perf_counter()
        configure_round: Dict[str, Any] = requests.post(
            f"{base}/configure-training-round", json={"participants": client_roster}
        ).json()
        print(f"[Round {round}] configured {configure_round['participants']}")

        time_wait_0 = time.perf_counter()

        while True:
            status_response: Dict[str, Any] = requests.get(f"{base}/status").json()
            received: List[str] = list(status_response["received"])
            expected: List[str] = list(status_response["expected"])
            print(f"[Round {round}] received {len(received)}/{len(expected)} updates.", end="\r")
            if set(received) == set(expected):
                break

            if time.perf_counter() - time_wait_0 > 120:
                print(f"\n[Round {round}] timeout waiting for updates -> proceeding...")
                break

            time.sleep(0.5)

        completed: Dict[str, Any] = requests.post(f"{base}/finish-round").json()
        time_1 = time.perf_counter()
        time_elapsed = time_1 - time_0
        per_round_time.append(time_elapsed)
        print(f"\n[Round {round}] aggregated. Now round={completed['round']}")

    time_total_1 = time.perf_counter()
    model = requests.get(f"{base}/model").json()
    weight = model.get("training_weights", []) or []
    weight_normal = (sum(x * x for x in weight)) ** 0.5 if weight else 0.0

    print("\n===== TRAINING SUMMARY =====")
    print(f"Rounds run           : {rounds}")
    if per_round_time:
        print(f"Time per round (s)   : {', '.join(f'{second:.2f}' for second in per_round_time)}")
        print(f"Total time (s)       : {time_total_1 - time_total_0:.2f}")
        print(f"Avg round time (s)   : {sum(per_round_time)/len(per_round_time):.2f}")
    print(f"Final server round   : {model.get('training_round')}")
    print(f"Weight vector length : {len(weight)}")
    print(f"||w||₂               : {weight_normal:.4f}")
    if weight:
        head = ", ".join(f"{x:.4f}" for x in weight[:5])
        print(f"First 5 weights      : [{head}{', …' if len(weight) > 5 else ''}]")
    print(f"Export URL           : {base}/export  (JSON download)\n")
    print("✅ Training finished.")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--server", default="http://127.0.0.1:8000", show_default=True, help="Base URL of the federated server.")
@click.option("--rounds", type=int, default=30, show_default=True, help="Number of training rounds to run.")
@click.option("--min-clients", "min_clients", type=int, default=3, show_default=True, help="Minimum number of clients required to start.")
def controller_cli(server: str, rounds: int, min_clients: int) -> None:
    coordinator(
        server=server,
        rounds=rounds,
        minimum_clients_registered=min_clients
    )


if __name__ == "__main__":
    controller_cli()
