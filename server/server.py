import io
import os
import time
import json
import numpy
from numpy.typing import NDArray
from server.model_state import GlobalModelState
from typing import Any, Dict, Iterable, Tuple
from flask import Flask, request, jsonify, Response, send_file

ROOT_DIRECTORY: str = os.path.dirname(os.path.dirname(__file__))

server = Flask(
    __name__,
    static_folder=None
)

model_state: GlobalModelState = GlobalModelState(feature_weight=12)


@server.route("/register", methods=["POST"])
def register() -> Response:
    data: Dict[str, Any] = request.json
    client_id: str = data["client_id"]
    model_state.register(client_id=client_id)
    return jsonify(
        {
            "OK": True,
            "clients": model_state.registered
        }
    )


@server.route("/roster", methods=["GET"])
def roster() -> Response:
    return jsonify(
        {
            "clients": model_state.registered
        }
    )


@server.route("/model", methods=["GET"])
def get_model() -> Response:
    return jsonify(
        {
            "training_round": model_state.round,
            "training_weights": model_state.model.get_model_weight().tolist(),
            "feature_weight": model_state.model._dim - 1
        }
    )


@server.route("/configure-training-round", methods=["POST"])
def configure_training_round() -> Response:
    data: Dict[str, Any] = request.json
    participants: Iterable[str] = data.get("participants", [])
    model_state.configure_training_round(
        participants=participants
    )
    return jsonify(
        {
            "OK": True,
            "participants": list(participants)
        }
    )


@server.route("/submit-update", methods=["POST"])
def submit_update() -> Response | Tuple[Response, int]:
    data: Dict[str, Any] = request.json
    client_id: str = data['client_id']
    round: int = data['round']
    vector_array: NDArray[numpy.float64] = numpy.asarray(
        data['masked_update'],
        dtype=float
    )

    if not model_state.expected:
        return jsonify({"OK": False, "error": "round_not_configured"}), 409

    if client_id not in model_state.expected:
        return jsonify({"OK": False, "error": "not_expected"}), 409

    if round != model_state.round:
        print(f"[server] reject {client_id}: wrong_round client={round} server={model_state.round}")
        return jsonify(
            {
                "OK": False,
                "error_message": "wrong round"
            }
        ), 400

    model_state.add_client_data_to_current_model(
        client_id=client_id,
        delta=vector_array
    )
    print(f"[server] accepted {client_id}: received={len(model_state.updates)}/{len(model_state.expected)}")
    completed: bool = model_state.check_all_data_received()
    return jsonify(
        {
            "OK": True,
            "received": len(model_state.updates),
            "all_received": completed
        }
    )


@server.route("/finish-round", methods=["POST"])
def finish_round() -> Response | Tuple[Response, int]:
    if not model_state.check_all_data_received():
        return jsonify(
            {
                "OK": False,
                "error_message": "incomplete"
            }
        ), 400
    round_status: int = model_state.process_and_update_to_global_model()
    return jsonify(
        {
            "OK": True,
            "round": round_status,
            "weight": model_state.model.get_model_weight().tolist()
        }
    )


@server.route("/status", methods=["GET"])
def model_status() -> Response:
    return jsonify(
        {
            "round": model_state.round,
            "registered": model_state.registered,
            "expected": list(model_state.expected),
            "received": list(model_state.updates.keys())
        }
    )


@server.route("/export", methods=["GET"])
def export_model_data():
    payload = {
        "round": model_state.round,
        "feature_weight": model_state.model._dim - 1,
        "training_weights": model_state.model.get_model_weight().tolist(),
        "history": getattr(model_state, "history", []),
        "export_time": int(time.time())
    }

    buffer = io.BytesIO(
        json.dumps(
            payload,
            indent=2
        ).encode("utf-8")
    )

    return send_file(
        buffer,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"model_round_{model_state.round}.json"
    )


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8000, debug=False)
