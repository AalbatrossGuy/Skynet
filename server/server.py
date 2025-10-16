import os
import numpy
from numpy.typing import NDArray
from model_state import GlobalModelState
from typing import Any, Dict, Iterable, List, Set, Tuple
from flask import Flask, request, jsonify, send_from_directory, Response

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
            "feature_weight": model_state.model.dim
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
    vector_array: NDArray[numpy.float64] = numpy.array(
        data['masked_update'],
        dtype=float
    )

    if round != model_state.round:
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
    completed: bool = model_state.check_all_data_received()
    return jsonify(
        {
            "OK": True,
            "received": len(model_state.updates),
            "all_received": completed
        }
    )

@app.route("/finish-round", methods=["POST"])
def finish_round() -> Response | Tuple[Response, int]:
    if not model_state.all_received():
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
            "weight": model_state.get_model_weight().tolist()
        }
    )
    
@app.route("/status", methods=["GET"])
def model_status() -> Response:
    return jsonify(
        {
            "round": model_state.round,
            "registered": model_state.registered,
            "expected": list(model_state.expected),
            "received": list(model_state.updates.keys())
        }    
    )