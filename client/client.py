import time, requests, numpy
from models.models import Logistic
from sklearn.metrics import accuracy_score
from client.data import generate_dataset_local
from models.crypto import pseudo_random_generator, derive_pair_seed

SECRET = b""

def client(args):
    base = args.server.rstrip("/")
    client_id = args.client_id
    
    requests.post(
        f"{base}/register",
        json = {
            "client_id": client_id
        }
    )
    
    feature_weights = requests.get(f"{base}/model").json()["feature_weight"]
    