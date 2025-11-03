# Skynet <img width="200" height="200" alt="skynet.png" src="https://github.com/user-attachments/assets/46dbb274-83ad-4bf3-adfc-0e6cc906444d" align="right" />
Skynet is a distributed research framework that trains a global model across multiple heterogeneous clients without sharing raw data and visualizes model evolution. The lightweight model is trained locally to preserve privacy, meet governance rules, reduce risk of model hijacking, and fitting the structured, private nature of threat data.
<br clear="right" />
## Overview
Skynet implements a **federated learning system** designed for cybersecurity prototype research.
It features:

- **Modular Components**
  - **Server** — coordinates aggregation, maintains the global model.
  - **Clients** — perform local training on synthetic or custom datasets.
  - **Controller** — orchestrates rounds, ensures all updates are received.
- **Secure Aggregation**  
  Clients mask updates with pairwise random generators (PRGs) so the server never sees raw gradients.
- **Analytics Suite**  
  CLI-driven chart generator with both **static PNGs** and **animated GIFs** (Matplotlib + Pillow).
- **Automation**  
  One-click orchestration through a single bash script (`scripts/skynet.sh`).
- **Visualizations**  
  Accuracy, weight norms, and training weight distributions are automatically exported for analysis.

---

## 📁 Project Structure
```
Skynet/
├── analytics/
│ ├── __init__.py
│ └── charts.py
├── client/
│ ├── client.py
│ ├── data.py
│ └── __init__.py
├── controller/
│ ├── controller.py
│ └── __init__.py <br>
├── models/
│ ├── crypto.py
│ ├── models.py
│ ├── network.py
│ └── __init__.py
├── server/
│ ├── server.py
│ ├── model_state.py
│ └── __init__.py
├── scripts/
│ ├── install.sh
│ └── skynet.sh
├── logs/
│ └── exports/
├── requirements.txt
└── README.md
```


---

## Installation

### Prerequisites

- Python **3.10+**
- `git`, `curl`, and `bash`

### One-liner (Recommended)

```bash
ROUNDS="5" MIN_CLIENTS="3" CLIENT_SAMPLES="300" CLIENT_ROUNDS="5" CLIENT_LR="0.5" SERVER_HOST="127.0.0.1"
SERVER_PORT="8000" curl -fsSL -o install.sh https://kishaloyroy.dev/install.sh
chmod +x install.sh && ./install.sh && cd Skynet
./scripts/skynet.sh start
```

### Install (Don't Run)
```bash
ROUNDS="5" MIN_CLIENTS="3" CLIENT_SAMPLES="300" CLIENT_ROUNDS="5" CLIENT_LR="0.5" SERVER_HOST="127.0.0.1"
SERVER_PORT="8000" bash <(curl -fsSL https://kishaloyroy.dev/install.sh)
```

### Manual Setup
```bash
git clone https://github.com/AalbatrossGuy/Skynet.git
cd Skynet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/skynet.sh start
./scripts/skynet.sh stop # For stopping
```
<br>

### Environment Variables
They can be set anytime before running any command:
```bash
export ROUNDS=10
export CLIENTS="A B C D"
export CLIENT_SAMPLES=500
export CLIENT_ROUNDS=6
export CLIENT_LR=0.4
export SERVER_HOST=0.0.0.0
export SERVER_PORT=8000
./scripts/skynet.sh start
```
---

### `skynet.sh` Command Description
| Command            | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `setup`            | Create venv and install requirements                         |
| `start`            | Start server, clients, and controller (auto export + charts) |
| `start-server`     | Start only the server                                        |
| `start-clients`    | Start only clients                                           |
| `start-controller` | Start only controller                                        |
| `stop`             | Stop all components                                          |
| `status`           | Show process and server status                               |
| `logs`             | Tail all logs in real time                                   |

<br>

### `skynet.sh` Environment Variable Description
| Variable          | Default                    | Description                                |
| ----------------- | -------------------------- | ------------------------------------------ |
| `ROUNDS`          | 5                          | Number of federated training rounds        |
| `MIN_CLIENTS`     | 3                          | Minimum clients per round                  |
| `CLIENTS`         | `A B C`                    | Client identifiers                         |
| `CLIENT_SAMPLES`  | 300                        | Samples per client dataset                 |
| `CLIENT_ROUNDS`   | 5                          | Local epochs per client                    |
| `CLIENT_LR`       | 0.5                        | Local learning rate                        |
| `SERVER_HOST`     | 127.0.0.1                  | Server bind address                        |
| `SERVER_PORT`     | 8000                       | Server port                                |
| `EXPORT_DIR`      | `logs/exports`             | Where exports & charts are saved           |
| `EXPORT_BASENAME` | `model_state_summary.json` | Base name for export file                  |
| `AUTO_STOP`       | 1                          | Automatically stop everything after export |
| `SETTLE_TIMEOUT`  | 10                         | Seconds to wait before exporting           |

---

### Export Directory Structure
```
logs/
├── client_A.out
├── client_B.out
├── controller.out
├── server.out
└── exports/
    ├── export.json
    ├── export_accuracy_per_client.png
    ├── export_weight_norm.png
    ├── export_final_weights.png
```

### Program Flow
-> **Initialization** — Server starts, clients register, and controller configures rounds. <br>
-> **Local Training** — Each client trains on its synthetic data and computes a local update. <br>
-> **Secure Aggregation** — Clients mask updates with pairwise random vectors (PRGs). <br>
-> **Global Update** — Server aggregates masked updates; masks cancel out. <br>
-> **Repeat** — Controller triggers the next round until completion. <br>
-> **Export + Visualization** — Server exports final state (export.json); analytics generates charts.
