# federated-explainable-rl-iot
Federated Explainable Reinforcement Learning (XRL) framework for IoT threat detection using PyTorch and TensorFlow. Supports centralized and federated training (FedAvg), DQN/PPO agents, supervised classifier head, and explainability (SHAP, LIME, IG). Includes dataset loaders, evaluation, FastAPI deployment, and CI.
Federated Explainable Reinforcement Learning (XRL) for IoT Threat Detection with PyTorch and TensorFlow backends. Includes centralized and federated training (FedAvg), explainability (SHAP, Integrated Gradients, LIME), and evaluation on popular intrusion datasets.

Badges (replace main with your default branch if needed):








Overview:
This repository implements a research-grade pipeline for IoT network threat detection using reinforcement learning agents and supervised heads, trained in both centralized and federated settings. The goal is to (1) detect intrusions and anomalous flows in IoT networks, (2) preserve data locality via federated averaging, and (3) produce faithful, human-interpretable explanations for model decisions. The codebase offers mirrored functionality in PyTorch and TensorFlow, making it suitable for reproducibility, benchmarking, and deployment.

Key Features:

Dual Backends: PyTorch and TensorFlow implementations with matching APIs.

Federated Learning: FedAvg via Flower; configurable client heterogeneity (data, compute, connectivity).

RL Agents: DQN and PPO baselines for sequential decision policies (e.g., adaptive feature selection, thresholding, or active defense action space).

Hybrid Training: RL policy with an auxiliary supervised classifier head for robust detection.

Explainability (XAI): SHAP (Tree/Deep), Integrated Gradients, LIME; per-sample and global importance reports.

Datasets: Ready loaders for CIC-IDS2017 and UNSW-NB15 (you place data locally). CSV/Parquet supported.

Reproducibility: Deterministic seeds, Hydra configs, Dockerfiles, CI checks.

Metrics: Accuracy, F1, ROC-AUC, Precision-Recall AUC; RL returns, cost-aware policies; fairness slices by device/site.

Export: ONNX and TorchScript (PyTorch), SavedModel and TFLite (TensorFlow). Optional quantization.

Deployment: Minimal FastAPI inference service with batched scoring and explanation endpoints.

Repository Structure:
.
├── backends
│ ├── torch_backend
│ │ ├── models
│ │ │ ├── rl_dqn.py
│ │ │ ├── rl_ppo.py
│ │ │ └── classifier_head.py
│ │ ├── train_centralized.py
│ │ ├── train_federated.py
│ │ ├── explain.py
│ │ └── export.py
│ └── tf_backend
│ ├── models
│ │ ├── rl_dqn.py
│ │ ├── rl_ppo.py
│ │ └── classifier_head.py
│ ├── train_centralized.py
│ ├── train_federated.py
│ ├── explain.py
│ └── export.py
├── federated
│ ├── flower_server.py
│ ├── flower_client_torch.py
│ └── flower_client_tf.py
├── data
│ ├── README_DATA.txt
│ └── (place datasets here)
├── configs
│ ├── default.yaml
│ ├── torch.yaml
│ └── tf.yaml
├── datasets
│ ├── cicids2017.py
│ └── unsw_nb15.py
├── evaluation
│ ├── metrics.py
│ └── fairness_slices.py
├── deployment
│ ├── fastapi_app.py
│ ├── requirements.txt
│ └── Dockerfile
├── scripts
│ ├── prepare_cicids2017.sh
│ ├── prepare_unsw_nb15.sh
│ ├── run_torch_centralized.sh
│ ├── run_tf_centralized.sh
│ ├── run_federated_torch.sh
│ ├── run_federated_tf.sh
│ └── explain_sample.sh
├── tests
│ ├── test_loaders.py
│ ├── test_models.py
│ └── test_end_to_end.py
├── .github
│ └── workflows
│ └── ci.yml
├── requirements_torch.txt
├── requirements_tf.txt
├── environment.yml
├── LICENSE
└── README.md (this file)

IoT Threat Model (example):
Actions: drop, allow, rate-limit, isolate device, request verification, escalate.
States: recent flow features, device metadata, historical decisions, queue length.
Rewards: +TP for blocking malicious, −FP penalty for blocking benign, small step cost, optional fairness regularizer.

Installation:
Option A: PyTorch stack
python -m venv .venv && . .venv/bin/activate (Linux/macOS)
or .venv\Scripts\activate (Windows)
pip install -r requirements_torch.txt

Option B: TensorFlow stack
python -m venv .venv && . .venv/bin/activate (Linux/macOS)
or .venv\Scripts\activate (Windows)
pip install -r requirements_tf.txt

Optional: Conda
conda env create -f environment.yml
conda activate ferlit

Datasets:

CIC-IDS2017: download CSVs to data/cicids2017 and run scripts/prepare_cicids2017.sh

UNSW-NB15: download to data/unsw_nb15 and run scripts/prepare_unsw_nb15.sh
The loaders handle standard splits; you can configure non-IID splits per client.

Quick Start (Centralized, PyTorch):
python backends/torch_backend/train_centralized.py
dataset=cicids2017
model=dqn
epochs=10
batch_size=512
lr=1e-3
aux_classifier=true
save_dir=outputs/torch_centralized

Quick Start (Centralized, TensorFlow):
python backends/tf_backend/train_centralized.py
dataset=unsw_nb15
model=ppo
epochs=10
batch_size=512
lr=1e-3
aux_classifier=true
save_dir=outputs/tf_centralized

Federated Training (Flower, PyTorch):

Terminal 1: server

python federated/flower_server.py
strategy=fedavg
rounds=10
min_available_clients=3

Terminal 2..N: clients

python federated/flower_client_torch.py
dataset=cicids2017
client_id=site_1
non_iid=true
local_epochs=1
save_dir=outputs/fed_torch/site_1

Repeat client command for multiple simulated sites with different client_id and data shards.

Federated Training (Flower, TensorFlow):

Server

python federated/flower_server.py strategy=fedavg rounds=10 min_available_clients=3

Clients

python federated/flower_client_tf.py dataset=unsw_nb15 client_id=site_1 non_iid=true local_epochs=1 save_dir=outputs/fed_tf/site_1

Explainability:
Per-sample explanations:
python backends/torch_backend/explain.py
checkpoint=outputs/torch_centralized/best.pt
method=shap
samples=50
out_dir=outputs/explain/torch

Global importance via Integrated Gradients:
python backends/tf_backend/explain.py
checkpoint=outputs/tf_centralized/best
method=integrated_gradients
samples=200
out_dir=outputs/explain/tf

Exports:
PyTorch to ONNX:
python backends/torch_backend/export.py
checkpoint=outputs/torch_centralized/best.pt
format=onnx
out=artifacts/model.onnx

TensorFlow to TFLite (float16):
python backends/tf_backend/export.py
checkpoint=outputs/tf_centralized/best
format=tflite
quant=float16
out=artifacts/model.tflite

API Inference (FastAPI):
pip install -r deployment/requirements.txt
uvicorn deployment.fastapi_app:app --host 0.0.0.0 --port 8000
Endpoints:
POST /predict -> batch intrusion predictions
POST /explain -> per-record feature attributions

Configuration (Hydra style keys shown inline):
dataset: cicids2017 | unsw_nb15 | custom_csv
model: dqn | ppo
aux_classifier: true | false
non_iid: true | false (federated)
rounds, local_epochs, batch_size, lr, gamma, epsilon, clip_range, entropy_coef
explain.method: shap | ig | lime
export.format: onnx | torchscript | savedmodel | tflite

Reproducibility:
Set seeds via config: seed=42
All dataloaders and client splits use seeded generators.
CI runs a tiny smoke test on synthetic data.

Research Protocol (default):

Centralized training on CIC-IDS2017 and UNSW-NB15; report ROC-AUC, F1, PR-AUC.

Federated training with K=5 clients, non-IID Dirichlet alpha=0.3; compare against centralized.

RL ablation: purely supervised vs hybrid RL+supervised; DQN vs PPO.

XAI evaluation: stability of attributions across clients; agreement with known attack features.

Fairness slices: device type, site, traffic volume; report worst-group gap in F1.

Cite This Repository (BibTeX template):
@software{zakawatliaqat-eng_fedxrl_iot_2025,
author = {Zakawat Liaqat},
title = {Federated Explainable Reinforcement Learning for IoT Threat Detection},
year = {2025},
url = {https://github.com/zakawatliaqat-eng/federated-explainable-rl-iot}

}

Contributing:
Pull requests are welcome. Please open an issue first for significant changes.
Coding style: black/ruff (PyTorch), yapf/flake8 (TensorFlow). See tests/ for unit coverage.

Security:
This code is for research. Do not deploy directly to production networks without thorough validation, threat modeling, and compliance checks.

License:
MIT License (default). You can change to Apache-2.0 or GPL-3.0 if your collaborators require it. Update the LICENSE file and the badge above accordingly.

Maintainer:
zakawatliaqat-eng

Minimal Code Stubs (you can paste these files to make the repo runnable):

File: backends/torch_backend/models/classifier_head.py

import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
def init(self, in_dim, num_classes):
super().init()
self.net = nn.Sequential(
nn.Linear(in_dim, 256),
nn.ReLU(),
nn.Linear(256, num_classes)
)
def forward(self, x):
return self.net(x)

File: backends/torch_backend/models/rl_dqn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
def init(self, state_dim, action_dim):
super().init()
self.q = nn.Sequential(
nn.Linear(state_dim, 256),
nn.ReLU(),
nn.Linear(256, action_dim)
)
def forward(self, s):
return self.q(s)
def act(self, s, eps=0.1):
if torch.rand(1).item() < eps:
return torch.randint(0, self.q[-1].out_features, (1,)).item()
with torch.no_grad():
return torch.argmax(self.forward(s)).item()

File: backends/torch_backend/train_centralized.py

import os, argparse, torch
from torch.utils.data import DataLoader, TensorDataset
from models.rl_dqn import DQN
from models.classifier_head import ClassifierHead

def load_dummy(n=2000, d=64, k=2):
x = torch.randn(n, d)
y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
return TensorDataset(x, y), d, k

def train(args):
ds, in_dim, num_classes = load_dummy()
dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

policy = DQN(in_dim, action_dim=3).to(device)
clf = ClassifierHead(in_dim, num_classes).to(device)
opt = torch.optim.Adam(list(policy.parameters())+list(clf.parameters()), lr=args.lr)
ce = torch.nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        q = policy(xb)
        logits = clf(xb)
        rl_loss = q.mean() * 0.0  # placeholder; integrate real RL loss with env
        sup_loss = ce(logits, yb)
        loss = sup_loss + rl_loss
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch+1} loss {loss.item():.4f}")

os.makedirs(args.save_dir, exist_ok=True)
torch.save({"policy": policy.state_dict(), "clf": clf.state_dict()}, os.path.join(args.save_dir, "best.pt"))
print("Saved", os.path.join(args.save_dir, "best.pt"))


if name == "main":
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--save_dir", type=str, default="outputs/torch_centralized")
args = p.parse_args()
train(args)

File: federated/flower_server.py

import flwr as fl
from flwr.server.strategy import FedAvg

def main():
strategy = FedAvg()
fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=5))

if name == "main":
main()

File: federated/flower_client_torch.py

import flwr as fl, torch
from torch.utils.data import DataLoader, TensorDataset
from backends.torch_backend.models.classifier_head import ClassifierHead

def get_data(n=400, d=64):
x = torch.randn(n, d)
y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
return TensorDataset(x, y)

class TorchClient(fl.client.NumPyClient):
def init(self):
self.model = ClassifierHead(64, 2)
self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
self.loss = torch.nn.CrossEntropyLoss()
self.trainloader = DataLoader(get_data(), batch_size=128, shuffle=True)
self.testloader = DataLoader(get_data(200), batch_size=128)
