#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "solution" / "model_weights.json"
PRIORS_OUTPUT_PATH = ROOT / "solution" / "strategy_priors.json"
HISTORICAL_DIR = ROOT / "data" / "historical_races"

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
COMPOUND_INDEX = {compound: index for index, compound in enumerate(COMPOUNDS)}
TRACK_INDEX = {
    "Bahrain": 0,
    "COTA": 1,
    "Monaco": 2,
    "Monza": 3,
    "Silverstone": 4,
    "Spa": 5,
    "Suzuka": 6,
}
MAX_AGE = 70
ENSEMBLE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
EPOCHS = 22
BLEND_WEIGHTS = [
    1.0,
    0.0007888366852607138,
    0.10051878031370937,
    0.14369256096526134,
    0.8330592116638168,
    0.01275848687114679,
]


def build_features(race: dict, strategy: dict):
    race_config = race["race_config"]
    features = np.zeros(len(COMPOUNDS) * MAX_AGE + 15, dtype=np.float32)

    current_tire = strategy["starting_tire"]
    lap_start = 1
    for stop in strategy["pit_stops"]:
        stint_laps = stop["lap"] - lap_start + 1
        offset = COMPOUND_INDEX[current_tire] * MAX_AGE
        features[offset : offset + stint_laps] += 1.0
        current_tire = stop["to_tire"]
        lap_start = stop["lap"] + 1

    stint_laps = race_config["total_laps"] - lap_start + 1
    offset = COMPOUND_INDEX[current_tire] * MAX_AGE
    features[offset : offset + stint_laps] += 1.0

    tail = len(COMPOUNDS) * MAX_AGE
    features[tail + 0] = race_config["total_laps"]
    features[tail + 1] = race_config["base_lap_time"]
    features[tail + 2] = race_config["pit_lane_time"]
    features[tail + 3] = race_config["track_temp"]
    features[tail + 4] = len(strategy["pit_stops"])
    features[tail + 5 + TRACK_INDEX[race_config["track"]]] = 1.0

    summary_tail = tail + 5 + len(TRACK_INDEX)
    for compound, compound_index in COMPOUND_INDEX.items():
        start = compound_index * MAX_AGE
        end = start + MAX_AGE
        features[summary_tail + compound_index] = features[start:end].sum()
    return features


def build_strategy_signatures(race_config: dict, strategy: dict):
    laps = race_config["total_laps"]
    stop_exact = tuple((round(p["lap"] / laps, 3), p["to_tire"]) for p in strategy["pit_stops"])
    stop_loose = tuple((round(p["lap"] / laps, 2), p["to_tire"]) for p in strategy["pit_stops"])

    stints = []
    current = strategy["starting_tire"]
    lap_start = 1
    for stop in strategy["pit_stops"]:
        stints.append((current, round((stop["lap"] - lap_start + 1) / laps, 2)))
        current = stop["to_tire"]
        lap_start = stop["lap"] + 1
    stints.append((current, round((laps - lap_start + 1) / laps, 2)))

    if strategy["pit_stops"]:
        avg_stop_position = round(
            sum(p["lap"] for p in strategy["pit_stops"]) / len(strategy["pit_stops"]) / laps, 2
        )
    else:
        avg_stop_position = 0.0

    return {
        "k1": (race_config["track"], race_config["track_temp"], strategy["starting_tire"], len(strategy["pit_stops"]), stop_exact),
        "k2": (
            race_config["track"],
            (race_config["track_temp"] // 3) * 3,
            strategy["starting_tire"],
            len(strategy["pit_stops"]),
            stop_loose,
        ),
        "k3": (race_config["track"], strategy["starting_tire"], len(strategy["pit_stops"]), tuple(stints)),
        "k4": (strategy["starting_tire"], len(strategy["pit_stops"]), stop_loose),
        "k5": (race_config["track"], strategy["starting_tire"], len(strategy["pit_stops"]), avg_stop_position),
    }


def load_training_tensor():
    races = []
    for path in sorted(HISTORICAL_DIR.glob("*.json")):
        races.extend(json.loads(path.read_text(encoding="utf-8")))

    prior_agg = {name: [defaultdict(float), defaultdict(int)] for name in ["k1", "k2", "k3", "k4", "k5"]}
    features = []
    ranks = []
    for race in races:
        rank_map = {driver_id: index for index, driver_id in enumerate(race["finishing_positions"])}
        race_config = race["race_config"]
        race_rows = []
        race_ranks = []
        for position in sorted(race["strategies"]):
            strategy = race["strategies"][position]
            race_rows.append(build_features(race, strategy))
            finish_rank = rank_map[strategy["driver_id"]]
            race_ranks.append(finish_rank)
            signatures = build_strategy_signatures(race_config, strategy)
            for name, signature in signatures.items():
                prior_agg[name][0][signature] += finish_rank + 1
                prior_agg[name][1][signature] += 1
        features.append(race_rows)
        ranks.append(race_ranks)

    x_train = torch.tensor(np.array(features)).float()
    y_train = torch.tensor(np.array(ranks))
    mean = x_train.mean(dim=(0, 1), keepdim=True)
    std = x_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std

    priors = {}
    for name, (sums, counts) in prior_agg.items():
        table = {}
        for key, total in sums.items():
            table[json.dumps(key)] = total / counts[key]
        priors[name] = table
    priors_payload = {
        "blend_weights": BLEND_WEIGHTS,
        "default_prior": 10.5,
        "priors": priors,
    }
    return x_train, y_train, mean, std, priors_payload


class RankNet(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


def pairwise_loss(scores, ranks):
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    target = torch.sign(ranks.unsqueeze(1) - ranks.unsqueeze(2)).float()
    mask = target != 0
    return torch.nn.functional.softplus(diff * target)[mask].mean()


def train_one_model(x_train, y_train, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = RankNet(x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(EPOCHS):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            scores = model(batch_x.reshape(-1, batch_x.shape[-1])).reshape(batch_x.shape[0], batch_x.shape[1])
            loss = pairwise_loss(scores, batch_y)
            loss.backward()
            optimizer.step()

    return {key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()}


def main():
    x_train, y_train, mean, std, priors_payload = load_training_tensor()
    models = [train_one_model(x_train, y_train, seed) for seed in ENSEMBLE_SEEDS]

    payload = {
        "feature_dim": int(x_train.shape[-1]),
        "mean": mean.numpy().reshape(-1).tolist(),
        "std": std.numpy().reshape(-1).tolist(),
        "models": models,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "epochs": EPOCHS,
    }
    OUTPUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    PRIORS_OUTPUT_PATH.write_text(json.dumps(priors_payload), encoding="utf-8")


if __name__ == "__main__":
    main()
