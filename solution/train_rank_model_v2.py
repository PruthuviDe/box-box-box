#!/usr/bin/env python3
"""
v2 training pipeline:
- Order-aware per-stint features (compound × stint_index × normalised_length)
- Confidence-weighted priors (count-dampened mean finishing rank)
- Larger network with Dropout + BatchNorm
- Pairwise + listwise mixed loss
- Cosine-annealing LR schedule
- More epochs (50)
- 12-seed ensemble
"""
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
COMPOUND_INDEX = {c: i for i, c in enumerate(COMPOUNDS)}
TRACK_INDEX = {
    "Bahrain": 0, "COTA": 1, "Monaco": 2, "Monza": 3,
    "Silverstone": 4, "Spa": 5, "Suzuka": 6,
}
N_COMPOUNDS = len(COMPOUNDS)
N_TRACKS = len(TRACK_INDEX)
MAX_STINTS = 4          # max pit stops = 3 → 4 stints
MAX_AGE = 70            # keep compound×age bins (legacy compat for inference)

ENSEMBLE_SEEDS = tuple(range(16))
EPOCHS = 50
PRIOR_CONF_CAP = 40     # beyond this count, full confidence


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _extract_stints(race_config: dict, strategy: dict):
    """Return list of (compound, laps, lap_start, lap_end) in race order."""
    laps_total = race_config["total_laps"]
    stints = []
    current = strategy["starting_tire"]
    lap_start = 1
    for stop in strategy["pit_stops"]:
        lap_end = stop["lap"]
        stints.append((current, lap_end - lap_start + 1, lap_start, lap_end))
        current = stop["to_tire"]
        lap_start = lap_end + 1
    lap_end = laps_total
    stints.append((current, laps_total - lap_start + 1, lap_start, lap_end))
    return stints


def build_features(race: dict, strategy: dict) -> np.ndarray:
    """
    Feature vector layout (total = 225 + 60 + 12 + 7 = 304 dims):
      [0 : 210]   compound × age bins (legacy, order-agnostic)       210
      [210: 270]  per-stint block: MAX_STINTS × (N_COMPOUNDS + 3)    MAX_STINTS*(3+3)=24 → 4*15=60
      [270: 282]  cross: compound × floor(track_temp/5)              3*4=12
      [282: 289]  track one-hot                                       7
      [289: 294]  scalars: total_laps, base_lap_time, pit_lane_time,  5
                           track_temp, num_stops
      [294: 297]  total laps per compound (normalised by total_laps)  3
      total = 297
    """
    rc = race["race_config"]
    total_laps = rc["total_laps"]
    track_temp = rc["track_temp"]
    n_stops = len(strategy["pit_stops"])
    stints = _extract_stints(rc, strategy)

    DIM = N_COMPOUNDS * MAX_AGE + MAX_STINTS * (N_COMPOUNDS + 3) + N_COMPOUNDS * 4 + N_TRACKS + 5 + N_COMPOUNDS
    features = np.zeros(DIM, dtype=np.float32)

    # --- Compound × age bins (legacy 210 dims) ---
    for stint_idx, (compound, stint_laps, lap_start, _) in enumerate(stints):
        offset = COMPOUND_INDEX[compound] * MAX_AGE
        features[offset: offset + stint_laps] += 1.0

    # --- Per-stint block (60 dims) ---
    base = N_COMPOUNDS * MAX_AGE
    for si, (compound, stint_laps, lap_start, _) in enumerate(stints):
        if si >= MAX_STINTS:
            break
        block = base + si * (N_COMPOUNDS + 3)
        features[block + COMPOUND_INDEX[compound]] = 1.0               # compound one-hot
        features[block + N_COMPOUNDS + 0] = stint_laps / total_laps    # normalised length
        features[block + N_COMPOUNDS + 1] = lap_start / total_laps     # when it starts
        features[block + N_COMPOUNDS + 2] = si / MAX_STINTS            # stint index

    # --- Compound × temp bucket cross (12 dims) ---
    temp_bucket = min(int(track_temp // 5), 3)   # buckets 0-3 (≤17, 18-22, 23-27, 28+)
    cross_base = N_COMPOUNDS * MAX_AGE + MAX_STINTS * (N_COMPOUNDS + 3)
    for ci in range(N_COMPOUNDS):
        features[cross_base + ci * 4 + temp_bucket] = 1.0

    # --- Track one-hot ---
    track_base = cross_base + N_COMPOUNDS * 4
    features[track_base + TRACK_INDEX[rc["track"]]] = 1.0

    # --- Scalars ---
    scalar_base = track_base + N_TRACKS
    features[scalar_base + 0] = total_laps / 70.0
    features[scalar_base + 1] = rc["base_lap_time"] / 100.0
    features[scalar_base + 2] = rc["pit_lane_time"] / 30.0
    features[scalar_base + 3] = track_temp / 40.0
    features[scalar_base + 4] = n_stops / 3.0

    # --- Total laps per compound (norm) ---
    compound_base = scalar_base + 5
    for ci in range(N_COMPOUNDS):
        start = ci * MAX_AGE
        features[compound_base + ci] = features[start: start + MAX_AGE].sum() / total_laps

    return features


FEATURE_DIM = (
    N_COMPOUNDS * MAX_AGE
    + MAX_STINTS * (N_COMPOUNDS + 3)
    + N_COMPOUNDS * 4
    + N_TRACKS
    + 5
    + N_COMPOUNDS
)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy signatures for priors
# ─────────────────────────────────────────────────────────────────────────────

def build_strategy_signatures(race_config: dict, strategy: dict) -> dict:
    laps = race_config["total_laps"]
    stints = _extract_stints(race_config, strategy)
    stint_tuples = tuple((c, round(l / laps, 2)) for c, l, _, _ in stints)
    stint_order_exact = tuple((c, l) for c, l, _, _ in stints)

    stop_exact = tuple((round(p["lap"] / laps, 3), p["to_tire"]) for p in strategy["pit_stops"])
    stop_loose = tuple((round(p["lap"] / laps, 2), p["to_tire"]) for p in strategy["pit_stops"])

    avg_stop = (
        round(sum(p["lap"] for p in strategy["pit_stops"]) / len(strategy["pit_stops"]) / laps, 2)
        if strategy["pit_stops"] else 0.0
    )

    # compound transition sequence e.g. (SOFT→HARD→MEDIUM)
    transition = tuple(c for c, _, _, _ in stints)

    temp_bucket = (race_config["track_temp"] // 3) * 3

    return {
        "k1": (race_config["track"], race_config["track_temp"], strategy["starting_tire"], len(strategy["pit_stops"]), stop_exact),
        "k2": (race_config["track"], temp_bucket, strategy["starting_tire"], len(strategy["pit_stops"]), stop_loose),
        "k3": (race_config["track"], strategy["starting_tire"], len(strategy["pit_stops"]), stint_tuples),
        "k4": (strategy["starting_tire"], len(strategy["pit_stops"]), stop_loose),
        "k5": (race_config["track"], strategy["starting_tire"], len(strategy["pit_stops"]), avg_stop),
        "k6": (race_config["track"], race_config["track_temp"], tuple(stint_order_exact)),   # exact order+length
        "k7": (race_config["track"], transition),                                             # compound sequence only
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_training_tensor():
    races = []
    for path in sorted(HISTORICAL_DIR.glob("*.json")):
        races.extend(json.loads(path.read_text(encoding="utf-8")))
    print(f"Loaded {len(races)} races")

    # prior aggregation: store (sum_rank, count) per signature key
    prior_agg = {name: [defaultdict(float), defaultdict(int)]
                 for name in ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]}

    features_list = []
    ranks_list = []

    for race in races:
        rank_map = {driver_id: idx for idx, driver_id in enumerate(race["finishing_positions"])}
        rc = race["race_config"]
        race_rows = []
        race_ranks = []
        for pos_key in sorted(race["strategies"]):
            strat = race["strategies"][pos_key]
            feat = build_features(race, strat)
            race_rows.append(feat)
            finish_rank = rank_map[strat["driver_id"]]
            race_ranks.append(finish_rank)
            sigs = build_strategy_signatures(rc, strat)
            for name, sig in sigs.items():
                key = json.dumps(sig)
                prior_agg[name][0][key] += finish_rank + 1
                prior_agg[name][1][key] += 1
        features_list.append(race_rows)
        ranks_list.append(race_ranks)

    x_train = torch.tensor(np.array(features_list, dtype=np.float32))
    y_train = torch.tensor(np.array(ranks_list, dtype=np.int64))

    # normalise
    mean = x_train.mean(dim=(0, 1), keepdim=True)
    std = x_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std

    # build confidence-weighted priors
    priors_out = {}
    for name, (sums, counts) in prior_agg.items():
        table = {}
        for key in sums:
            raw_mean = sums[key] / counts[key]
            conf = min(counts[key], PRIOR_CONF_CAP) / PRIOR_CONF_CAP
            # shrink toward global mean (10.5) when low-confidence
            table[key] = conf * raw_mean + (1.0 - conf) * 10.5
        priors_out[name] = table

    # optimise blend weights via grid search on a held-out slice
    blend_weights = _fit_blend_weights(races, priors_out)

    priors_payload = {
        "version": 2,
        "blend_weights": blend_weights,
        "default_prior": 10.5,
        "priors": priors_out,
    }
    return x_train, y_train, mean, std, priors_payload


def _prior_rank_list(race: dict, priors_out: dict, weights: list) -> list:
    """Return predicted finishing order using only priors (for blend tuning)."""
    rc = race["race_config"]
    scored = []
    for pos_key in sorted(race["strategies"]):
        strat = race["strategies"][pos_key]
        sigs = build_strategy_signatures(rc, strat)
        score = 0.0
        for ki, name in enumerate(("k1", "k2", "k3", "k4", "k5", "k6", "k7"), start=1):
            key = json.dumps(sigs[name])
            val = priors_out[name].get(key, 10.5)
            score += weights[ki] * val
        scored.append((score, strat["driver_id"]))
    scored.sort()
    return [d for _, d in scored]


def _fit_blend_weights(races: list, priors_out: dict) -> list:
    """
    Simple grid search over k3 (stint-tuples, highest signal) weight.
    Keep other weights fixed; return best combo on last 3000 races.
    This is fast enough to run in <30 s even on CPU.
    """
    eval_races = races[-3000:]

    # Fixed baseline weights
    base = [1.0, 0.001, 0.10, 0.14, 0.0, 0.012, 0.0, 0.0]  # idx 0=nn(ignored), 1-7=k1-k7

    best_acc = -1
    best_weights = base[:]

    k3_search = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    k6_search = [0.2, 0.5, 0.8, 1.0]
    k7_search = [0.1, 0.3, 0.5]

    for k3w in k3_search:
        for k6w in k6_search:
            for k7w in k7_search:
                w = [base[0], base[1], base[2], k3w, base[4], base[5], k6w, k7w]
                correct = 0
                for race in eval_races:
                    pred = _prior_rank_list(race, priors_out, w)
                    if pred == race["finishing_positions"]:
                        correct += 1
                acc = correct / len(eval_races)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = w[:]

    print(f"Best prior-only acc on last 3000 races: {best_acc:.4f}  weights={best_weights}")
    return best_weights


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class RankNetV2(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pairwise_loss(scores, ranks):
    """Pairwise RankNet softplus loss."""
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    target = torch.sign(ranks.unsqueeze(1).float() - ranks.unsqueeze(2).float())
    mask = target != 0
    return torch.nn.functional.softplus(diff * target)[mask].mean()


def listwise_loss(scores, ranks):
    """
    Approximate listwise loss via negative log-likelihood of the correct
    permutation (ListNet top-1 approximation).
    """
    # scores shape: (batch, n_drivers)  lower score = better rank
    # negate so smallest rank → highest logit
    log_probs = torch.log_softmax(-scores, dim=-1)
    # reward correct rank 0 driver
    best = ranks.argmin(dim=-1)
    return -log_probs[torch.arange(log_probs.size(0)), best].mean()


def train_one_model(x_train, y_train, seed: int, device):
    torch.manual_seed(seed)
    n_races, n_drivers, n_features = x_train.shape

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = RankNetV2(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            flat_x = batch_x.reshape(-1, n_features)
            scores = model(flat_x).reshape(batch_x.shape[0], n_drivers)
            pw = pairwise_loss(scores, batch_y)
            lw = listwise_loss(scores, batch_y)
            loss = pw + 0.3 * lw
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  seed={seed}  epoch={epoch+1}/{EPOCHS}  loss={loss.item():.4f}")

    return {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    x_train, y_train, mean, std, priors_payload = load_training_tensor()
    print(f"Feature dim: {x_train.shape[-1]}  Races: {x_train.shape[0]}")

    models = []
    for seed in ENSEMBLE_SEEDS:
        print(f"\n--- Training seed {seed} ---")
        state = train_one_model(x_train, y_train, seed, device)
        models.append(state)

    payload = {
        "version": 2,
        "feature_dim": int(x_train.shape[-1]),
        "mean": mean.numpy().reshape(-1).tolist(),
        "std": std.numpy().reshape(-1).tolist(),
        "models": models,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "epochs": EPOCHS,
        "arch": "512-256-128 BN+Dropout",
    }
    OUTPUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    PRIORS_OUTPUT_PATH.write_text(json.dumps(priors_payload), encoding="utf-8")
    print(f"\nSaved model → {OUTPUT_PATH}")
    print(f"Saved priors → {PRIORS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
