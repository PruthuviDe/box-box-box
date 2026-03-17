#!/usr/bin/env python3
"""Final submission entry point for Box Box Box race prediction.
Supports both v1 (3-layer MLP) and v2 (BN+Dropout 4-layer MLP) model formats.
"""
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "solution" / "model_weights.json"
PRIORS_PATH = ROOT / "solution" / "strategy_priors.json"

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
N_COMPOUNDS = len(COMPOUNDS)
N_TRACKS = len(TRACK_INDEX)
MAX_AGE = 70
MAX_STINTS = 4

COMPOUND_PACE = {
    "SOFT": -0.65,
    "MEDIUM": 0.0,
    "HARD": 0.55,
}
COMPOUND_DEG = {
    "SOFT": 0.055,
    "MEDIUM": 0.032,
    "HARD": 0.018,
}

# ── detect model version once at load time ──────────────────────────────────
_MODEL_VERSION = 1  # updated after load_model() inspects the payload


def _extract_stints(race_config: dict, strategy: dict):
    total_laps = race_config["total_laps"]
    stints = []
    current = strategy["starting_tire"]
    lap_start = 1
    for stop in strategy["pit_stops"]:
        lap_end = stop["lap"]
        stints.append((current, lap_end - lap_start + 1, lap_start, lap_end))
        current = stop["to_tire"]
        lap_start = lap_end + 1
    stints.append((current, total_laps - lap_start + 1, lap_start, total_laps))
    return stints


def load_model():
    global _MODEL_VERSION
    with MODEL_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    version = payload.get("version", 1)
    _MODEL_VERSION = version

    raw_models = payload.get("models", [payload.get("state_dict")])

    if version == 2:
        # v2: 512-BN-ReLU-Drop-256-BN-ReLU-Drop-128-ReLU-1
        # BatchNorm keys: net.1.weight/bias/running_mean/running_var/num_batches_tracked
        #                 net.5.weight/bias/running_mean/running_var/num_batches_tracked
        models = []
        for state in raw_models:
            models.append({
                "w1": state["net.0.weight"], "b1": state["net.0.bias"],
                "bn1_w": state["net.1.weight"], "bn1_b": state["net.1.bias"],
                "bn1_rm": state["net.1.running_mean"], "bn1_rv": state["net.1.running_var"],
                "w2": state["net.4.weight"], "b2": state["net.4.bias"],
                "bn2_w": state["net.5.weight"], "bn2_b": state["net.5.bias"],
                "bn2_rm": state["net.5.running_mean"], "bn2_rv": state["net.5.running_var"],
                "w3": state["net.8.weight"], "b3": state["net.8.bias"],
                "w4": state["net.10.weight"][0], "b4": state["net.10.bias"][0],
            })
    else:
        # v1: 256-ReLU-128-ReLU-1
        models = []
        for state in raw_models:
            models.append({
                "w1": state["net.0.weight"], "b1": state["net.0.bias"],
                "w2": state["net.2.weight"], "b2": state["net.2.bias"],
                "w3": state["net.4.weight"][0], "b3": state["net.4.bias"][0],
            })

    return {
        "version": version,
        "mean": payload["mean"],
        "std": payload["std"],
        "models": models,
    }


MODEL = load_model() if MODEL_PATH.exists() else None
PRIORS = None
if PRIORS_PATH.exists():
    with PRIORS_PATH.open("r", encoding="utf-8") as handle:
        PRIORS = json.load(handle)


# ── strategy signatures ──────────────────────────────────────────────────────

def strategy_signatures(race_config: dict, strategy: dict):
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
    transition = tuple(c for c, _, _, _ in stints)
    temp_bucket = (race_config["track_temp"] // 3) * 3

    return {
        "k1": (race_config["track"], race_config["track_temp"], strategy["starting_tire"], len(strategy["pit_stops"]), stop_exact),
        "k2": (race_config["track"], temp_bucket, strategy["starting_tire"], len(strategy["pit_stops"]), stop_loose),
        "k3": (race_config["track"], strategy["starting_tire"], len(strategy["pit_stops"]), stint_tuples),
        "k4": (strategy["starting_tire"], len(strategy["pit_stops"]), stop_loose),
        "k5": (race_config["track"], strategy["starting_tire"], len(strategy["pit_stops"]), avg_stop),
        "k6": (race_config["track"], race_config["track_temp"], tuple(stint_order_exact)),
        "k7": (race_config["track"], transition),
    }


def prior_score(race_config: dict, strategy: dict, blend_weights_override=None):
    if PRIORS is None:
        return 10.5

    sigs = strategy_signatures(race_config, strategy)
    default_prior = PRIORS.get("default_prior", 10.5)
    prior_tables = PRIORS["priors"]
    blend_weights = blend_weights_override if blend_weights_override is not None else PRIORS["blend_weights"]

    # Support both v1 (k1-k5) and v2 (k1-k7) prior tables
    all_keys = ("k1", "k2", "k3", "k4", "k5", "k6", "k7")
    weighted = 0.0
    for idx, key in enumerate(all_keys, start=1):
        if key not in prior_tables or idx >= len(blend_weights):
            break
        table = prior_tables[key]
        weighted += blend_weights[idx] * table.get(json.dumps(sigs[key]), default_prior)
    return weighted


# ── feature builders ─────────────────────────────────────────────────────────

def _build_features_v2(race_config: dict, strategy: dict):
    total_laps = race_config["total_laps"]
    track_temp = race_config["track_temp"]
    n_stops = len(strategy["pit_stops"])
    stints = _extract_stints(race_config, strategy)

    DIM = (N_COMPOUNDS * MAX_AGE
           + MAX_STINTS * (N_COMPOUNDS + 3)
           + N_COMPOUNDS * 4
           + N_TRACKS + 5 + N_COMPOUNDS)
    features = [0.0] * DIM

    # compound × age bins
    for compound, stint_laps, lap_start, _ in stints:
        offset = COMPOUND_INDEX[compound] * MAX_AGE
        for age in range(stint_laps):
            features[offset + age] += 1.0

    # per-stint block
    base = N_COMPOUNDS * MAX_AGE
    for si, (compound, stint_laps, lap_start, _) in enumerate(stints):
        if si >= MAX_STINTS:
            break
        block = base + si * (N_COMPOUNDS + 3)
        features[block + COMPOUND_INDEX[compound]] = 1.0
        features[block + N_COMPOUNDS + 0] = stint_laps / total_laps
        features[block + N_COMPOUNDS + 1] = lap_start / total_laps
        features[block + N_COMPOUNDS + 2] = si / MAX_STINTS

    # compound × temp-bucket
    temp_bucket = min(int(track_temp // 5), 3)
    cross_base = N_COMPOUNDS * MAX_AGE + MAX_STINTS * (N_COMPOUNDS + 3)
    for ci in range(N_COMPOUNDS):
        features[cross_base + ci * 4 + temp_bucket] = 1.0

    # track one-hot
    track_base = cross_base + N_COMPOUNDS * 4
    features[track_base + TRACK_INDEX[race_config["track"]]] = 1.0

    # scalars
    scalar_base = track_base + N_TRACKS
    features[scalar_base + 0] = total_laps / 70.0
    features[scalar_base + 1] = race_config["base_lap_time"] / 100.0
    features[scalar_base + 2] = race_config["pit_lane_time"] / 30.0
    features[scalar_base + 3] = track_temp / 40.0
    features[scalar_base + 4] = n_stops / 3.0

    # total laps per compound
    compound_base = scalar_base + 5
    for ci in range(N_COMPOUNDS):
        start = ci * MAX_AGE
        features[compound_base + ci] = sum(features[start: start + MAX_AGE]) / total_laps

    return features


def build_features(test_case: dict, strategy: dict):
    """Dispatch to v1 or v2 feature builder based on loaded model version."""
    race_config = test_case["race_config"]
    if _MODEL_VERSION == 2:
        return _build_features_v2(race_config, strategy)

    # v1 legacy feature builder
    features = [0.0] * (N_COMPOUNDS * MAX_AGE + 15)
    current_tire = strategy["starting_tire"]
    lap_start = 1
    for stop in strategy["pit_stops"]:
        stint_laps = stop["lap"] - lap_start + 1
        offset = COMPOUND_INDEX[current_tire] * MAX_AGE
        for age in range(stint_laps):
            features[offset + age] += 1.0
        current_tire = stop["to_tire"]
        lap_start = stop["lap"] + 1
    stint_laps = race_config["total_laps"] - lap_start + 1
    offset = COMPOUND_INDEX[current_tire] * MAX_AGE
    for age in range(stint_laps):
        features[offset + age] += 1.0
    tail = N_COMPOUNDS * MAX_AGE
    features[tail + 0] = float(race_config["total_laps"])
    features[tail + 1] = float(race_config["base_lap_time"])
    features[tail + 2] = float(race_config["pit_lane_time"])
    features[tail + 3] = float(race_config["track_temp"])
    features[tail + 4] = float(len(strategy["pit_stops"]))
    features[tail + 5 + TRACK_INDEX[race_config["track"]]] = 1.0
    summary_tail = tail + 5 + N_TRACKS
    for ci, compound in enumerate(COMPOUNDS):
        start = ci * MAX_AGE
        features[summary_tail + ci] = float(sum(features[start: start + MAX_AGE]))
    return features


# ── inference helpers ─────────────────────────────────────────────────────────

def _relu(x):
    return x if x > 0.0 else 0.0


def _bn_layer(inputs, rm, rv, w, b, eps=1e-5):
    """Batch-norm in eval mode (uses running stats)."""
    return [
        w[i] * (inputs[i] - rm[i]) / math.sqrt(rv[i] + eps) + b[i]
        for i in range(len(inputs))
    ]


def relu_layer(inputs, weights, bias):
    outputs = []
    for row, row_bias in zip(weights, bias):
        total = row_bias
        for value, weight in zip(inputs, row):
            total += value * weight
        outputs.append(_relu(total))
    return outputs


def _linear(inputs, weights, bias):
    """Single linear layer (no activation), returns list."""
    outputs = []
    for row, row_bias in zip(weights, bias):
        total = row_bias
        for value, weight in zip(inputs, row):
            total += value * weight
        outputs.append(total)
    return outputs


def _score_one_model_v2(normalized, model):
    h = relu_layer(normalized, model["w1"], model["b1"])
    h = _bn_layer(h, model["bn1_rm"], model["bn1_rv"], model["bn1_w"], model["bn1_b"])
    h = [_relu(x) for x in h]
    # dropout is identity at inference time
    h = relu_layer(h, model["w2"], model["b2"])
    h = _bn_layer(h, model["bn2_rm"], model["bn2_rv"], model["bn2_w"], model["bn2_b"])
    h = [_relu(x) for x in h]
    h = relu_layer(h, model["w3"], model["b3"])
    total = model["b4"]
    for value, weight in zip(h, model["w4"]):
        total += value * weight
    return total


def score_one_model(normalized, model):
    if _MODEL_VERSION == 2:
        return _score_one_model_v2(normalized, model)
    # v1
    hidden1 = relu_layer(normalized, model["w1"], model["b1"])
    hidden2 = relu_layer(hidden1, model["w2"], model["b2"])
    total = model["b3"]
    for value, weight in zip(hidden2, model["w3"]):
        total += value * weight
    return total


def score_strategy(features):
    normalized = [
        (v - m) / s
        for v, m, s in zip(features, MODEL["mean"], MODEL["std"])
    ]
    total = 0.0
    for model in MODEL["models"]:
        total += score_one_model(normalized, model)
    return total


def stint_cost(compound: str, laps: int, track_temp: int) -> float:
    base = COMPOUND_PACE[compound] * laps
    temp_factor = 1.0 + (track_temp - 30) * 0.02
    degradation = COMPOUND_DEG[compound] * temp_factor
    return base + degradation * laps * (laps + 1) / 2


def heuristic_positions(test_case: dict):
    race_config = test_case["race_config"]
    scored = []
    for strategy in test_case["strategies"].values():
        total = 0.0
        current_tire = strategy["starting_tire"]
        lap_start = 1

        for stop in strategy["pit_stops"]:
            lap_end = stop["lap"]
            total += stint_cost(current_tire, lap_end - lap_start + 1, race_config["track_temp"])
            total += race_config["pit_lane_time"]
            current_tire = stop["to_tire"]
            lap_start = lap_end + 1

        if lap_start <= race_config["total_laps"]:
            total += stint_cost(
                current_tire,
                race_config["total_laps"] - lap_start + 1,
                race_config["track_temp"],
            )

        scored.append((total, int(strategy["driver_id"][1:]), strategy["driver_id"]))

    scored.sort()
    return [driver_id for _, _, driver_id in scored]


def predict_positions(test_case: dict):
    if MODEL is None:
        return heuristic_positions(test_case)

    race_track = test_case["race_config"]["track"]
    race_temp_bucket = (test_case["race_config"]["track_temp"] // 3) * 3
    default_blend = PRIORS.get("blend_weights") if PRIORS else None
    track_overrides = PRIORS.get("track_blend_overrides", {}) if PRIORS else {}
    track_temp_overrides = PRIORS.get("track_temp_blend_overrides", {}) if PRIORS else {}
    track_temp_key = f"{race_track}|{race_temp_bucket}"
    blend_weights = track_temp_overrides.get(track_temp_key, track_overrides.get(race_track, default_blend))
    grid_tiebreak = PRIORS.get("grid_tiebreak_weight", 0.0) if PRIORS else 0.0
    group_tie_overrides = PRIORS.get("track_temp_grid_tiebreak_overrides", {}) if PRIORS else {}
    grid_tiebreak = group_tie_overrides.get(track_temp_key, grid_tiebreak)

    scored = []
    for position_key, strategy in test_case["strategies"].items():
        grid_position = int(position_key.replace("pos", ""))
        nn_score = score_strategy(build_features(test_case, strategy))
        if PRIORS is None or blend_weights is None:
            score = nn_score
        else:
            score = blend_weights[0] * nn_score + prior_score(test_case["race_config"], strategy, blend_weights)
            score += grid_tiebreak * grid_position
        scored.append((score, strategy["driver_id"]))

    scored.sort()
    return [driver_id for _, driver_id in scored]


def main():
    test_case = json.load(sys.stdin)
    output = {
        "race_id": test_case["race_id"],
        "finishing_positions": predict_positions(test_case),
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
