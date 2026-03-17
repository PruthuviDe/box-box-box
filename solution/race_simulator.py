#!/usr/bin/env python3
"""Final submission entry point for Box Box Box race prediction."""
import json
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
MAX_AGE = 70

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


def load_model():
    with MODEL_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "models" in payload:
        raw_models = payload["models"]
    else:
        raw_models = [payload["state_dict"]]

    models = []
    for state in raw_models:
        models.append(
            {
                "w1": state["net.0.weight"],
                "b1": state["net.0.bias"],
                "w2": state["net.2.weight"],
                "b2": state["net.2.bias"],
                "w3": state["net.4.weight"][0],
                "b3": state["net.4.bias"][0],
            }
        )

    return {
        "mean": payload["mean"],
        "std": payload["std"],
        "models": models,
    }


MODEL = load_model() if MODEL_PATH.exists() else None
PRIORS = None
if PRIORS_PATH.exists():
    with PRIORS_PATH.open("r", encoding="utf-8") as handle:
        PRIORS = json.load(handle)


def strategy_signatures(race_config: dict, strategy: dict):
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


def prior_score(race_config: dict, strategy: dict):
    if PRIORS is None:
        return 10.5

    signatures = strategy_signatures(race_config, strategy)
    default_prior = PRIORS.get("default_prior", 10.5)
    prior_tables = PRIORS["priors"]
    blend_weights = PRIORS["blend_weights"]

    weighted = 0.0
    for idx, key in enumerate(("k1", "k2", "k3", "k4", "k5"), start=1):
        table = prior_tables[key]
        signature = json.dumps(signatures[key])
        weighted += blend_weights[idx] * table.get(signature, default_prior)
    return weighted


def build_features(test_case: dict, strategy: dict):
    race_config = test_case["race_config"]
    features = [0.0] * (len(COMPOUNDS) * MAX_AGE + 15)

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

    tail = len(COMPOUNDS) * MAX_AGE
    features[tail + 0] = float(race_config["total_laps"])
    features[tail + 1] = float(race_config["base_lap_time"])
    features[tail + 2] = float(race_config["pit_lane_time"])
    features[tail + 3] = float(race_config["track_temp"])
    features[tail + 4] = float(len(strategy["pit_stops"]))
    features[tail + 5 + TRACK_INDEX[race_config["track"]]] = 1.0

    summary_tail = tail + 5 + len(TRACK_INDEX)
    for compound, compound_index in COMPOUND_INDEX.items():
        start = compound_index * MAX_AGE
        end = start + MAX_AGE
        features[summary_tail + compound_index] = float(sum(features[start:end]))
    return features


def relu_layer(inputs, weights, bias):
    outputs = []
    for row, row_bias in zip(weights, bias):
        total = row_bias
        for value, weight in zip(inputs, row):
            total += value * weight
        outputs.append(total if total > 0.0 else 0.0)
    return outputs


def score_one_model(normalized, model):
    hidden1 = relu_layer(normalized, model["w1"], model["b1"])
    hidden2 = relu_layer(hidden1, model["w2"], model["b2"])

    total = model["b3"]
    for value, weight in zip(hidden2, model["w3"]):
        total += value * weight
    return total


def score_strategy(features):
    normalized = []
    for value, mean, std in zip(features, MODEL["mean"], MODEL["std"]):
        normalized.append((value - mean) / std)

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

    scored = []
    for strategy in test_case["strategies"].values():
        nn_score = score_strategy(build_features(test_case, strategy))
        if PRIORS is None:
            score = nn_score
        else:
            score = PRIORS["blend_weights"][0] * nn_score + prior_score(test_case["race_config"], strategy)
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
