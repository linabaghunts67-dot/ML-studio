from dataclasses import dataclass
import pandas as pd


@dataclass
class ScenarioConfig:
    id: str
    name: str
    path: str
    target: str
    task_type: str


SCENARIOS = {
    "price": ScenarioConfig(
        id="price",
        name="Price Prediction (Phones)",
        path="data/raw/mobile_price_raw.csv",
        target="price",
        task_type="regression",
    ),
    "default": ScenarioConfig(
        id="default",
        name="Credit Default Risk",
        path="data/raw/home_credit_raw.csv",
        target="TARGET",
        task_type="classification",
    ),
    "churn": ScenarioConfig(
        id="churn",
        name="Customer Churn",
        path="data/raw/telco_churn_raw.csv",
        target="Churn",
        task_type="classification",
    ),
}


def get_scenario_config(scenario_id: str) -> ScenarioConfig:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")
    return SCENARIOS[scenario_id]


def load_data(scenario_id: str) -> pd.DataFrame:
    cfg = get_scenario_config(scenario_id)
    return pd.read_csv(cfg.path)
