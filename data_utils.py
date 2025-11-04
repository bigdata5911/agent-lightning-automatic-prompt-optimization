"""Data loading utilities for Spider dataset."""

import json
import os
from typing import Any, Dict, List


def load_spider_json(json_path: str) -> List[Dict[str, Any]]:
    """Load Spider JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_spider_dataset(json_path: str, database_dir: str = "databases") -> List[Dict[str, Any]]:
    """Load Spider dataset and filter valid examples."""
    data = load_spider_json(json_path)

    valid_data = []
    for item in data:
        db_path = os.path.join(
            database_dir, item["db_id"], f"{item['db_id']}.sqlite")
        if os.path.exists(db_path):
            valid_data.append(item)

    print(
        f"Loaded {len(valid_data)}/{len(data)} examples with valid databases from {json_path}")
    return valid_data
