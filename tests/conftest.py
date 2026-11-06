import pandas as pd
import pytest


@pytest.fixture
def benchmark():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "region": ["north", "south", "east", "west"],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "units": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def stored():
    return pd.DataFrame(
        {
            "ID": ["1", "2", "3", "5"],
            "Region": ["north", "south", "east", "north"],
            "Amount": [100, 200.5, 300, 500],
            "Units": [10.4, 20, 30, 50],
            "loaded_at": "2026-09-29",
        }
    )
