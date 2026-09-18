import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_processing"))
import time
import pandas as pd
import numpy as np
from processing_workflow import ProcessingWorkflow


def test_pipeline_handles_500_messy_rows_without_data_loss():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "Company Name": ["Acme", "Beta", "Gamma", "Delta", "Epsilon"] * (n // 5),
        "Revenue": np.random.choice([100.0, 200.0, None, "bad_value", 500.0], n),
        "Region": ["South", "north", "EAST", "west", "North"] * (n // 5),
        "Report Date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
    })
    wf = ProcessingWorkflow()
    result = wf.run_with_profile(df, "company_financials")
    assert len(result) == n
    region_cols = [c for c in result.columns if c.startswith("region_")]
    assert sorted(region_cols) == ["region_east", "region_north", "region_south", "region_west"]


def test_pipeline_processes_5000_rows_under_one_second():
    n = 5000
    df = pd.DataFrame({
        "Company Name": ["Company_" + str(i % 100) for i in range(n)],
        "Revenue": np.random.uniform(50, 1000, n),
        "Region": np.random.choice(["South", "north", "EAST", "west"], n),
        "Report Date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
    })
    wf = ProcessingWorkflow()
    start = time.time()
    result = wf.run_with_profile(df, "company_financials")
    elapsed = time.time() - start
    assert len(result) == n
    assert elapsed < 1.0