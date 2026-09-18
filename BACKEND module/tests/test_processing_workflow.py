import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_processing"))
import pandas as pd
import pytest
from processing_workflow import ProcessingWorkflow, ProcessingWorkflowError


def make_sample():
    return pd.DataFrame({
        "Company Name": ["Acme", "Acme", "Beta"],
        "Revenue": [100.0, 100.0, 250.0],
        "Region": ["South", "South", "north"],
        "Report Date": ["2026-01-05", "2026-01-05", "2026-02-10"],
    })


def test_run_deduplicates_and_transforms():
    wf = ProcessingWorkflow()
    result = wf.run(make_sample(), normalize_cols=["revenue"], categorical_cols=["region"])
    assert len(result) == 2
    assert "region_north" in result.columns
    assert "region_south" in result.columns


def test_run_with_profile_loads_settings_from_config():
    wf = ProcessingWorkflow()
    result = wf.run_with_profile(make_sample(), "company_financials")
    assert len(result) == 2
    assert "region_north" in result.columns
    
    
def test_run_with_profile_unknown_profile_raises_error():
    wf = ProcessingWorkflow()
    with pytest.raises(ProcessingWorkflowError):
        wf.run_with_profile(pd.DataFrame({"a": [1]}), "nonexistent_profile")


def test_run_applies_memory_optimization_by_default():
    wf = ProcessingWorkflow()
    result = wf.run(make_sample(), categorical_cols=["region"])
    assert result["region_north"].dtype.name in ("int64", "int32", "int8")


def test_run_handles_non_numeric_values_in_normalize_cols():
    messy = pd.DataFrame({
        "Company Name": ["Acme", None, "Beta", ""],
        "Revenue": [100.0, None, "not_a_number", 250.0],
        "Region": ["South", "North", None, "south"],
    })
    wf = ProcessingWorkflow()
    result = wf.run(messy, normalize_cols=["revenue"], categorical_cols=["region"])
    assert len(result) == 4
    assert result["revenue"].notna().all()
    
def test_run_rejects_none_input():
    wf = ProcessingWorkflow()
    with pytest.raises(ProcessingWorkflowError):
        wf.run(None)


def test_run_rejects_non_dataframe_input():
    wf = ProcessingWorkflow()
    with pytest.raises(ProcessingWorkflowError):
        wf.run("not a dataframe")


def test_run_rejects_empty_dataframe():
    wf = ProcessingWorkflow()
    with pytest.raises(ProcessingWorkflowError):
        wf.run(pd.DataFrame())