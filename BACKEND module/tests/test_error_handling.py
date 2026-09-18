import pytest
from error_handling import safe_run, DataSourceError, DataValidationError, ErrorCollector


def test_safe_run_succeeds_after_retries():
    calls = {"count": 0}

    @safe_run(retries=2, delay_seconds=0, exceptions=(DataSourceError,))
    def flaky():
        calls["count"] += 1
        if calls["count"] <= 2:
            raise DataSourceError("temporary failure")
        return "OK"

    assert flaky() == "OK"
    assert calls["count"] == 3


def test_safe_run_raises_after_exhausting_retries():
    @safe_run(retries=1, delay_seconds=0, exceptions=(DataSourceError,))
    def always_fails():
        raise DataSourceError("permanent failure")

    with pytest.raises(DataSourceError):
        always_fails()


def test_error_collector_tracks_errors():
    collector = ErrorCollector()
    values = [1, -5, "bad", 42]

    for idx, val in enumerate(values):
        try:
            if not isinstance(val, int) or val < 0:
                raise DataValidationError(f"Invalid value: {val!r}")
        except DataValidationError as e:
            collector.add(idx, e)

    assert collector.has_errors()
    assert collector.count() == 2