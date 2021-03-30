#!/usr/bin/env python3
import pytest
import analysis
import sys # noqa F401


@pytest.fixture
def ref_df():
    df = analysis.ReadIO("", "test_df.csv").read_in_df()
    var = df.var(axis=0)
    return df.drop(df.columns[var < 1e-5], axis=1)


@pytest.fixture
def test_df():
    df = analysis.ReadIO("", "test_df.csv").read_in_df()
    return analysis.Process(df)


@pytest.mark.filter_df
def test_filter_data(test_df, ref_df):
    test_df.filter_data("raw_data")
    assert test_df.data["filtered_data"].equals(ref_df)


if __name__ == "__main__":
    pytest.main([__file__])
