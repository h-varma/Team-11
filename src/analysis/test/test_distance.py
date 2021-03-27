import pytest
import numpy as np
import sys
sys.path.append("../")
import analysis

sys.path.append("../")

def obj_ar(ar):
    return analysis.ProcessStatistical(ar)

@pytest.fixture
def test_ar():
    test_data = [[1,1],[2,2]]
    return obj_ar(np.array(test_data))

@pytest.mark.statistical
def test_distance(test_ar):
    res = np.array([0, np.sqrt(2)])
    assert np.array_equal(test_ar.distance(
        "raw_data", [0,0], [0,1]),
        res)

if __name__ == "__main__":
    pytest.main([__file__])