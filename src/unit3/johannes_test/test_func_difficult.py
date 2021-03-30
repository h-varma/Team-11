import pytest
import numpy as np
import func_difficult


# python3 -m pytest test_function1.py -v
# python3 -m pytest -m circles
@pytest.mark.circles  # Markers can be used to categorize tests
def test_circumference():
    """Test different outcomes of the calculation of the circumference
    """
    assert func_difficult.circumference_circle(1) == 2 * np.pi, "returns pi *2"
    assert func_difficult.circumference_circle(0) == 0, "is 0"
    assert func_difficult.circumference_circle(10) == 2 * np.pi * 10


# @pytest.mark.skip(reason="My reason to skip this test")
def test_values():
    """check negative values
    """
    with pytest.raises(ValueError):
        func_difficult.circumference_circle(-42)
        func_difficult.circumference_circle(-1000000000)


# helper function for the fixtures
def create_test_array(dim1, dim2):
    arr = np.ones((dim1, dim2), dtype=float)
    arr[1, :] = arr[1, :] * 2
    return arr


# create input fixture
@pytest.fixture
def test_L2norm_input():
    test_array = create_test_array(2, 2)
    return test_array


# create output fixture
@pytest.fixture
def test_L2norm_output():
    test_array_output = np.zeros(2)  # we have vector with zeros, YES!!!!!!!
    return test_array_output


# finally test the euclidian distance of the arrays
def test_euclidean_array(test_L2norm_input, test_L2norm_output):
    test_array = func_difficult.euclidean_distance([0, 0], [0, 1], test_L2norm_input)
    assert np.array_equal(test_L2norm_output, test_array)


if __name__ == "__main__":  # checks if the running script (__name__) is the main script
    pytest.main([__file__])  # executing pytest in order to run the file
