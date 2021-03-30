# author Johannes Fürle, 25.3.21
# pytest module usage
import pytest
import numpy as np
import function1 as func1


# python3 -m pytest test_function1.py -v
# python3 -m pytest -m circles
@pytest.mark.circles  # Markers can be used to categorize tests
def test_circumference():
    """Test different outcomes of the calculation of the circumference
    """
    assert func1.circumference_circle(1) == 2 * np.pi, "returns pi *2"
    assert func1.circumference_circle(0) == 0, "is 0"
    assert func1.circumference_circle(10) == 2 * np.pi * 10


@pytest.mark.skip(reason="My reason to skip this test")
def test_values():
    """check negative values
    """
    with pytest.raises(ValueError):
        func1.circumference_circle(-42)
        func1.circumference_circle(-1000000000)


if __name__ == "__main__":  # checks if the running script (__name__) is the main script
    pytest.main([__file__])  # executing pytest in order to run the file
