# author Johannes Fürle, 25.3.21
# task unit3, test your own function with pytest
import numpy as np

def circumference_circle(radius:float):
    """Calculate for a given radius the circumference of a circle

    :param radius: radius of circle
    :type radius: float in cm
    """
    if radius < 0.:
        raise ValueError("The radius must be >=0")
    circumference = 2*np.pi*radius
    print("the circumference is C = %fcm"%circumference)
    return circumference