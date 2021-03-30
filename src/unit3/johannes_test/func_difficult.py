import numpy as np


def circumference_circle(radius: float):
    """Calculate for a given radius the circumference of a circle

    :param radius: radius of circle
    :type radius: float in cm
    """
    if radius < 0.:
        raise ValueError("The radius must be >=0")
    circumference = 2 * np.pi * radius
    print("the circumference is C = %fcm" % circumference)
    return circumference


def euclidean_distance(list_ref, list_comp, vectors):
    """Calculates the Euclidean distance (L2 norm) between pairs of vectors.

    Args:
        list_ref (integer list): A list with the indices of the reference vectors.
        list_comp (integer list): A list with the indices of the vectors to\
        compare to.
        data (numpy array): The data object.

    Returns:
        numpy array: The Euclidean distance (L2 norm) for comparison vs. reference\
        vectors."""
    distances = np.zeros(len(list_ref))
    for i in range(len(list_ref)):
        distances[i] = np.linalg.norm(vectors[list_comp[i]] - vectors[list_ref[i]])
    return distances
