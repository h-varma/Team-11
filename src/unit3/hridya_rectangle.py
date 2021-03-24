def rectangle(length, width):
    if length <= 0 or width <= 0:
        raise ValueError("The side length cannot be negative or zero!")
    measures = {"area": length * width}
    print(f'The rectangle has area {format(measures["area"])}')
    return measures
