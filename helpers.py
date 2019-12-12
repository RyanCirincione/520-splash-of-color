def tokenize(array):
    out = 0
    for element in array:
        out += element
        out *= 100
    return out