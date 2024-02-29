import numpy as np

def euclidean_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

def manhattan_norm(vector):
    return np.sum(np.abs(vector))

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.sqrt(np.sum(np.square(vector1))) * np.sqrt(np.sum(np.square(vector2)))
    similarity = dot_product / norm_product
    return similarity

def p_norm(vector, p):
    return np.power(np.sum(np.power(np.abs(vector), p)), 1/p)