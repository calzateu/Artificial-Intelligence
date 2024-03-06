import numpy as np

def euclidean_norm(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1 - vector2)))

def manhattan_norm(vector1, vector2):
    return np.sum(np.abs(vector1 - vector2))

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.sqrt(np.sum(np.square(vector1))) * np.sqrt(np.sum(np.square(vector2)))
    similarity = dot_product / norm_product
    return similarity

def p_norm(vector1, vector2, p):
    return np.power(np.sum(np.power(np.abs(vector1 - vector2), p)), 1/p)