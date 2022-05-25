import numpy as np


class CosineSimilarity:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def norm_2_vector(vector):
        return np.linalg.norm(vector)
    
    @staticmethod
    def norm_2_matrix(matrix):
        matrix = np.array(matrix)
        matrix = matrix.T
        return np.linalg.norm(matrix, axis=0)
    
    def cos_similarity(self, vec_a, vec_b):
        a_norm = self.norm_2_vector(vec_a)
        b_norm = self.norm_2_vector(vec_b)
        cos = np.dot(vec_a, vec_b) / (a_norm*b_norm)
        return cos

    def cos_similarity_matrix_vec_mat(self, vec_a, mat_b):
        mat_b = np.array([np.array(i) for i in mat_b]) # to 2-dim np.array
        mat_b_t = mat_b.T # transpose
        dot_product = np.dot(vec_a, mat_b_t) # (1, embed_dim) * (embed_dim, batch_size) -> (1, batch_size)
        vec_a_2_norm_2 = self.norm_2_vector(vec_a) # number
        mat_b_2_norm_2 = self.norm_2_matrix(mat_b) # (1, batch_size)
        cos_sim = dot_product / (vec_a_2_norm_2*mat_b_2_norm_2) # (1, batch_size)
        return cos_sim

    def cos_similarity_matrix(self, mat_a, mat_b):
        if len(mat_a)==0 or len(mat_b)==0:
            cos_sim = [[]*len(mat_b)]*len(mat_a)
            return cos_sim
        mat_b = np.array([np.array(i) for i in mat_b]) # to 2-dim np.array
        mat_b_t = mat_b.T # transpose
        dot_product = np.dot(mat_a, mat_b_t) # (batch_size1, embed_dim) * (embed_dim, batch_size2) -> (batch_size1, batch_size2)
        mat_a_2_norm_2 = self.norm_2_matrix(mat_a) # (batch_size1, 1)
        mat_b_2_norm_2 = self.norm_2_matrix(mat_b) # (1, batch_size2)
        mat_a_2_norm_2_mat = np.array([mat_a_2_norm_2]*len(mat_b_2_norm_2)).T # (batch_size1, batch_size2)
        mat_b_2_norm_2_diag = np.diagflat(mat_b_2_norm_2) # (batch_size2, batch_size2) 
        cos_sim = dot_product / np.dot(mat_a_2_norm_2_mat, mat_b_2_norm_2_diag) # (batch_size1, batch_size2)

        return cos_sim