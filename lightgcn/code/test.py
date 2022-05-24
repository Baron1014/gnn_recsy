from scipy.sparse import csr_matrix
import numpy as np

a = np.array([0, 0, 1, 1])
b = np.array([1, 2, 3, 0])

# print(np.ones(4), (a, b))
print(csr_matrix((np.ones(4), (a, b)),
                shape=(2, 4)).toarray())