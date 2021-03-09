import numpy as np
from numpy.linalg import eig, inv

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(A)
value, vectors = eig(A)
print('Eigenvalue of matrix: ',value)
print('Eigenvectors of matrix: ', vectors)
print('A*V =lambda*V')
print(A.dot(vectors[:, 0]))
print(value[0]*vectors[:,0])

print('Consructing the original matrix:')
print()
Q= vectors
R = inv(Q)
l= np.diag(value)
print(Q)
print(R)
print(l)

B =Q.dot(l).dot(R)
print(B)