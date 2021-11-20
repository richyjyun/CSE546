import numpy as np

# A.11 (a)
A = np.matrix('0,2,4;2,4,2;3,3,1')
Ainv = np.linalg.inv(A)
print('\nAinv=\n',Ainv)

# A.11 (b)
b = np.matrix('-2;-2;-4')
c = np.matrix('1;1;1')
print('\nAinv*b=\n',np.matmul(Ainv,b))
print('\nA*c=\n',np.matmul(A,c))