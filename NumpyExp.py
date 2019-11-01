from numpy import array
from numpy import empty
from numpy import zeros
from numpy import vstack
from numpy import hstack
from numpy.linalg import norm
from numpy import inf
from numpy import tril
from numpy import triu
from numpy import diag
from numpy import identity
from numpy.linalg import inv

l = [1.0, 2.0, 3.0]


a = array(l)
print(a)
print(type(a), a.shape)
print(a.dtype)


a = empty([3,3])
print(a)


a = zeros([3,5])
print(a)


# create first array
a1 = array([1,2,3])
print(a1)
# create second array
a2 = array([4,5,6])
print(a2)
# vertical stack
a3 = vstack((a1, a2))
print(a3)
print(a3.shape)

a4 = hstack((a1, a2))
print(a4)
print(a4.shape)

data = array([
[11, 22],
[33, 44],
[55, 66]])
# index data
print("Data",data[0,0])

data = array([
[11, 22, 33],
[44, 55, 66],
[77, 88, 99]])
# separate data
X, y = data[:, :-1], data[:, -1]
print("x",X)
print("y",y)

print('Rows: %d' % X.shape[0])
print('Cols: %d' % X.shape[1])


data = array([11, 22, 33, 44, 55])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)

# Array Broadcasting

a = array([1, 2, 3])
print(a)
# define scalar
b = 2
print(b)
# broadcast
c = a + b
print("BroadCasting",c)

# define array
A = array([
[1, 2, 3],
[1, 2, 3]])
print(A)
# define scalar
b = 2
print(b)
# broadcast
C = A + b
print("BroadCasting 2D",C)


A = array([
[1, 2, 3],
[1, 2, 3]])
print(A)
# define one-dimensional array
b = array([1, 2, 3])
print(b)
# broadcast
C = A + b
print("BroadCastingCD",C)



a = array([1, 2, 3])
print(a)
# define second vector
b = array([1, 2, 3])
print(b)
# multiply vectors
c = a.dot(b)
print(c)


a = array([1, 2.5, 3])
print(a)
# calculate norm
l1 = norm(a, 1)
print(l1)


# define vector
a = array([1, 2, 3])
print(a)
# calculate norm
l2 = norm(a)
print(l2)

a = array([10, 2, 3])
print(a)
# calculate norm
maxnorm = norm(a, inf)
print(maxnorm)

M = array([
[1, 2, 3],
[1, 2, 3],
[1, 2, 3]])
print(M)
# lower triangular matrix
lower = tril(M)
print(lower)
# upper triangular matrix
upper = triu(M)
print(upper)


M = array([
[1, 2, 3],
[1, 2, 3],
[1, 2, 3]])
print(M)
# extract diagonal vector
d = diag(M)
print(d)
# create diagonal matrix from vector
D = diag(d)
print(D)


# identity matrix
I = identity(3)
print(I)

# orthogonal matrix
Q = array([
[1, 0],
[0, -1]])
print(Q)
# inverse equivalence
V = inv(Q)
print(Q.T)
print(V)
# identity equivalence
I = Q.dot(Q.T)
print(I)