# https://gluon-crash-course.mxnet.io/ndarray.html

from mxnet import nd

print(nd.array(((1,2,3),(5,6,7))))


x = nd.ones((2,3))
print(x)

y = nd.random.uniform(-1, 1, (2,3))
print(y)


x = nd.full((2,3),2.0)
print(x)


print(x.shape, y.size, x.dtype)

# operations

print(x * y)

print(y.exp())

print(nd.dot(x, y.T))

print(y[1,2])

print(y[:, 1:3])

y[:, 1:3] = 2
print(y)

y[1:2, 0:2] = 4
print(y)

# converting between mxnet ndarray and numpy

a = x.asnumpy()
print(type(a), a)

print(nd.array(a))
