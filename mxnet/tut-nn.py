# https://gluon-crash-course.mxnet.io/nn.html

from mxnet import nd
from mxnet.gluon import nn


layer = nn.Dense(2)
print(layer)

layer.initialize()

x = nd.random.uniform(-1,1,(3,4))
print(layer(x))


print(layer.weight.data())

# chain layers into a neural network

net = nn.Sequential()

with net.name_scope():
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))

print(net)


net.initialize()
x = nd.random.uniform(shape=(4,1,28,28))
y = net(x)
print(y.shape)

print(net[0].weight.data().shape, net[5].bias.data().shape)


# create a neural network flexibly

class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        super(MixMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.blk = nn.Sequential()
            self.blk.add(
                nn.Dense(3, activation='relu'),
                nn.Dense(4, activation='relu'))
            self.dense = nn.Dense(5)

    def forward(self, x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)


net = MixMLP()
print(net)

net.initialize()
x = nd.random.uniform(shape=(2,2))
print(net(x))

print(net.blk[1].weight.data())
