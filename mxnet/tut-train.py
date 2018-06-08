# https://gluon-crash-course.mxnet.io/train.html

from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import matplotlib.pyplot as plt
from time import time

# get data

mnist_train = datasets.FashionMNIST(train=True)
X, y = mnist_train[0]
print('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)


text_labels = [
    't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

X, y = mnist_train[0:6]
_, figs = plt.subplots(1, X.shape[0], figsize=(15,15))
for f, x, yi in zip(figs, X, y):
    f.imshow(x.reshape((28, 28)).asnumpy())
    ax = f.axes
    ax.set_title(text_labels[int(yi)])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)])

mnist_train = mnist_train.transform_first(transformer)

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

for data, label in train_data:
    print(data.shape, label.shape)
    break

mnist_valid = gluon.data.vision.FashionMNIST(train=True)
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer),
    batch_size=batch_size, num_workers=4)

# define the model

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

net.initialize(init=init.Xavier())


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})

# train

print('train')

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time()
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)

    for data, label in valid_data:
        valid_acc += acc(net(data), label)

    print('Epoch {}: Loss: {:.3f}, Train acc {:.3f}, Test acc {:.3f}, Time {:.1f} sec'.format(
        epoch, train_loss / len(train_data),
        train_acc / len(train_data),
        valid_acc / len(valid_data), time() - tic))

print('save params')
net.save_params('net.params')
