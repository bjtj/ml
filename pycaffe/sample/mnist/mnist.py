#!/usr/bin/env python

import os
import caffe
from caffe import layers as L, params as P, proto, to_proto

root = '.'
train_list = os.path.join(root, 'mnist/train/train.txt')
test_list = os.path.join(root, 'mnist/test/test.txt')
train_proto = os.path.join(root, 'mnist/train.prototxt')
test_proto = os.path.join(root, 'mnist/test.prototxt')
solver_proto = os.path.join(root, 'mnist/solver.prototxt')


def Lenet(img_list, batch_size, include_acc=False):
    data, label = L.ImageData(source=img_list, batch_size=batch_size, ntop=2, root_folder=root,
                              transform_param=dict(scale=0.00390625))
    conv1 = L.Convolution(data, kernel_size=5, stride=1, num_output=20, pad=0, weight_filler=dict(type='xavier'))
    pool1 = L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    conv2 = L.Convolution(pool1, kernel_size=5, stride=1, num_output=50, pad=0, weight_filler=dict(type='xavier'))
    pool2 = L.Pooling(conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    fc3 = L.InnerProduct(pool2, num_output=500, weight_filler=dict(type='xavier'))
    relu3 = L.ReLU(fc3, in_place=True)
    fc4 = L.InnerProduct(relu3, num_output=10, weight_filler=dict(type='xavier'))
    loss = L.SoftmaxWithLoss(fc4, label)

    if include_acc:
        acc = L.Accuracy(fc4, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)


def write_net():
    with open(train_proto, 'w') as f:
        f.write(str(Lenet(train_list, batch_size=64)))

    with open(test_proto, 'w') as f:
        f.write(str(Lenet(test_list, batch_size=100, include_acc=True)))


def gen_solver(solver_file, train_net, test_net):
    s = proto.caffe_pb2.SolverParameter()
    s.train_net = train_net
    s.test_net.append(test_net)
    s.test_interval = 938
    s.test_iter.append(500)
    s.max_iter = 9380
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.lr_policy = 'step'
    s.stepsize = 3000
    s.gamma = 0.1
    s.display = 20
    s.snapshot = 938
    s.snapshot_prefix = os.path.join(root, 'mnist/lenet')
    s.type = 'SGD'
    s.solver_mode = proto.caffe_pb2.SolverParameter.GPU
    with open(solver_file, 'w') as f:
        f.write(str(s))

def train(solver_proto):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)
    solver.solve()


if __name__ == '__main__':
    write_net()
    gen_solver(solver_proto, train_proto, test_proto)
    train(solver_proto)
