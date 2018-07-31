import TensorFlow

var x = Tensor<Float>([[1, 2], [3, 4]])

for i in 1...5 {
    x += x • x
}

print(x)
