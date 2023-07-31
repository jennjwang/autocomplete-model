import tensorflow as tf

sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
var = tf.Variable(2.5)
cost = lambda: 2 + var ** 2

sgd.minimize(cost, var_list=[var])
print(cost().numpy())
print(var.numpy())

for _ in range(100):
    sgd.minimize(cost, var_list=[var])

print(cost().numpy())
print(var.numpy())
