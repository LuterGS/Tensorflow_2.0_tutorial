import tensorflow as tf

test = tf.random.normal([3,2], 0, 10)
test2 = test[0]

print(test)
print(test2)