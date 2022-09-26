import tensorflow as tf
A = tf.data.Dataset.range(128)
A = A.repeat().batch(32)
for a in A.take(2):
    print(a)
