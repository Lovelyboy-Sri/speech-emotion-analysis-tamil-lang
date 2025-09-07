import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import tensorflow as tf
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
