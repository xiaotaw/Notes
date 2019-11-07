#!/usr/bin/python

# tensorflow=1.8, python=2.7

import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    a = tf.Variable(5.0, name="a")
    b = tf.Variable(6.0, name="b")
    c = tf.multiply(a, b, name="c")
    
    sess.run(tf.global_variables_initializer())

    print(a.eval())
    print(b.eval())
    print(c.eval())

    tf.train.write_graph(sess.graph_def, "./", "graph.pb", as_text=False)
