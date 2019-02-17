from toposort import toposort
import contextlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.contrib.graph_editor import util as graph_util
import time
import sys

sys.setrecursionlimit(10000)
# refers back to current module if we decide to split helpers out
util = sys.modules[__name__]

# getting rid of "WARNING:tensorflow:VARIABLES collection name is deprecated"
setattr(tf.GraphKeys, "VARIABLES", "variables")

def gradients(ys, xs, grad_ys=None, **kwargs):
    if not isinstance(ys, list):
        ys = [ys]
    if not isinstance(xs, list):
        xs = [xs]

    bwd_ops = ge.get_backward_walk_ops([y.op for y in ys],
                                       inclusive=True)

    for index, op in enumerate(bwd_ops):
        tf.logging.info("bwd_ops: [{}] :{}".format(index, op.name))

    # forward ops are all ops that are candidates for recomputation
    fwd_ops = ge.get_forward_walk_ops([x.op for x in xs],
                                      inclusive=True,
                                      within_ops=bwd_ops)
    for index, op in enumerate(fwd_ops):
        tf.logging.info("fwd_ops: [{}] : {}".format(index, op.name))
