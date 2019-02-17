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

# save original gradients since tf.gradient could be monkey-patched to point
# to our version
from tensorflow.python.ops import gradients as tf_gradients_lib

tf_gradients = tf_gradients_lib.gradients

MIN_CHECKPOINT_NODE_SIZE = 1024  # use lower value during testing


# specific versions we can use to do process-wide replacement of tf.gradients
def gradients_speed(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='speed', **kwargs)


def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)


def gradients_collection(ys, xs, grad_ys=None, **kwargs):
    return gradients(ys, xs, grad_ys, checkpoints='collection', **kwargs)


def gradients(ys, xs, grad_ys=None, checkpoints='collection', **kwargs):
    print("-------------------------------")
    debug_print("Editing model for OME")
    incpu_count = 0
    #    print("Calling memsaving gradients with", checkpoints)
    if not isinstance(ys, list):
        ys = [ys]
    if not isinstance(xs, list):
        xs = [xs]

    bwd_ops = ge.get_backward_walk_ops([y.op for y in ys],
                                       inclusive=True)

    for index, op in enumerate(bwd_ops):
        debug_print("bwd_ops: [{}] :{}".format(index, op.name), 1)

    # forward ops are all ops that are candidates for recomputation
    fwd_ops = ge.get_forward_walk_ops([x.op for x in xs],
                                      inclusive=True,
                                      within_ops=bwd_ops)
    for index, op in enumerate(fwd_ops):
        debug_print("fwd_ops: [{}] : {}".format(index, op.name), 1)

    # exclude ops with no inputs
    fwd_ops = [op for op in fwd_ops if op.inputs]

    # don't recompute xs, remove variables
    xs_ops = _to_ops(xs)
    fwd_ops = [op for op in fwd_ops if not op in xs_ops]
    fwd_ops = [op for op in fwd_ops if not '/assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/Assign' in op.name]
    fwd_ops = [op for op in fwd_ops if not '/read' in op.name]
    ts_all = ge.filter_ts(fwd_ops, True)  # get the tensors
    ts_all = [t for t in ts_all if '/read' not in t.name]
    ts_all = set(ts_all) - set(xs) - set(ys)

    # construct list of tensors to checkpoint during forward pass, if not
    # given as input
    if type(checkpoints) is not list:
        # remove very small tensors and some weird ops
        def fixdims(t):  # tf.Dimension values are not compatible with int, convert manually
            try:
                return [int(e if e.value is not None else 64) for e in t]
            except:
                return [0]  # unknown shape

        ts_all = [t for t in ts_all if np.prod(fixdims(t.shape)) > MIN_CHECKPOINT_NODE_SIZE]
        ts_all = [t for t in ts_all if 'L2Loss' not in t.name]
        ts_all = [t for t in ts_all if 'entropy' not in t.name]
        ts_all = [t for t in ts_all if 'FusedBatchNorm' not in t.name]
        ts_all = [t for t in ts_all if 'Switch' not in t.name]
        ts_all = [t for t in ts_all if 'dropout' not in t.name]
        # DV: FP16_FIX - need to add 'Cast' layer here to make it work for FP16
        ts_all = [t for t in ts_all if 'Cast' not in t.name]

        # filter out all tensors that are inputs of the backward graph
        with util.capture_ops() as bwd_ops:
            tf_gradients(ys, xs, grad_ys, **kwargs)

        bwd_inputs = [t for op in bwd_ops for t in op.inputs]
        # list of tensors in forward graph that is in input to bwd graph
        ts_filtered = list(set(bwd_inputs).intersection(ts_all))
        debug_print("Using tensors {}".format(ts_filtered), 1)

        # try two slightly different ways of getting bottlenecks tensors
        # to checkpoint
        for ts in [ts_filtered, ts_all]:

            # get all bottlenecks in the graph
            bottleneck_ts = []
            for t in ts:
                b = set(ge.get_backward_walk_ops(t.op, inclusive=True, within_ops=fwd_ops))
                f = set(ge.get_forward_walk_ops(t.op, inclusive=False, within_ops=fwd_ops))
                # check that there are not shortcuts
                b_inp = set([inp for op in b for inp in op.inputs]).intersection(ts_all)
                f_inp = set([inp for op in f for inp in op.inputs]).intersection(ts_all)
                if not set(b_inp).intersection(f_inp) and len(b_inp) + len(f_inp) >= len(ts_all):
                    bottleneck_ts.append(t)  # we have a bottleneck!
                else:
                    debug_print("Rejected bottleneck candidate and ops {}".format(
                        [t] + list(set(ts_all) - set(b_inp) - set(f_inp))), 2)

            # success? or try again without filtering?
            if len(bottleneck_ts) >= np.sqrt(len(ts_filtered)):  # yes, enough bottlenecks found!
                break

        if not bottleneck_ts:
            raise Exception(
                'unable to find bottleneck tensors! please provide checkpoint nodes manually, or use checkpoints="speed".')

        # sort the bottlenecks
        bottlenecks_sorted_lists = tf_toposort(bottleneck_ts, within_ops=fwd_ops)
        sorted_bottlenecks = [t for ts in bottlenecks_sorted_lists for t in ts]

        # save an approximately optimal number ~ sqrt(N)
        N = len(ts_filtered)
        if len(bottleneck_ts) <= np.ceil(np.sqrt(N)):
            checkpoints = sorted_bottlenecks
        else:
            step = int(np.ceil(len(bottleneck_ts) / np.sqrt(N)))
            checkpoints = sorted_bottlenecks[step::step]

    checkpoints = list(set(checkpoints).intersection(ts_all))

    # at this point automatic selection happened and checkpoints is list of nodes
    assert isinstance(checkpoints, list)

    debug_print("Checkpoint nodes used: {}".format(checkpoints), 1)
    # better error handling of special cases
    # xs are already handled as checkpoint nodes, so no need to include them
    xs_intersect_checkpoints = set(xs).intersection(set(checkpoints))
    if xs_intersect_checkpoints:
        debug_print("Warning, some input nodes are also checkpoint nodes: {}".format(
            xs_intersect_checkpoints), 2)
    ys_intersect_checkpoints = set(ys).intersection(set(checkpoints))
    debug_print("ys: {}, checkpoints: {}, intersect: {}".format(ys, checkpoints,
                                                                ys_intersect_checkpoints), 1)
    # saving an output node (ys) gives no benefit in memory while creating
    # new edge cases, exclude them
    if ys_intersect_checkpoints:
        debug_print("Warning, some output nodes are also checkpoints nodes: {}".format(
            format_ops(ys_intersect_checkpoints)), 2)

    # remove initial and terminal nodes from checkpoints list if present
    checkpoints = list(set(checkpoints) - set(ys) - set(xs))

    # check that we have some nodes to checkpoint
    if not checkpoints:
        raise Exception('no checkpoints nodes found or given as input! ')

    debug_print("Select {} nodes to checkpoint nodes.".format(len(checkpoints)), 0)

    # disconnect dependencies between checkpointed tensors
    checkpoints_disconnected = {}
    for x in checkpoints:
        frontier_ops = set(graph_util.get_consuming_ops(x.op.outputs))
        debug_print("my frontier ops: {}".format(frontier_ops), 1)

        bw_frontier_ops = frontier_ops & set(bwd_ops)
        debug_print("my bw frontier ops: {}".format(bw_frontier_ops), 1)

        if len(bw_frontier_ops) > 1:
            continue

        if x.op and x.op.name is not None:
            grad_node = tf.stop_gradient(x, name=x.op.name + "_sg")
        else:
            grad_node = tf.stop_gradient(x)

        swapout_op = _add_swapout(grad_node.op, grad_node.op.outputs)
        incpu_count = incpu_count + 1
        swapin_op = _add_swapin(swapout_op, bw_frontier_ops, grad_node.op.outputs)
        checkpoints_disconnected[x] = swapin_op
        my_add_control_inputs(x, bw_frontier_ops, swapin_op)
        # control dependency -> swap_in
        # self._add_control_dependency(src_op, dest_op, swapin_op)

    # g = tf.get_default_graph()
    # print(g.get_operations())

    # partial derivatives to the checkpointed tensors and xs
    ops_to_copy = fast_backward_ops(seed_ops=[y.op for y in ys],
                                    stop_at_ts=checkpoints, within_ops=fwd_ops)
    debug_print("Found {} ops to copy within fwd_ops {}, seed {}, stop_at {}".format(
        len(ops_to_copy), fwd_ops, [r.op for r in ys], checkpoints), 1)
    debug_print("ops_to_copy = {}".format(ops_to_copy), 1)
    debug_print("Processing list {}".format(ys), 1)
    copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
    for origin_op, op in info._transformed_ops.items():
        op._set_device(origin_op.node_def.device)
    copied_ops = info._transformed_ops.values()
    debug_print("Copied {} to {}".format(ops_to_copy, copied_ops), 1)
    ge.reroute_ts(checkpoints_disconnected.values(), checkpoints_disconnected.keys(), can_modify=copied_ops)
    debug_print("Rewired {} in place of {} restricted to {}".format(
        checkpoints_disconnected.values(), checkpoints_disconnected.keys(), copied_ops), 1)

    # get gradients with respect to current boundary + original x's
    copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
    boundary = list(checkpoints_disconnected.values())
    dv = tf_gradients(ys=copied_ys, xs=boundary + xs, grad_ys=grad_ys, **kwargs)
    debug_print("Got gradients {}".format(dv), 1)
    debug_print("for {}".format(copied_ys), 1)
    debug_print("with respect to {}".format(boundary + xs), 1)

    inputs_to_do_before = [y.op for y in ys]
    if grad_ys is not None:
        inputs_to_do_before += grad_ys
    wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
    my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

    # partial derivatives to the checkpointed nodes
    # dictionary of "node: backprop" for nodes in the boundary
    d_checkpoints = {r: dr for r, dr in zip(checkpoints_disconnected.keys(),
                                            dv[:len(checkpoints_disconnected)])}
    # partial derivatives to xs (usually the params of the neural net)
    d_xs = dv[len(checkpoints_disconnected):]

    # incorporate derivatives flowing through the checkpointed nodes
    checkpoints_sorted_lists = tf_toposort(checkpoints, within_ops=fwd_ops)
    for ts in checkpoints_sorted_lists[::-1]:
        debug_print("Processing list {}".format(ts), 1)
        checkpoints_other = [r for r in checkpoints if r not in ts]
        checkpoints_disconnected_other = [checkpoints_disconnected[r] for r in checkpoints_other]

        # copy part of the graph below current checkpoint node, stopping at
        # other checkpoints nodes
        ops_to_copy = fast_backward_ops(within_ops=fwd_ops, seed_ops=[r.op for r in ts], stop_at_ts=checkpoints_other)
        debug_print("Found {} ops to copy within {}, seed {}, stop_at {}".format(
            len(ops_to_copy), fwd_ops, [r.op for r in ts],
            checkpoints_other), 1)
        debug_print("ops_to_copy = {}".format(ops_to_copy), 1)
        if not ops_to_copy:  # we're done!
            break
        copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
        for origin_op, op in info._transformed_ops.items():
            op._set_device(origin_op.node_def.device)
        copied_ops = info._transformed_ops.values()
        debug_print("Copied {} to {}".format(ops_to_copy, copied_ops), 1)
        ge.reroute_ts(checkpoints_disconnected_other, checkpoints_other, can_modify=copied_ops)
        debug_print("Rewired {} in place of {} restricted to {}".format(
            checkpoints_disconnected_other, checkpoints_other, copied_ops), 1)

        # gradient flowing through the checkpointed node
        boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
        substitute_backprops = [d_checkpoints[r] for r in ts]
        dv = tf_gradients(boundary,
                          checkpoints_disconnected_other + xs,
                          grad_ys=substitute_backprops, **kwargs)
        debug_print("Got gradients {}".format(dv), 1)
        debug_print("for {}".format(boundary), 1)
        debug_print("with respect to {}".format(checkpoints_disconnected_other + xs), 1)
        debug_print("with boundary backprop substitutions {}".format(substitute_backprops), 1)

        inputs_to_do_before = [d_checkpoints[r].op for r in ts]
        wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
        my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

        # partial derivatives to the checkpointed nodes
        for r, dr in zip(checkpoints_other, dv[:len(checkpoints_other)]):
            if dr is not None:
                if d_checkpoints[r] is None:
                    d_checkpoints[r] = dr
                else:
                    d_checkpoints[r] += dr

        def _unsparsify(x):
            if not isinstance(x, tf.IndexedSlices):
                return x
            assert x.dense_shape is not None, "memory_saving_gradients encountered sparse gradients of unknown shape"
            indices = x.indices
            while indices.shape.ndims < x.values.shape.ndims:
                indices = tf.expand_dims(indices, -1)
            return tf.scatter_nd(indices, x.values, x.dense_shape)

        # partial derivatives to xs (usually the params of the neural net)
        d_xs_new = dv[len(checkpoints_other):]
        for j in range(len(xs)):
            if d_xs_new[j] is not None:
                if d_xs[j] is None:
                    d_xs[j] = _unsparsify(d_xs_new[j])
                else:
                    d_xs[j] += _unsparsify(d_xs_new[j])

    return d_xs


def tf_toposort(ts, within_ops=None):
    all_ops = ge.get_forward_walk_ops([x.op for x in ts], within_ops=within_ops)

    deps = {}
    for op in all_ops:
        for o in op.outputs:
            deps[o] = set(op.inputs)
    sorted_ts = toposort(deps)

    # only keep the tensors from our original list
    ts_sorted_lists = []
    for l in sorted_ts:
        keep = list(set(l).intersection(ts))
        if keep:
            ts_sorted_lists.append(keep)

    return ts_sorted_lists


def fast_backward_ops(within_ops, seed_ops, stop_at_ts):
    bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
    ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
    return list(ops)


@contextlib.contextmanager
def capture_ops():
    """Decorator to capture ops created in the block.
    with capture_ops() as ops:
      # create some ops
    print(ops) # => prints ops created.
    """

    micros = int(time.time() * 10 ** 6)
    scope_name = str(micros)
    op_list = []
    with tf.name_scope(scope_name):
        yield op_list

    g = tf.get_default_graph()
    op_list.extend(ge.select_ops(scope_name + "/.*", graph=g))


def _to_op(tensor_or_op):
    if hasattr(tensor_or_op, "op"):
        return tensor_or_op.op
    return tensor_or_op


def _to_ops(iterable):
    if not _is_iterable(iterable):
        return iterable
    return [_to_op(i) for i in iterable]


def _is_iterable(o):
    try:
        _ = iter(o)
    except Exception:
        return False
    return True


def _add_swapout(src_op, ts0):
    with tf.device("/cpu:0"):
        swap_out = tf.identity(ts0, name="lms/swapout")

    # Connect: src-node -> swap-out
    print (ts0)
    print(src_op.outputs)
    print(swap_out[0].op.outputs)
    _connect_ops(src_op, swap_out[0].op)

    debug_print("Tensor {} will be placed on {}".format(
        ts0.name, "/cpu:0"), 1)

    return swap_out.op


def _add_swapin(swapout_op, dest_op, ts0):
    with tf.device("/cpu:0"):
        swap_in = tf.identity(ts0, name="lms/swapin")

    # Connect: swap_out -> swap_in
    _connect_ops(swapout_op, swap_in.op)

    # Connect: swap_in -> dest
    _connect_ops(swap_in.op, dest_op)

    return swap_in.op


def _connect_ops(src_op, dest_op, remap_inputs=False,
                 remap_outputs=True):
    src_sgv = ge.sgv(src_op, graph=tf.get_default_graph())
    dest_sgv = ge.sgv(dest_op, graph=tf.get_default_graph())

    if remap_outputs:
        src_sgv = src_sgv.remap_outputs([0])
    if remap_inputs:
        dest_sgv = dest_sgv.remap_inputs([0])

    ge.connect(src_sgv, dest_sgv)


DEBUG_LOGGING = True
DEBUG_LEVEL = 0


def debug_print(message, level=0):
    """Like logger.log, but also replaces all TensorFlow ops/tensors with their
    names. Sensitive to value of DEBUG_LOGGING, see enable_debug/disable_debug

    Usage:
      debug_print("see tensors {} for {}", tensorlist, [1,2,3])
    """

    if DEBUG_LOGGING and (DEBUG_LEVEL >= level):
        # formatted_args = [format_ops(arg) for arg in args]
        # tf.logging.info("[Test][{}] {}".format(level, tuple(formatted_args)))
        tf.logging.info("[LMS][{}] {}".format(level, message))
        # print("DEBUG " + s % tuple(formatted_args))


def format_ops(ops, sort_outputs=True):
    """Helper method for printing ops. Converts Tensor/Operation op to op.name,
    rest to str(op)."""

    if hasattr(ops, '__iter__') and not isinstance(ops, str):
        l = [(op.name if hasattr(op, "name") else str(op)) for op in ops]
        if sort_outputs:
            return sorted(l)
        return l
    else:
        return ops.name if hasattr(ops, "name") else str(ops)


def my_add_control_inputs(wait_to_do_ops, inputs_to_do_before):
    for op in wait_to_do_ops:
        ci = [i for i in inputs_to_do_before if op.control_inputs is None or i not in op.control_inputs]
        ge.add_control_inputs(op, ci)

