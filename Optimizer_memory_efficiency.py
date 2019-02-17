from toposort import toposort
import contextlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
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


class OME(object):

    def __init__(self, graph=None,
                 grad_ys=None,
                 debug=False,
                 debug_level=1,
                 cpu_device="/cpu:0",
                 **kwargs):

        self._graph = graph
        self._grad_ys = grad_ys
        self._topo_sort = None
        self._cpu_device = cpu_device
        self._debug = debug
        self._debug_level = debug_level

        # keep log of tensors on host
        self._incpu_count = 0

        # store a dictionary of visited ops to avoid multiple visits
        self._ops_dict = {}
        self.kwargs = kwargs

    def run(self, graph=None):

        if graph:
            self._graph = graph

        if not self._graph:
            raise ValueError('The dataflow graph is required but has not been'
                             ' provided.')

        self._log_info("Editing model for LMS")
        start_time = time.time()

        loss_ops = tf.get_default_graph().get_operations()[-1:]
        xs_ops = tf.trainable_variables()

        # forward ops are all ops that are candidates for recomputation
        #    print("Calling memsaving gradients with", checkpoints)
        if not isinstance(loss_ops, list):
            ys = [loss_ops]
        if not isinstance(xs_ops, list):
            xs = [xs_ops]

        bwd_ops = ge.get_backward_walk_ops([y.op for y in loss_ops],
                                           inclusive=True)

        self._log_info("bwd_ops {}".format(bwd_ops), 2)

        # forward ops are all ops that are candidates for recomputation
        fwd_ops = ge.get_forward_walk_ops([x.op for x in xs],
                                          inclusive=True,
                                          within_ops=bwd_ops)
        self._log_info("fwd_ops: %s".format(fwd_ops, 2))

        # exclude ops with no inputs
        fwd_ops = [op for op in fwd_ops if op.inputs]

        # don't recompute xs, remove variables
        xs_ops = self._to_ops(xs)
        fwd_ops = [op for op in fwd_ops if not op in xs_ops]
        fwd_ops = [op for op in fwd_ops if not '/assign' in op.name]
        fwd_ops = [op for op in fwd_ops if not '/Assign' in op.name]
        fwd_ops = [op for op in fwd_ops if not '/read' in op.name]
        ts_all = ge.filter_ts(fwd_ops, True)  # get the tensors
        ts_all = [t for t in ts_all if '/read' not in t.name]
        ts_all = set(ts_all) - set(xs) - set(ys)

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
            tf_gradients(ys, xs, self._grad_ys, **self.kwargs)

        bwd_inputs = [t for op in bwd_ops for t in op.inputs]
        # list of tensors in forward graph that is in input to bwd graph
        ts_filtered = list(set(bwd_inputs).intersection(ts_all))
        self._log_info("Using tensors {}".format(ts_filtered), 1)

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
                    self._log_info("Rejected bottleneck candidate and ops {}".format(
                        [t] + list(set(ts_all) - set(b_inp) - set(f_inp))), 2)

            # success? or try again without filtering?
            if len(bottleneck_ts) >= np.sqrt(len(ts_filtered)):  # yes, enough bottlenecks found!
                break

        if not bottleneck_ts:
            raise Exception(
                'unable to find bottleneck tensors! please provide checkpoint nodes manually, or use checkpoints="speed".')

        # sort the bottlenecks
        bottlenecks_sorted_lists = self.tf_toposort(bottleneck_ts, within_ops=fwd_ops)
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

        self._log_info("Checkpoint nodes used: %s", checkpoints, 0)
        # better error handling of special cases
        # xs are already handled as checkpoint nodes, so no need to include them
        xs_intersect_checkpoints = set(xs).intersection(set(checkpoints))
        if xs_intersect_checkpoints:
            self._log_info("Warning, some input nodes are also checkpoint nodes: {}".format(
                xs_intersect_checkpoints), 2)
        ys_intersect_checkpoints = set(ys).intersection(set(checkpoints))
        self._log_info("ys: {}, checkpoints: {}, intersect: {}".format(ys, checkpoints,
                                                                       ys_intersect_checkpoints), 1)
        # saving an output node (ys) gives no benefit in memory while creating
        # new edge cases, exclude them
        if ys_intersect_checkpoints:
            self._log_info("Warning, some output nodes are also checkpoints nodes: {}".format(
                self.format_ops(ys_intersect_checkpoints)), 2)

        # remove initial and terminal nodes from checkpoints list if present
        checkpoints = list(set(checkpoints) - set(ys) - set(xs))

        # check that we have some nodes to checkpoint
        if not checkpoints:
            raise Exception('no checkpoints nodes found or given as input! ')

        # disconnect dependencies between checkpointed tensors
        checkpoints_disconnected = {}
        for x in checkpoints:
            if x.op and x.op.name is not None:
                grad_node = tf.stop_gradient(x, name=x.op.name + "_sg")
            else:
                grad_node = tf.stop_gradient(x)
            checkpoints_disconnected[x] = grad_node

        # partial derivatives to the checkpointed tensors and xs
        ops_to_copy = self.fast_backward_ops(seed_ops=[y.op for y in ys],
                                             stop_at_ts=checkpoints, within_ops=fwd_ops)
        self._log_info("Found {} ops to copy within fwd_ops {}, seed {}, stop_at {}".format(
            len(ops_to_copy), fwd_ops, [r.op for r in ys], checkpoints), 1)
        self._log_info("ops_to_copy = {}".format(ops_to_copy), 2)
        self._log_info("Processing list {}".format(ys), 2)
        copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
        for origin_op, op in info._transformed_ops.items():
            op._set_device(origin_op.node_def.device)
        copied_ops = info._transformed_ops.values()
        self._log_info("Copied {} to {}".format(ops_to_copy, copied_ops), 2)
        ge.reroute_ts(checkpoints_disconnected.values(), checkpoints_disconnected.keys(), can_modify=copied_ops)
        self._log_info("Rewired {} in place of {} restricted to {}".format(
            checkpoints_disconnected.values(), checkpoints_disconnected.keys(), copied_ops), 2)

        # get gradients with respect to current boundary + original x's
        copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
        boundary = list(checkpoints_disconnected.values())
        dv = tf_gradients(ys=copied_ys, xs=boundary + xs)
        self._log_info("Got gradients {}".format(dv), 2)
        self._log_info("for {}".format(copied_ys), 2)
        self._log_info("with respect to {}".format(boundary + xs), 2)

        inputs_to_do_before = [y.op for y in ys]
        if self._grad_ys is not None:
            inputs_to_do_before += self._grad_ys
        wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
        my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

        # partial derivatives to the checkpointed nodes
        # dictionary of "node: backprop" for nodes in the boundary
        d_checkpoints = {r: dr for r, dr in zip(checkpoints_disconnected.keys(),
                                                dv[:len(checkpoints_disconnected)])}
        # partial derivatives to xs (usually the params of the neural net)
        d_xs = dv[len(checkpoints_disconnected):]

        # incorporate derivatives flowing through the checkpointed nodes
        checkpoints_sorted_lists = self.tf_toposort(checkpoints, within_ops=fwd_ops)
        for ts in checkpoints_sorted_lists[::-1]:
            self._log_info("Processing list {}".format(ts), 2)
            checkpoints_other = [r for r in checkpoints if r not in ts]
            checkpoints_disconnected_other = [checkpoints_disconnected[r] for r in checkpoints_other]

            # copy part of the graph below current checkpoint node, stopping at
            # other checkpoints nodes
            ops_to_copy = self.fast_backward_ops(within_ops=fwd_ops, seed_ops=[r.op for r in ts],
                                            stop_at_ts=checkpoints_other)
            self._log_info("Found {} ops to copy within {}, seed {}, stop_at {}".format(
                        len(ops_to_copy), fwd_ops, [r.op for r in ts],
                        checkpoints_other), 2)
            self._log_info("ops_to_copy = {}".format(ops_to_copy), 2)
            if not ops_to_copy:  # we're done!
                break
            copied_sgv, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
            for origin_op, op in info._transformed_ops.items():
                op._set_device(origin_op.node_def.device)
            copied_ops = info._transformed_ops.values()
            self._log_info("Copied {} to {}".format(ops_to_copy, copied_ops), 2)
            ge.reroute_ts(checkpoints_disconnected_other, checkpoints_other, can_modify=copied_ops)
            self._log_info("Rewired {} in place of {} restricted to {}".format(
                        checkpoints_disconnected_other, checkpoints_other, copied_ops), 2)

            # gradient flowing through the checkpointed node
            boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
            substitute_backprops = [d_checkpoints[r] for r in ts]
            dv = tf_gradients(boundary,
                              checkpoints_disconnected_other + xs,
                              grad_ys=substitute_backprops, **self.kwargs)
            self._log_info("Got gradients {}".format(dv), 2)
            self._log_info("for {}".format(boundary), 2)
            self._log_info("with respect to {}".format(checkpoints_disconnected_other + xs), 2)
            self._log_info("with boundary backprop substitutions {}".format(substitute_backprops), 2)

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

    def tf_toposort(self, ts, within_ops=None):
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

    def fast_backward_ops(self, within_ops, seed_ops, stop_at_ts):
        bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
        ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
        return list(ops)

    @contextlib.contextmanager
    def capture_ops(self):
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

    def _to_op(self, tensor_or_op):
        if hasattr(tensor_or_op, "op"):
            return tensor_or_op.op
        return tensor_or_op

    def _to_ops(self, iterable):
        if not self._is_iterable(iterable):
            return iterable
        return [self._to_op(i) for i in iterable]

    def _is_iterable(self, o):
        try:
            _ = iter(o)
        except Exception:
            return False
        return True

    def _log_info(self, message, level=0):
        """Log debug information.
        Args:
          message: a formatted string.
          level: an `integer`.
        """
        if level == 0 or (self._debug and self._debug_level >= level):
            # Use tf.logging.info instead of print, since print
            # is not thread safe, which can break tests.
            tf.logging.info("[OME][{}] {}".format(level, message))

    def format_ops(self, ops, sort_outputs=True):
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

    -----------------------------------------------------

    reachable_ops = set()
    for seed_op in seed_ops:
        reachable_ops |= set(self._get_forward_walk_ops(seed_op))

    for op in reachable_ops:
        if 'lms/swap' in op.name:
            self._log_info('This model has already been updated with LMS '
                           'swap operations. LMS will not re-process it.')
            return
    # exclusive ops
    self._excl_ops = self._filter_scopes_and_types(reachable_ops,
                                                   self._excl_scopes,
                                                   self._excl_types)
    # inclusive ops
    self._incl_ops = self._filter_scopes_and_types(reachable_ops,
                                                   self._incl_scopes,
                                                   self._incl_types)

    reachable_ops -= self._grad_ops

    # build a topological sort
    self._topo_sort = topos.TOPOS(seed_ops, self._grad_ops)
    self._topo_sort.build()
    for i in range(0, self._topo_sort.size):
        self._log_info("[{}]: {}".format(
            i, [op.name for op in self._topo_sort.get_ops(i)]), 1)

    self._do_action(seed_ops)

    # check the validation of the new model
    new_reachable_ops = set()
    for seed_op in seed_ops:
        new_reachable_ops |= set(ge.get_forward_walk_ops(seed_op))
    new_reachable_ops -= self._grad_ops
    if (new_reachable_ops >= reachable_ops):
        self._log_info("Edited model is valid and logically equivalent to the original one")
        self._log_info("Added {} ops into the model".format(len(new_reachable_ops - reachable_ops)))
    else:
        self._log_info("Edited model is invalid. Running this may produce unexpected result")

    self._log_info("Editing model for LMS, took: {} ms".format(
        (time.time() - start_time) * 1000))
    self._log_info(
        "{} tensors will be swapped out(in) to(from) the host".format(
            self._incpu_count))
    return (new_reachable_ops - reachable_ops)


def _do_action(self, src_ops):
    """Add swapin and swapout ops for ops that are reachable from `src_ops`.
    Args:
      src_ops: a list of `tf.Operation`
    """
    open_set = Queue.Queue()
    closed_set = set()

    for op in src_ops:
        open_set.put(op)

    while not open_set.empty():
        src_op = open_set.get()

        # get next ops before the graph is changed
        next_ops = set()
        for t in src_op.outputs:
            frontier_ops = set(util.get_consuming_ops(t))
            next_ops |= frontier_ops - self._grad_ops

        # do action for src_op
        self._insert_swap_nodes(src_op)
        if self._swapped_max_tensors():
            return

        for op in next_ops:
            if op in closed_set:
                continue
            if op not in open_set.queue:
                open_set.put(op)

        closed_set.add(src_op)

