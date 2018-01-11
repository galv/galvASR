import os
import tensorflow as tf

from tensorflow.python.eager import execute as _execute
from tensorflow.python.eager import context as _context
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.core.framework import types_pb2 as _types_pb2
from tensorflow.python.framework import op_def_registry as _registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library

# This file was based on the gen_dataset_ops.py generated file (see
# bazel-genfiles in the tensorflow directory). If it is possible, I'd
# like to use the same bazel build rule to generate that file to
# generate this one as well, but it might be tricky to (1) load
# tensorflow's build rules and (2) to use bazel within
# CMake. TODO(galv)

# Is this the right place to load the library?
kaldi_io = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../tensorflow_ext/libtensorflow_ext.so'))


def kaldi_table_dataset_with_op_name(op_name, r_specifier, name=None):
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(op_name, r_specifier=r_specifier, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    r_specifier = _ops.convert_to_tensor(r_specifier, _dtypes.string)
    _inputs_flat = [r_specifier]
    _attrs = None
    _result = _execute.execute(op_name, 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(op_name, _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def _create_kaldi_table_dataset_op_proto(op_name):
  # TODO(galv): See if I can call a C++ function to get this protobuf
  # directly from whatever REGISTER_OP expands to, so we can follow
  # DRY.
  op = _op_def_pb2.OpDef()
  # Does this need to be bytes rather than unicode str? Hmm...
  op.name = op_name

  r_specifier = _op_def_pb2.OpDef.ArgDef()
  r_specifier.name = "r_specifier"
  r_specifier.type = _types_pb2.DT_STRING
  op.input_arg.extend([r_specifier])

  output_handle = _op_def_pb2.OpDef.ArgDef()
  output_handle.name = "handle"
  output_handle.type = _types_pb2.DT_VARIANT
  op.output_arg.extend([output_handle])

  op.is_stateful = True

  return op


def _create_op_def_library(op_protos):
  for op_proto in op_protos:
    registered_ops = _registry.get_registered_ops()
    if op_proto.name not in registered_ops:
      raise LookupError("Op with name {0} not registered".format(op_proto.name))

    op_def_lib = _op_def_library.OpDefLibrary()
    ops_proto = _op_def_pb2.OpList()
    ops_proto.op.extend([op_proto])

  # Fails if the interfaces ("op schemas") don't match between the
  # previously registered op and this one.
  _registry.register_op_list(ops_proto)

  op_def_lib.add_op_list(ops_proto)

  return op_def_lib


_op_def_lib = _create_op_def_library(
  [_create_kaldi_table_dataset_op_proto("KaldiFloat32MatrixDataset"),])
  # _create_kaldi_table_dataset_op_proto("KaldiFloat32VectorDataset"),
  # _create_kaldi_table_dataset_op_proto("KaldiInt32VectorDataset")])
