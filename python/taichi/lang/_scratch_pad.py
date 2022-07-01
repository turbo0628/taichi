import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang.impl import get_runtime
from taichi.lang.util import (python_scope, to_numpy_type, to_paddle_type,
                              to_pytorch_type)
from taichi.types.primitive_types import f32, f64, i32, i64



class ScratchPad:
    """Taichi field with SNode implementation.

    A field is constructed by a list of field members.
    For example, a scalar field has 1 field member, while a 3x3 matrix field has 9 field members.
    A field member is a Python Expr wrapping a C++ GlobalVariableExpression.
    A C++ GlobalVariableExpression wraps the corresponding SNode.

    Args:
        vars (List[Expr]): Field members.
    """
    def __init__(self, _vars):
        # print(_vars)
        self.vars = _vars
        # assert len(_vars) == 1
        assert isinstance(_vars, int)
        get_runtime().prog.current_ast_builder().expr_alloca_scratch_pad((_vars,), f32)


    @property
    def dtype(self):
        """Gets data type of each individual value.

        Returns:
            DataType: Data type of each individual value.
        """
        return self._snode._dtype

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self.host_accessors[0].setter(value, *self._pad_key(key))

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        # Check for potential slicing behaviour
        # for instance: x[0, :]
        padded_key = self._pad_key(key)
        for key in padded_key:
            if not isinstance(key, int):
                raise TypeError(
                    f"Detected illegal element of type: {type(key)}. "
                    f"Please be aware that slicing a ti.field is not supported so far."
                )
        return self.host_accessors[0].getter(*padded_key)

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return '<ti.field>'

# class SNodeHostAccessor:
#     def __init__(self, snode):
#         if _ti_core.is_real(snode.data_type()):

#             def getter(*key):
#                 assert len(key) == _ti_core.get_max_num_indices()
#                 return snode.read_float(key)

#             def setter(value, *key):
#                 assert len(key) == _ti_core.get_max_num_indices()
#                 snode.write_float(key, value)
#         else:
#             if _ti_core.is_signed(snode.data_type()):

#                 def getter(*key):
#                     assert len(key) == _ti_core.get_max_num_indices()
#                     return snode.read_int(key)
#             else:

#                 def getter(*key):
#                     assert len(key) == _ti_core.get_max_num_indices()
#                     return snode.read_uint(key)

#             def setter(value, *key):
#                 assert len(key) == _ti_core.get_max_num_indices()
#                 snode.write_int(key, value)

#         self.getter = getter
#         self.setter = setter


# class SNodeHostAccess:
#     def __init__(self, accessor, key):
#         self.accessor = accessor
#         self.key = key


__all__ = ["ScratchPad"]
