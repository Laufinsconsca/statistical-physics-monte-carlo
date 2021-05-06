from abc import ABC

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import box
from numba.extending import make_attribute_wrapper
from numba.extending import models, register_model
from numba.extending import typeof_impl
from numba.extending import unbox, NativeValue


class ExecutionProgress(object):
    """
    Класс, хранящий параметры вывода процента выполнения в консоль
    """
    output_progress_to_console = False
    lower_bound = 0
    number_of_decimal_places = 0

    def __init__(self, output_progress_to_console, lower_bound,
                 number_of_decimal_places):
        self.output_progress_to_console = output_progress_to_console
        self.lower_bound = lower_bound
        self.number_of_decimal_places = number_of_decimal_places


"""
Код ниже позволяет использовать класс в коде Numba
"""


class ExecutionProgressType(types.Type, ABC):
    def __init__(self):
        super(ExecutionProgressType, self).__init__(name='ExecutionProgress')


execution_progress_type = ExecutionProgressType()


@typeof_impl.register(ExecutionProgress)
def typeof_index(val, c):
    return execution_progress_type


@register_model(ExecutionProgressType)
class ExecutionProgressModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('output_progress_to_console', types.boolean),
            ('lower_bound', types.float64),
            ('number_of_decimal_places', types.intc)
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(ExecutionProgressType, 'output_progress_to_console', 'output_progress_to_console')
make_attribute_wrapper(ExecutionProgressType, 'lower_bound', 'lower_bound')
make_attribute_wrapper(ExecutionProgressType, 'number_of_decimal_places', 'number_of_decimal_places')


@unbox(ExecutionProgressType)
def unbox_execution_progress(typ, obj, c):
    """
    Convert a ExecutionProgress object to a native structure.
    """
    output_progress_to_console_obj = c.pyapi.object_getattr_string(obj, "output_progress_to_console")
    is_true = c.pyapi.object_istrue(output_progress_to_console_obj)
    zero = ir.Constant(is_true.type, 0)
    lower_bound_obj = c.pyapi.object_getattr_string(obj, "lower_bound")
    number_of_decimal_places_obj = c.pyapi.object_getattr_string(obj, "number_of_decimal_places")
    execution_progress = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    execution_progress.output_progress_to_console = c.builder.icmp_signed('!=', is_true, zero)
    execution_progress.lower_bound = c.pyapi.float_as_double(lower_bound_obj)
    ll_type = c.context.get_argument_type(types.intc)
    val = cgutils.alloca_once(c.builder, ll_type)
    longobj = c.pyapi.number_long(number_of_decimal_places_obj)
    with c.pyapi.if_object_ok(longobj):
        llval = c.pyapi.long_as_longlong(longobj)
        c.pyapi.decref(longobj)
        c.builder.store(c.builder.trunc(llval, ll_type), val)
    execution_progress.number_of_decimal_places = c.builder.load(val)
    c.pyapi.decref(lower_bound_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(execution_progress._getvalue(), is_error=is_error)


@box(ExecutionProgressType)
def box_execution_progress(typ, val, c):
    """
    Convert a native structure to an ExecutionProgress object.
    """
    execution_progress = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    output_progress_to_console_obj = \
        c.pyapi.bool_from_long(c.builder.zext(execution_progress.output_progress_to_console, c.pyapi.long))
    lower_bound_obj = c.pyapi.float_from_double(execution_progress.lower_bound)
    ullval = c.builder.zext(execution_progress.number_of_decimal_places, c.pyapi.ulonglong)
    number_of_decimal_places_obj = c.pyapi.long_from_ulonglong(ullval)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(ExecutionProgress))
    res = c.pyapi.call_function_objargs(class_obj, (output_progress_to_console_obj, lower_bound_obj,
                                                    number_of_decimal_places_obj))
    c.pyapi.decref(output_progress_to_console_obj)
    c.pyapi.decref(lower_bound_obj)
    c.pyapi.decref(number_of_decimal_places_obj)
    c.pyapi.decref(class_obj)
    return res