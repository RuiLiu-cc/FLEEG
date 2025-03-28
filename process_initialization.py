import threading
from typing import Callable

__local_data = threading.local()


def default_initializer(*init_args):
    # We save fun_kwargs for further processing and call the initialization function
    __local_data.data = {}
    if len(init_args) == 3:
        __local_data.data = init_args[2]
    initializers: list[Callable] = init_args[0]
    if not isinstance(initializers, list):
        initializers = [initializers]
    for initializer in initializers:
        initializer(**init_args[1])


def get_process_data() -> dict:
    return __local_data.data


def forward(fun, *args, **kwargs):
    return fun(*args, queue_network=get_process_data()["queue_network"], **kwargs)
