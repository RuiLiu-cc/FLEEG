import concurrent
import traceback
from log import get_logger


class ExecutorPool(concurrent.futures._base.Executor):
    def __init__(self, executor):
        self.__executor = executor
        self.__futures = []

    def submit(self, fn, *args, **kwargs):
        """Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Returns:
            A Future representing the given call.
        """
        future = self.__executor.submit(ExecutorPool._fun_wrapper, fn, *args, **kwargs)
        self.__futures.append(future)
        return future

    def wait_results(
        self, timeout=None, return_when=concurrent.futures._base.ALL_COMPLETED
    ):
        concurrent.futures.wait(
            self.__futures, timeout=timeout, return_when=return_when
        )
        results = []
        for future in self.__futures:
            try:
                result = future.result()
                get_logger().info("future result is %s", result)
            except BaseException as e:
                get_logger().warning("future has exception is %s", str(e))
            results.append(result)
        self.__futures.clear()
        return results

    def shutdown(self):
        self.__executor.shutdown(wait=True)

    @classmethod
    def _fun_wrapper(cls, fn, *args, **kwargs):
        try:
            # if inspect.iscoroutinefunction(fn):
            #     return asyncio.run(fn(*args, **kwargs))
            return fn(*args, **kwargs)
        except Exception as e:
            get_logger().error("catch exception:%s", e)
            get_logger().error("traceback:%s", traceback.format_exc())
            return None
