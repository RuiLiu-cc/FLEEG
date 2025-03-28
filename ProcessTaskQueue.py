import torch.multiprocessing

# revise from cyy's codes here:
# https://github.com/cyyever/distributed_learning_simulator/blob/main/topology/central_topology.py
# https://github.com/cyyever/naive_python_lib/blob/main/cyy_naive_lib/data_structure/task_queue.py


class TaskQueue:
    def __init__(self):
        self.__queues: dict = {}

    def __create_queue(self):
        return torch.multiprocessing.get_context("spawn").Queue()

    def get_queue(self, name: str, default=None):
        return self.__queues.get(name, default)

    def add_queue(self, name: str) -> None:
        assert name not in self.__queues
        self.__queues[name] = self.__create_queue()

    def has_data(self, queue_name: str = "result") -> bool:
        return not self.get_queue(queue_name).empty()

    def get_data(self, queue_name: str = "result"):
        result_queue = self.get_queue(queue_name)
        return result_queue.get()

    def put_data(self, data, queue_name: str = "result"):
        result_queue = self.get_queue(queue_name)
        return result_queue.put(data)


class CommunicationNework:
    def __init__(self, Common_config):
        super().__init__()
        self.__queue = TaskQueue()
        for worker_id in range(Common_config["num_clients"]):
            self.__queue.add_queue(f"result_{worker_id}")
            self.__queue.add_queue(f"request_{worker_id}")

    def get_from_client(self, worker_id):
        return self.__queue.get_data(queue_name=f"request_{worker_id}")

    def get_from_server(self, worker_id):
        return self.__queue.get_data(queue_name=f"result_{worker_id}")

    def server_has_data(self, worker_id: int) -> bool:
        return self.__queue.has_data(queue_name=f"result_{worker_id}")

    def client_has_data(self, worker_id: int) -> bool:
        return self.__queue.has_data(queue_name=f"request_{worker_id}")

    def send_to_server(self, data, worker_id):
        self.__queue.put_data(data=data, queue_name=f"request_{worker_id}")

    def send_to_client(self, data, worker_id):
        self.__queue.put_data(data=data, queue_name=f"result_{worker_id}")
