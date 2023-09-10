from queue import Queue


class Store:
    def __init__(self) -> None:
        self.queue: Queue = Queue()
        self.simulation_running: bool = False
        self.simulation_paused: bool = False
        self.reset: bool = False

    def put(self, data):
        self.queue.put(data)

    def get(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()

    def set_simulation_running(self, simulation_running):
        self.simulation_running = simulation_running

    def is_simulation_running(self):
        return self.simulation_running

    def is_simulation_paused(self):
        return self.simulation_paused

    def set_simulation_paused(self, simulation_paused):
        self.simulation_paused = simulation_paused

    def set_simulation_reset(self, reset):
        self.reset = reset

    def is_simulation_reset(self):
        return self.reset
