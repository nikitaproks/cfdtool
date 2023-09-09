import threading
import logging
import time

import jax

from utils.schemas import Settings
from utils.classes import Store
from server import setup_app
from simulation import simulation


jax.config.update("jax_enable_x64", True)

logging.basicConfig(
    format="%(asctime)s, %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
)


def run_simulation(store: Store, settings: Settings):
    while True:
        if store.is_simulation_running():
            t_simulation = threading.Thread(
                target=simulation, args=(store, settings)
            )
            logging.info("Simulation thread started")
            t_simulation.start()
            t_simulation.join()  # Wait for the simulation to finish

            if store.is_simulation_reset():
                logging.info("Simulation thread finished because of reset")
            else:
                logging.info("Simulation thread finished on its own")
            logging.info("Waiting for user to start new simulation")
            store.set_simulation_running(False)
            store.set_simulation_reset(False)
            store.set_simulation_paused(False)
        else:
            time.sleep(0.5)


def run_server(store: Store, settings: Settings):
    app = setup_app(store, settings)
    app.run_server(debug=True)


if __name__ == "__main__":
    store = Store()
    settings = Settings(width=400, height=10, iterations=1000000)

    t_simulation = threading.Thread(
        target=run_simulation, args=(store, settings)
    )
    t_simulation.start()

    # t_server = threading.Thread(target=run_server, args=(store,))
    # t_server.start()

    run_server(store, settings)
