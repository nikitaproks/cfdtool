import time
import logging

from utils.classes import Store
from utils.schemas import Settings

logger = logging.getLogger(__name__)


def simulation_management(store: Store) -> bool:
    if store.is_simulation_paused():
        logger.info("Simulation paused")
        while store.is_simulation_paused():
            if store.is_simulation_reset():
                logger.info("Simulation reset. Exiting simulation loop")
                return False
            time.sleep(0.5)
        logger.info("Simulation continued running")
    return True


def simulation(store: Store, settings: Settings):
    print("Simulation started running")
    for z in range(0, settings.iterations):
        continue_simulation: bool = simulation_management(store)
        if not continue_simulation:
            break
        store.put([[z + j + i for j in range(10)] for i in range(10)])
