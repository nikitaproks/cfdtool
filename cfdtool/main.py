import jax
from simulation.circle_simulation import simulation


def main():
    print("Hello World!")
    jax.config.update("jax_enable_x64", True)  # Needs to be
    simulation()


if __name__ == "__main__":
    main()
