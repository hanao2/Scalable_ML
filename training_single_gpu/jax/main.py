import jax
import jax.numpy as jnp
from model import CustomModel

def main():
    dtype = jnp.bfloat16
    x = jnp.ones(shape=(100, 128), dtype=dtype)
    rngs = {'params': jax.random.PRNGKey(0)} #, 'dropout': jax.random.PRNGKey(1)} # to keep everything reproducible, jax requires random number generators for parameters (to initialize them) and dropout layer (to leave-out neurons) separately.
    model = CustomModel(dtype=dtype)
    table = model.tabulate(rngs=rngs, x=x, train=True) # you need to pass the arguments of the __call__ function as well as random number generators! Welcome to functional programming! :)
    print(table)


if __name__ == "__main__":
    main()
