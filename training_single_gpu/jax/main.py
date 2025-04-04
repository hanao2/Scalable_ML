import yaml
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from model import CustomModel
from utils import custom_loss

def mixed_precision(samples: int, config: ConfigDict):
    dtype = jnp.float32 # compare this with jnp.bfloat16 and jnp.float16
    x = jnp.ones(shape=(samples, config.input_size), dtype=dtype)
    key = jax.random.PRNGKey(0)
    params_key, dropout_key = jax.random.split(key, num=2)
    config.dtype = dtype
    model = CustomModel(config=config)

    # tabulate model
    table = model.tabulate(rngs=params_key, x=x, train=True) # you need to pass the arguments of the __call__ function as well as random number generators! Welcome to functional programming! :)
    print(table)

    # run a forward pass
    params = model.init(rngs=params_key, x=x, train=True)
    out = model.apply(params, x, train=True, rngs=dropout_key)


def activation_checkpointing(samples: int, config: ConfigDict, remat: bool):
    dtype = jnp.bfloat16
    x = jnp.ones(shape=(samples, config.input_size), dtype=dtype)
    grad_fn = jax.grad(custom_loss)
    _ = grad_fn(x, remat=False)

 
def gradient_accumulation(samples: int, config: ConfigDict, batch_size: int, num_minibatches: int):
    rng = jax.random.PRNGKey(0)
    rng_input, rng_label, rng_param, rng_model = jax.random.split(rng, num=4)

if __name__ == "__main__":
    with open("../utils/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config["network_size"])
    #mixed_precision(samples=100, config=config)
    #activation_checkpointing(samples=100, config=config, remat=False)

