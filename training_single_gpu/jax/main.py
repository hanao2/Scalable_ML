import yaml
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from model import CustomModel
from utils import CustomActivation

def mixed_precision(samples, config: ConfigDict):
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

def activation_checkpointing(remat: bool):
    dtype = jnp.bfloat16
    x = jnp.ones(shape=(100, 128), dtype=dtype)
    act_fn = CustomActivation
    if remat:
        act_fn = jax.remat(act_fn)
    grad_fn = jax.grad(act_fn)
    out = grad_fn(x)
    print(out)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config["network_size"])
    mixed_precision(samples=100, config=config)
    #activation_checkpointing(remat=False)
