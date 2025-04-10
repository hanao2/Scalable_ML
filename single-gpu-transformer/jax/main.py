import yaml
import jax
import jax.numpy as jnp
import optax
from ml_collections import ConfigDict
import warnings
from model import Transformer
from utils import set_XLA_flags, Batch, TrainState, get_num_params
print(jax.devices())


set_XLA_flags()

if __name__ == "__main__":
    with open("../utils/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config)
    num_heads = config.hidden_size // config.head_dim
    if not num_heads == config.num_heads:
        warnings.warn(
            f"Your requested number of heads ({config.num_heads}) does not match your hidden size and head dimension (number of heads = {num_heads}), so we update it for you!",
            stacklevel=2)
        config.num_heads = num_heads

    model = Transformer(config=config)
    rng = jax.random.PRNGKey(config.seed)
    rng_token, rng_params, rng_model = jax.random.split(rng, 3)
    optimizer = optax.warmup_exponential_decay_schedule(
        init_value=0,
        peak_value=config.learning_rate,
        warmup_setup=10,
        transition_steps=1,
        decay_rate=0.99,
    )
    tokens = jax.random.randint(
        rng_token,
        (config.batch_size, config.max_seq_length),
        1,
        config.vocab_size,
    )
    batch = Batch(
        inputs=jnp.pad(tokens[:, :-1], ((0, 0), (1, 0)), constant_balues=0),
        labels=tokens
    )
    params = model.init(
        rng_params,
        batch.inputs[: config.batch_size] // config.num_minibatches,
        train=False,
    )["params"]
    state = TrainState.create(
        params,
        apply_fn=model.apply,
        tx=optimizer,
        rng=rng_model,
    )
    print(f'Number of parameters: {get_num_params(state):_}')
