import yaml
import jax
import jax.numpy as jnp
import optax
from ml_collections import ConfigDict
import warnings
from tqdm import tqdm
from model import Transformer
from utils import set_XLA_flags, Batch, TrainState, get_num_params, train_step_transformer
import pdb


if __name__ == "__main__":
    print(jax.devices())
    with open("../utils/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config)
    config.dtype = jnp.bfloat16
    config.softmax_dtype = jnp.float32
    num_heads = config.hidden_size // config.head_dim
    if num_heads != config.num_heads:
        warnings.warn(
            f"Your requested number of heads ({config.num_heads}) does not match your hidden size and head dimension (number of heads = {num_heads}), so we update it for you!",
            stacklevel=2)
        config.num_heads = num_heads

    model = Transformer(config=config)
    rng = jax.random.PRNGKey(config.seed)
    rng_token, rng_params, rng_model = jax.random.split(rng, 3)
    optimizer = optax.adam(
        learning_rate=optax.warmup_exponential_decay_schedule(
            init_value=0,
            peak_value=float(config.learning_rate),
            warmup_steps=10,
            transition_steps=1,
            decay_rate=0.99,
        )
    )
    tokens = jax.random.randint(
        rng_token,
        (config.batch_size, config.max_seq_len),
        1,
        config.vocab_size,
    )
    batch = Batch(
        inputs=jnp.pad(tokens[:, :-1], ((0, 0), (1, 0)), constant_values=0),
        labels=tokens
    )

    params = model.init(
        rngs=rng_params,
        x=batch.inputs[: config.batch_size] // config.num_minibatches,
        train=False,
    )["params"]
    state = TrainState.create(
        params=params,
        apply_fn=model.apply,
        tx=optimizer,
        rng=rng_model,
    )
    # pdb.set_trace()
    print(f'Number of parameters: {get_num_params(state):_}')

    # for _ in tqdm(range(4)):
    #     state, metrics = train_step_transformer(state=state, metrics=None, batch=batch, num_minibatches=config.num_minibatches, train=True)
    # print("loss: ", metrics["loss"][0] / metrics["loss"][1])
    # print("loss: ", metrics["accuracy"][0] / metrics["accuracy"][1])
