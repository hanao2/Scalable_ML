import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.struct import dataclass
from typing import Callable, Any, Tuple
from ml_collections import ConfigDict
from model import CustomModel
from utils import custom_loss
from utils import Batch, TrainState, train


PyTree = Any  # nested dictionaries
Metrics = Tuple[str, Tuple[jax.Array, ...]]


def set_XLA_flags():
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )


def custom_activation(x: jax.Array) -> jax.Array:
    jax.debug.print('CustomActivation running ...')
    out = jax.nn.sigmoid(x)
    return out


def custom_loss(x: jax.Array, remat: bool) -> jax.Array:
    act_fn = custom_activation
    if remat:
        act_fn = jax.remat(act_fn)
    return jnp.mean(act_fn(x))


class TrainState(train_state.TrainState):
    rng: jax.random.PRNGKey


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


def classification_loss(batch: Batch,
                        apply_fn: Callable,
                        params: PyTree,
                        rng: jax.random.PRNGKey) -> Tuple[PyTree,
                                                          Metrics]:
    inputs, labels = batch.inputs, batch.labels
    logits = apply_fn({"params": params}, x=inputs,
                      train=True, rngs=rng)  # the rng is for dropout
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    pred_class = jnp.argmax(logits, axis=-1)
    correct_preds = jnp.equal(pred_class, labels)
    metrics = {
        "loss": (
            loss.sum(),
            batch.inputs.shape[0]),
        "accuracy": (
            correct_preds.sum(),
            batch.inputs.shape[0])}
    loss = loss.mean()
    return loss, metrics


def accumulate_gradients(
        batch: Batch,
        num_minibatches: int,
        rng: jax.random.PRNGKey,
        state: TrainState):
    minibatch_size = batch.inputs.shape[0] // num_minibatches
    rngs = jax.random.split(rng, num=num_minibatches)
    vgrad = jax.value_and_grad(classification_loss, has_aux=True, argnums=2)
    grads = None
    metrics = None
    for minibatch_idx in range(num_minibatches):
        with jax.named_scope(f"Minibatch {minibatch_idx+1}"):
            start_ind = minibatch_idx * minibatch_size
            end_ind = start_ind + minibatch_size
            minibatch = jax.tree_map(lambda x: x[start_ind:end_ind], batch)
            # , allow_int=True)
            (_, metric), grad = vgrad(minibatch,
                                      state.apply_fn, state.params, rngs[minibatch_idx])

            if grads is None:
                grads = grad
                metrics = metric
            else:
                grads = jax.tree_map(jnp.add, grads, grad)
                metrics = jax.tree_map(jnp.add, metrics, metric)
    grads = jax.tree_map(lambda x: x / num_minibatches, grads)
    return grads, metrics

# @jax.jit(static_argnums=(2,))


def train_step(batch: Batch, state: TrainState, num_minibatches: int):
    next_rng, cur_rng = jax.random.split(state.rng, num=2)
    grads, metrics = accumulate_gradients(
        batch, num_minibatches, cur_rng, state)
    state = state.apply_gradients(grads=grads, rng=next_rng)
    return state, grads, metrics


def train(
        batch: Batch,
        state: TrainState,
        num_epochs: int,
        num_minibatches: int):
    jit_train_step = jax.jit(train_step, static_argnums=(2,))
    for _ in range(num_epochs):
        state, grads, metrics = jit_train_step(batch, state, num_minibatches)
    return grads, metrics


def mixed_precision(samples: int, config: ConfigDict):
    dtype = jnp.float32  # compare this with jnp.bfloat16 and jnp.float16
    x = jnp.ones(shape=(samples, config.input_size), dtype=dtype)
    key = jax.random.PRNGKey(0)
    params_key, dropout_key = jax.random.split(key, num=2)
    config.dtype = dtype
    model = CustomModel(config=config)

    # tabulate model
    # you need to pass the arguments of the __call__ function as well as
    # random number generators! Welcome to functional programming! :)
    table = model.tabulate(rngs=params_key, x=x, train=True)
    print(table)

    # run a forward pass
    params = model.init(rngs=params_key, x=x, train=True)
    out = model.apply(params, x, train=True, rngs=dropout_key)


def activation_checkpointing(samples: int, config: ConfigDict, remat: bool):
    dtype = jnp.bfloat16
    config.dtype = dtype
    x = jnp.ones(shape=(samples, config.input_size), dtype=dtype)
    grad_fn = jax.grad(custom_loss)
    _ = grad_fn(x, remat=False)


def gradient_accumulation(
        samples: int,
        config: ConfigDict,
        num_minibatches: int,
        dropout_rate: float):
    dtype = jnp.bfloat16
    config.dropout_rate = dropout_rate
    config.dtype = dtype
    rng = jax.random.PRNGKey(0)
    rng_input, rng_label, rng_param, rng_model = jax.random.split(rng, num=4)
    batch = Batch(
        inputs=jax.random.normal(rng_input, (samples, config.input_size)),
        labels=jax.random.randint(rng_label, (samples,), 0, config.num_classes)
    )
    model = CustomModel(config=config)
    params = model.init(rng_param, batch.inputs, train=False)["params"]
    state = TrainState.create(
        params=params,
        apply_fn=model.apply,
        tx=optax.adam(1e-3),
        rng=rng_model
    )

    grads, metrics = train(batch, state, num_epochs=5,
                           num_minibatches=num_minibatches)
    print(f'accuracy:', metrics['accuracy'][0] / metrics['accuracy'][1])
    print(f'loss:', metrics['loss'][0] / metrics['loss'][1])
