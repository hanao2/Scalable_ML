import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.struct import dataclass
from typing import Callable, Any, Tuple
from ml_collections import ConfigDict


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


def gelu(x: jax.Array) -> jax.Array:
    """GeLU activation function with approximate tanh."""
    jax.debug.print("Executing GeLU")
    x3 = jnp.power(x, 3)
    tanh_input = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
    return 0.5 * x * (1 + jnp.tanh(tanh_input))


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


def get_num_params(state: TrainState) -> int:
    return sum(np.prod(x.shape)
               for x in jax.tree_util.tree_leaves(state.params))
