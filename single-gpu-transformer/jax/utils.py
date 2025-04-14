import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.struct import dataclass
from typing import Callable, Any, Tuple, Union
from ml_collections import ConfigDict
import functools


PyTree = Any  # nested dictionaries
Metrics = Tuple[str, Tuple[jax.Array, ...]]


def set_XLA_flags():
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        # "--xla_enable_async_collective_permute=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )


# def gelu(x: jax.Array) -> jax.Array:
#     """GeLU activation function with approximate tanh."""
#     jax.debug.print("Executing GeLU")
#     x3 = jnp.power(x, 3)
#     tanh_input = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
#     return 0.5 * x * (1 + jnp.tanh(tanh_input))


class TrainState(train_state.TrainState):
    rng: jax.random.PRNGKey


@dataclass
class Batch:
    inputs: jax.Array
    labels: jax.Array


def next_token_pred_loss(params: PyTree,
                         apply_fn: Callable,
                         rng: jax.random.PRNGKey,
                         batch: Batch,
                         train: bool) -> Tuple[PyTree,
                                               Metrics]:
    logits = apply_fn({"params": params}, rngs={
                      "dropout": rng}, x=batch.inputs, train=train)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch.labels)
    correct_preds = jnp.equal(jnp.argmax(logits, axis=-1), batch.labels)
    batch_size = np.prod(batch.labels.shape)
    metrics = {
        "loss": (
            loss.sum(), batch_size), "accuracy": (
            correct_preds.sum(), batch_size)}
    loss = loss.mean()
    return loss, metrics


def accumulate_gradients(
        batch: Batch,
        num_minibatches: int,
        rng: jax.random.PRNGKey,
        state: TrainState,
        loss_fn: Callable,
        train: bool):
    minibatch_size = batch.inputs.shape[0] // num_minibatches
    rngs = jax.random.split(rng, num=num_minibatches)
    vgrad = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)
    grads = None
    metrics = None
    for minibatch_idx in range(num_minibatches):
        with jax.named_scope(f"Minibatch {minibatch_idx+1}"):
            start_ind = minibatch_idx * minibatch_size
            end_ind = start_ind + minibatch_size
            minibatch = jax.tree_map(lambda x: x[start_ind:end_ind], batch)
            # , allow_int=True)
            (_, metric), grad = vgrad(state.params,
                                      state.apply_fn,
                                      rngs[minibatch_idx],
                                      minibatch,
                                      train=train)

            if grads is None:
                grads = grad
                metrics = metric
            else:
                grads = jax.tree_map(jnp.add, grads, grad)
                metrics = jax.tree_map(jnp.add, metrics, metric)
    grads = jax.tree_map(lambda x: x / num_minibatches, grads)
    return grads, metrics


def get_num_params(state: TrainState) -> int:
    return sum(np.prod(x.shape)
               for x in jax.tree_util.tree_leaves(state.params))


@functools.partial(jax.jit, donate_argnames=("state", "metrics"),
                   static_argnames=("num_minibatches", "train"))
def train_step_transformer(state: TrainState,
                           metrics: Union[Metrics,
                                          None],
                           batch: Batch,
                           num_minibatches: int,
                           train: bool) -> Tuple[PyTree,
                                                 Tuple[str,
                                                       Metrics]]:
    next_rng, cur_rng = jax.random.split(state.rng, 2)
    grads, metric = accumulate_gradients(
        batch=batch,
        num_minibatches=num_minibatches,
        rng=cur_rng,
        state=state,
        loss_fn=next_token_pred_loss,
        train=train,
    )
    state = state.apply_gradients(grads=grads, rng=next_rng)
    if metrics is None:
        metrics = metric
    else:
        metrics = jax.tree.map(jnp.add, metrics, metric)
    return state, metrics
