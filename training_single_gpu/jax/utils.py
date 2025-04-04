import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.struct import dataclass
from typing import Callable, Any, Tuple

PyTree = Any # nested dictionaries
Metrics = Tuple[str, Tuple[jax.Array, ...]]

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

def classification_loss(batch: Batch, apply_fn: Callable, params: PyTree, rng: jax.random.PRNGKey) -> Tuple[PyTree, Metrics]:
    inputs, labels = batch.inputs, batch.labels
    logits = apply_fn(params=params, x=inputs, train=True, rngs=rng) # the rng is for dropout
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    pred_class = jnp.argmax(logits, axis=-1)
    correct_preds = jnp.equal(pred_class, labels)
    metrics = {"loss": (loss.sum(), batch.shape[0]), "accuracy": (correct_preds.sum(), batch.shape[0])}
    return loss, metrics

def accumulate_gradients(batch: Batch, num_minibatches: int, rng: jax.random.PRNGKey, state: TrainState):
    minibatch_size = batch.shape[0] // num_minibatches
    rngs = jax.random.split(rng, num=num_minibatches)
    grads = None
    metrics = None
    for minibatch_idx in range(num_minibatches):
        with jax.named_scope(f"Minibatch {minibatch+1}"):
            start_ind = minibatch_idx * minibatch_size
            end_ind = start_ind + minibatch_size
            minibatch = jax.tree_map(lambda x: x[start_ind:end_ind], batch)
            vgrad = jax.value_and_grad(classification_loss, has_aux=True)
            (_, metric), grad = vgrad(minibatch, state.apply_fn, state.params, rngs[minibatch_idx])
            
            if grads is None:
                grads = grad
                metrics = metric
            else:
                grads = jax.tree_map(jnp.add, grads, grad)
                metrics = jax.tree_map(jnp.add, metrics, metric)
    grads = jax.tree_map(lambda x: x / num_minibatches, grads)
    return grads, metrics


def train_step(batch: Batch, state: TrainState, rng: jax.random.PRNGKey, num_minibatches: int):
    next_rng, cur_rng = jax.random.split(rng, num=2)
    grads, metrics = accumulate_gradients(batch, num_minibatches, cur_rng, state)
    state = state.apply_gradients(grads, next_rng)
    return state, grads, metrics

def train(batch: Batch, state: TrainState, rng: jax.random.PRNGKey, num_epochs=5):
    for _ in range(num_epochs):
        state, grads, metrics = train_step(batch, state, rng)
    return grads, metrics










# ask Alireza decorator versus parent class
# jax.jit automatic or not?



