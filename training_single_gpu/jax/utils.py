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

def accumulate_gradients(batch: Batch, num_minibatches: int):
    minibatch_size = batch.shape[0] // 




# ask Alireza decorator versus parent class
# jax.jit automatic or not?



