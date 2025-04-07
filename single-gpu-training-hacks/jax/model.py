import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from ml_collections import ConfigDict

class CustomModel(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array: # in pytorch we use 'forward' whereas in jax and tensorflow we use __call__.
        x = nn.Dense(self.config.hidden_size, dtype=self.config.dtype)(x) # unlike pytorch, jax doesn't require passing the input size. Would find it based on the input (x).
        x = nn.LayerNorm(dtype=self.config.dtype)(x)
        x = nn.tanh(x)
        x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.config.num_classes, dtype=self.config.dtype)(x)
        x = x.astype(jnp.float32) # This conversion is necessary before applying sigmoid for numerical stability and precision. Prevents under/overflow in sigmoid.
        x = nn.log_sigmoid(x)
        return x
