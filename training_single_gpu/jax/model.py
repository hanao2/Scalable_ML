import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

class CustomModel(nn.Module):
    dtype: Any
    hidden_size: int = 256
    num_classes: int = 100
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array: # in pytorch we use 'forward' whereas in jax and tensorflow we use __call__.
        x = nn.Dense(self.hidden_size, dtype=self.dtype)(x) # unlike pytorch, jax doesn't require passing the input size. Would find it based on the input (x).
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.tanh(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = x.astype(jnp.float32) # This conversion is necessary before applying sigmoid for numerical stability and precision. Prevents under/overflow in sigmoid.
        x = nn.log_sigmoid(x)
        return x
