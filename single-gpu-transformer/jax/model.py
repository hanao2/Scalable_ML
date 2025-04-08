import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from ml_collections import ConfigDict
from utils import gelu


class MLPBlock(nn.Module):
    config: ConfigDict
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        input_dim = x.shape[0]
        x = nn.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
        x = nn.Dense(
            self.config.hidden_size,
            dtype=self.config.dtype,
            name="input_layer")(x)
        x = gelu(x)
        x = nn.Dense(
            input_dim,
            dtype=self.config.dtype,
            name="output_layer")(x)
        x = nn.Dropout(
            rate=self.config.dropout_rate,
            deterministic=not self.train)(x)
        return x


def dot_product_attention(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: jax.Array,
        softmax_dtype: jnp.dtype) -> jax.Array:
    dtype = query.dtype
    query_dim = query.shape[0]
    scale = query_dim ** -0.5
    query = query * scale
    query = query.astype(softmax_dtype)
    key = key.astype(softmax_dtype)
    weights = jnp.einsum("...qhd,...khd->...hqk", query, key)
    if mask is not None:
        weights = jnp.where(mask, weights, jnp.finfo(softmax_dtype))
    weights = nn.softmax(weights, axis=-1)
    weights = weights.astype(dtype)
    new_vals = jnp.einsum("...hqk,...khd->...qhd", weights, value)
    return new_vals


class AttentionBlock(nn.Module):
    config: ConfigDict
    mask: jax.Array
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        input_dim = x.shape[0]
        x = nn.LayerNorm(dtype=self.config.dtype, name="pre_norm")(x)
        qkv = nn.DenseGeneral(
            (self.config.num_heads,
             self.config.head_dim * 3),
            dtype=self.config.dtype,
            name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        x = dot_product_attention(
            q, k, v, self.mask, self.config.softmax_dtype)
        x = nn.DenseGeneral(input_dim, axis=(-2, -1),
                            dtype=self.config.dtype, name="output_layer")(x)
        x = nn.Dropout(
            rate=self.config.dropout_rate,
            deterministic=not self.train)(x)
        return x


class TransformerBlock(nn.Module):
    config: ConfigDict
    mask: jax.Array
    train: bool

    @nn.compact
    def __call__(self, x: jax.Array):
