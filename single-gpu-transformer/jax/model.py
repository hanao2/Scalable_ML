import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from ml_collections import ConfigDict
import functools
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
    def __call__(self, x: jax.Array) -> jax.Array:
        # MLP block
        mlp = MLPBlock
        if "MLP" in self.config.remat:
            mlp = nn.remat(mlp, prevent_cse=False)
        # residual connection
        x = x + mlp(self.config, self.train, name="mlp")(x)
        # Attention n=block
        attn = AttentionBlock
        if "Attn" in self.config.remat:
            attn = nn.remat(attn, prevent_cse=False)
        x = x + attn(self.config, self.mask, self.train,
                     name="attn")(x)  # residual connection
        return x


class Transformer(nn.Module):
    config: ConfigDict

    @nn.compact
    def __call__(
            self,
            x: jax.Array,
            mask: jax.Array,
            train: bool) -> jax.Array:
        if mask is None and self.config.causal_mask:
            mask = nn.make_causal_mask(x, dtype=jnp.bool_)
        # Input layer (implement embedding and positional encoding)
        x = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.config.dtype,
            name="embed",
        )(x)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.config.max_seq_len, self.config.hidden_size),
        )
        pos_emb = pos_emb.astype(self.config.dtype)
        x = x + pos_emb[None, :x.shape[1]]
        # Transformer blocks
        transformer_fn = functools.partial(
            TransformerBlock, self.config, mask, train)
        if "transformer" in self.config.remat:
            transformer_fn = nn.remat(transformer_fn, prevent_cse=False)
        if self.config.scan_layers:
            transformer = transformer_fn(name="transformer")
            x, _ = nn.scan(
                # carry is the input (x) to each block/module
                lambda module, carry, _: (module(carry), None),
                variable_axes={"params": 0},  # ask
                split_rngs={"params": True, "dropout": True},  # ask
                length=self.config.num_transformer_layers,
            )(transformer, x, ())
        else:
            for idx in range(self.config.num_transformer_layers):
                x = transformer_fn(name=f"block_{idx}")(x)
        # Output layer
        x = nn.Layernorm(dtype=self.config.dtype, name="post_norm")(x)
        x = nn.Dense(
            features=self.config.num_outputs,  # what is the output?
            dtype=self.config.dtype,
            name="output_layer",
        )(x)
        x = x.astype(jnp.float32)  # why this?
        return x
