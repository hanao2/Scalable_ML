# Efficient Training of a  Transformer Model on a Single GPU

Transformer is a large model, and onse should be familiar with how to train it using limited resources. In this project we develop a transformer model, and look into benefitting from the three training hacks (mixed precision, activation checkpointing and gradient accumulation) that we have discussed in a separate [repo](../single-gpu-training-hacks/).

We also try to profile this model, as a way to optimize its performance.

You might wonder how useful the `nn.compact` decorator is when defining the model architecture. In Flax, there are two ways to define submodules and variables: 1) Explicitly: through `setup` method, which is somewhat similar to the `__init__` constructor in PyTorch. 2) In-line: through `nn.compact` decorator, which allows defining the whole module in a single method. You can find instructions on using each [here](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/setup_or_nncompact.html).

The different transformer components such as MLP block ([MLPBlock](jax/model.py#L9)), attention layer ([dot_product_attention](jax/model.py#L32)), attention block ([AttentionBlock](jax/model.py#L53)) and their combination ([TransformerBlock](jax/model.py#L85)) are scripted separately.



