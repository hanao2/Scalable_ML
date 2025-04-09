# Efficient Training of a  Transformer Model on a Single GPU

Transformer is a large model, and onse should be familiar with how to train it using limited resources. In this project we develop a transformer model, and look into benefitting from the three training hacks (mixed precision, activation checkpointing and gradient accumulation) that we have discussed in a separate [repo](../single-gpu-training-hacks/).

We also try to profile this model, as a way to optimize its performance.

You might wonder how useful the `nn.compact` decorator is when defining the model architecture. In Flax, there are two ways to define submodules and variables: 1) Explicitly: through `setup` method, which is somewhat similar to the `__init__` constructor in PyTorch. 2) In-line: through `nn.compact` decorator, which allows defining the whole module in a single method. You can find instructions on using each [here](https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/setup_or_nncompact.html).

The different transformer components such as MLP block ([MLPBlock](jax/model.py#L9)), attention layer ([dot_product_attention](jax/model.py#L32)), attention block ([AttentionBlock](jax/model.py#L53)) and their combination ([TransformerBlock](jax/model.py#L78)) are scripted separately. FOr the rematerialization/checkpointing approach, the target function/submodules are defined in the config file. Here we use the `remat` function from Flax, which is different from the JAX one that we used [earlier](../single-gpu-training-hacks/jax/utils.py#L19).

There are different types of masking for the attention block. Padded mask would allow for including sequences of different length in the same batch. It zero-pads the mask for instances with $L < L_{max}$. Causal (look-ahead) mask would leave-out the following tokens in order to prevent the model from cheating by looking at the future tokens in advance. The causal mask is an upper-triangular matrix, and is especially important in autoregressive models such as GPT. Lastly, cutomized masks could be applied where specific tokens are masked out in the attention layer. 

In the [Transformer](jax/model.py#L100) class, there are some points that I would like to mention. Initially we have data of size `batch * sequence length` with the sequence length being the maximum length among all batch instances. Then an embedding layer is applied on this input to featurize each token acoording to its ID and by referring to the lookup table. The embedded data would then be of size `batch * sequence length * hidden size`. Moreover, positional encoding is performed by initiating an array of size `maximum sequence length * hidden size`.This positional encoding is then added to the embedded data, which is then fed to the [TransformerBlock](jax/model.py#L78). However, only the first `sequence length` indices of the positional encoding is added, as `sequense length <= maximum sequence length`.


