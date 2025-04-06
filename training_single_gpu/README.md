# Scale Training on a Single GPU

Streamlining the training procedure doesnot necessarily mean multi-GPU training. It could happen on a single GPU as well, through approaches like mixed-precision, gradient accumulation or activation checkpointing. These are all techniques to reduce the memory consumption of the model. In this project we work through each approach in a separate python file and report some visuals here to better grasp their advantage.

We will implement these techniques using both JAX and PyTorch (JAX for now). To implement our model we use `Flax` which is a neural network library and ecosystem for JAX (disclosure: I keep confusing flask with flax! :smile:). 

When tabulating (`model.tabulate`) or initiating the model (`model.init`), you should pass a random number generator for the parameters ('params'), otherwise you receive the following error message
```
flax.errors.InvalidRngError: Dense_0 needs PRNG for "params" (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.InvalidRngError)
```
By calling `tabulate`, you're just building the structure of the network, and no actual computation or parameter updates are taking place. Dropout, however, is typically disabled during this stage. It's treated as part of the model architecture definition, but no randomness is involved yet. The dropout mask is not actually generated, so you don't need an RNG for it at this point. However, dropout RNG is required once you run forward pass through `model.apply`. Otherwise, you'll get the following error message
```
flax.errors.InvalidRngError: Dropout_0 needs PRNG for "dropout" (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.InvalidRngError)
```
There are exclusive functions in `main.py`, each implementing/practicing reducing the memory footprint in one of these approaches: mixed precision, activation checkpointing and gradient accumulation. Also, to dictate the network configuration, such as network size and dropout rate, we use `ConfigDict` from the `ml_collections` library which allows for dot-accessing the keys.
