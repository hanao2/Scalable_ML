# Scale Training on a Single GPU

Streamlining the training procedure doesnot necessarily mean multi-GPU training. It could happen on a single GPU as well, through approaches like mixed-precision, gradient accumulation or activation checkpointing. These are all techniques to reduce the memory consumption of the model. In this project we work through each approach in a separate python file and report some visuals here to better grasp their advantage.

We will implement these techniques using both JAX and PyTorch (JAX for now). To implement our model we use `Flax` which is a neural network library and ecosystem for JAX (disclosure: I keep confusing flask with flax!! :)). 