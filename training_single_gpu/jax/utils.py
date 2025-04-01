import jax

def CustomActivation(x):
    jax.debug.print('CustomActivation running ...')
    out = jax.nn.sigmoid(x)
    return out.sum()
