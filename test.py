import jax
import jax.numpy as jnp

# import tensorflow as tf

# x = jnp.arange(5)
# isinstance(x, jax.Array)  # returns True both inside and outside traced functions.

# def f(x: Array) -> Array:  # type annotations are valid for traced and non-traced types.
#   return x

# x = jnp.arange(10)
# y = jnp.asarray(x)
# print(x)
# print(type(x))
# print('================================================')
# print(y)
# print(type(y))
# print('================================================')
# print(y.shape)

print(jnp.finfo(jnp.float32).max)
