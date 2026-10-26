"""Idle CUDA placeholder for gpuq.sh: one tiny array keeps a compute context on the GPU."""

import time

import jax.numpy as jnp

x = jnp.zeros(16)
x.block_until_ready()
while True:
    time.sleep(3600)
