"""Small, bounded device test; no training or checkpoint writes."""
import faulthandler, time
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
import jax
import jax.numpy as jnp
print('DEVICES', jax.devices(), flush=True)
t0 = time.monotonic()
x = jax.random.normal(jax.random.PRNGKey(0), (32, 32))
x.block_until_ready()
print('RANDOM_SECONDS', time.monotonic()-t0, flush=True)
q, r = jnp.linalg.qr(x)
q.block_until_ready()
print('QR_SECONDS', time.monotonic()-t0, flush=True)
faulthandler.cancel_dump_traceback_later()
