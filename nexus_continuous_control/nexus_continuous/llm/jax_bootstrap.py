"""Make 'jax' importable for the extension.

Call 'ensure_jax()' before importing anything that does 'import
jax.numpy' (currently just 'interpreter.py').

This exists so 'test_llm.py' and the demo notebook can exercise
'interpreter.py''s real code path (rule evaluation, reward-term
compilation, mask/meta-policy construction) in environments that don't have
the real JAX/CUDA stack installed, without ever silently shadowing a genuine
JAX installation.
"""

from __future__ import annotations 
import importlib.util 
import os 
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STUB_DIR = os.path.join(_THIS_DIR, "_jax_stub")
_ensured = False 
using_stub = False 

def ensure_jax() -> bool:
    """ Idempotently make 'import jax' work. 
        Returns True iff the stub was used. 
    """
    global _ensured, using_stub 
    if _ensured:
        return using_stub
    
    if importlib.util.find_spec("jax") is None:
        if _STUB_DIR not in sys.path: 
            sys.path.insert(0, _STUB_DIR)
        using_stub = True 
    _ensured = True 
    return using_stub