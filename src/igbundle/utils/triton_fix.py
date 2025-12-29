"""
Triton/TorchAO Compatibility Fix for Windows

This module MUST be imported BEFORE torch or transformers.
It patches missing Triton components that PyTorch/TorchAO expect.

Usage:
    # At the very top of your script, before any other imports:
    import sys
    import os
    os.environ['DISABLE_TORCHAO'] = '1'  # Optional: block TorchAO
    sys.path.insert(0, 'path/to/src')
    from igbundle.utils import triton_fix  # This auto-applies the fix
    
    # Now safe to import torch, transformers, etc.
    import torch
"""

import sys
import os
import types
import warnings

def apply_triton_fix():
    """
    Workaround for Triton 3.x / Torch 2.6 compatibility issue on Windows.
    Fixes:
    - ImportError: cannot import name 'AttrsDescriptor' from 'triton.backends.compiler'
    - ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'
    """
    
    # Skip if not Windows - Triton works fine on Linux
    if os.name != 'nt':
        return
    
    # Dummy AttrsDescriptor class
    class AttrsDescriptor:
        """Dummy AttrsDescriptor for Windows compatibility."""
        divisibility_16 = ()
        equal_to_1 = ()
        ids_of_folded_args = ()
        divisibility_8 = ()
        
        def __init__(self, *args, divisibility_16=(), equal_to_1=(), 
                     ids_of_folded_args=(), divisibility_8=(), **kwargs):
            self.divisibility_16 = divisibility_16
            self.equal_to_1 = equal_to_1
            self.ids_of_folded_args = ids_of_folded_args
            self.divisibility_8 = divisibility_8
            
        @classmethod
        def from_dict(cls, data):
            return cls(**data) if isinstance(data, dict) else cls()
    
    try:
        import triton
        _patch_triton_module(triton, AttrsDescriptor)
    except ImportError:
        # Triton not installed - create full mock
        _create_mock_triton(AttrsDescriptor)
        
def _patch_triton_module(triton, AttrsDescriptor):
    """Patch an existing triton installation."""
    # triton.backends.compiler path
    if not hasattr(triton, 'backends'):
        triton.backends = types.ModuleType('triton.backends')
        sys.modules['triton.backends'] = triton.backends
    
    if not hasattr(triton.backends, 'compiler'):
        triton.backends.compiler = types.ModuleType('triton.backends.compiler')
        sys.modules['triton.backends.compiler'] = triton.backends.compiler
        
    if not hasattr(triton.backends.compiler, 'AttrsDescriptor'):
        triton.backends.compiler.AttrsDescriptor = AttrsDescriptor

    # triton.compiler.compiler path
    if not hasattr(triton, 'compiler'):
        triton.compiler = types.ModuleType('triton.compiler')
        sys.modules['triton.compiler'] = triton.compiler
        
    if not hasattr(triton.compiler, 'compiler'):
        triton.compiler.compiler = types.ModuleType('triton.compiler.compiler')
        sys.modules['triton.compiler.compiler'] = triton.compiler.compiler
        
    if not hasattr(triton.compiler.compiler, 'AttrsDescriptor'):
        triton.compiler.compiler.AttrsDescriptor = AttrsDescriptor

def _create_mock_triton(AttrsDescriptor):
    """Create a complete mock triton module structure."""
    # Create main triton module
    triton = types.ModuleType('triton')
    sys.modules['triton'] = triton
    
    # Create backends.compiler
    triton.backends = types.ModuleType('triton.backends')
    triton.backends.compiler = types.ModuleType('triton.backends.compiler')
    triton.backends.compiler.AttrsDescriptor = AttrsDescriptor
    sys.modules['triton.backends'] = triton.backends
    sys.modules['triton.backends.compiler'] = triton.backends.compiler
    
    # Create compiler.compiler
    triton.compiler = types.ModuleType('triton.compiler')
    triton.compiler.compiler = types.ModuleType('triton.compiler.compiler')
    triton.compiler.compiler.AttrsDescriptor = AttrsDescriptor
    sys.modules['triton.compiler'] = triton.compiler
    sys.modules['triton.compiler.compiler'] = triton.compiler.compiler

class TorchAOBlocker:
    """Import blocker for TorchAO to prevent compatibility issues."""
    
    def find_module(self, name, path=None):
        if name == 'torchao' or name.startswith('torchao.'):
            return self
        return None
    
    def load_module(self, name):
        # Create a dummy module that doesn't crash on import
        dummy = types.ModuleType(name)
        dummy.__file__ = '<blocked>'
        dummy.__loader__ = self
        dummy.__package__ = name.rsplit('.', 1)[0] if '.' in name else name
        sys.modules[name] = dummy
        return dummy

def disable_torchao():
    """
    Prevent TorchAO from loading to avoid compatibility issues.
    Call this BEFORE importing transformers.
    """
    # Check if already blocked
    for finder in sys.meta_path:
        if isinstance(finder, TorchAOBlocker):
            return
    
    # Insert blocker at start of meta_path
    sys.meta_path.insert(0, TorchAOBlocker())
    print("TorchAO import blocked for Windows compatibility.")

# Auto-apply fixes on import
apply_triton_fix()

# Disable TorchAO if environment variable is set
if os.environ.get('DISABLE_TORCHAO', '0') == '1':
    disable_torchao()
