import torch
import torch.nn as nn
from ..modules.adapter import IGBundleAdapter

class IGBundleBlockWrapper(nn.Module):
    """
    Wrapper that injects IGBundle adapter into transformer blocks.
    Properly delegates all attributes to original_block for compatibility
    with newer transformers versions (4.45+).
    """
    
    # Attributes that should NOT be delegated (our own attributes)
    _OWN_ATTRS = {'original_block', 'adapter', '_parameters', '_buffers', '_modules', 
                  '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', 
                  '_state_dict_hooks', '_load_state_dict_pre_hooks', 'training'}
    
    def __init__(self, original_block, adapter):
        super().__init__()
        # Use simple assignment, which triggers our __setattr__
        # Our __setattr__ will use nn.Module.__setattr__ for these, ensuring registration
        self.original_block = original_block
        self.adapter = adapter
        
    def __getattr__(self, name):
        """Delegate attribute access to original_block for transformer compatibility."""
        # 1. Try generic nn.Module lookups (modules, parameters, buffers)
        # We try/except because we want to fallback to delegation
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
            
        # 2. Check if we should stop here
        if name in self._OWN_ATTRS:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 3. Delegate to original_block
        # We access original_block via super().__getattr__ implicitly if we just use self.original_block?
        # No, self.original_block triggers this __getattr__ again if not in __dict__.
        # But it SHOULD be in _modules found by super().__getattr__.
        # So we can safely verify we have it.
        try:
             original_block = super().__getattr__('original_block')
        except AttributeError:
             # Should not happen if initialized correctly
             raise AttributeError(f"'{type(self).__name__}' missing 'original_block'")
             
        return getattr(original_block, name)
    
    def __setattr__(self, name, value):
        """Allow setting our own attributes, delegate others to original_block."""
        if name in self._OWN_ATTRS or name.startswith('_'):
            # Use nn.Module.__setattr__ to ensure proper registration of Modules/Params
            nn.Module.__setattr__(self, name, value)
        else:
            try:
                # We need to get original_block without triggering infinite recursion if it's missing
                # But here we assume it exists or we fall back to self
                if 'original_block' in self._modules:
                    original_block = self._modules['original_block']
                    setattr(original_block, name, value)
                else:
                     # Fallback if original_block not set yet?
                     # Actually if we are failing to finding original_block, we might default to local set?
                     # But original_block is set in __init__ which goes to 'if' branch.
                     # So this else is for OTHER attributes.
                     nn.Module.__setattr__(self, name, value)
            except AttributeError:
                nn.Module.__setattr__(self, name, value)

    @property
    def attention_type(self):
        """Return attention_type for newer transformers (4.45+) compatibility."""
        original_block = object.__getattribute__(self, 'original_block')
        if hasattr(original_block, 'attention_type'):
            return original_block.attention_type
        # Fallback: check for _attn_implementation (common in HF models)
        if hasattr(original_block, '_attn_implementation'):
            return original_block._attn_implementation
        # Default to 'sdpa' (Scaled Dot-Product Attention)
        return 'sdpa'
        
    def forward(self, hidden_states, *args, **kwargs):
        # Call original block
        # HF blocks usually return (hidden, params...) or just hidden tuple
        outputs = self.original_block(hidden_states, *args, **kwargs)
        
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
            
        # Call adapter
        # Adapter needs h. Context? We can use h as context or something else.
        # For now, context is None (self-reference)
        
        # FIX: Ensure h matches adapter dtype (e.g. if h is Half from 4bit but adapter is BFloat16)
        original_dtype = h.dtype
        adapter_dtype = next(self.adapter.parameters()).dtype
        h_casted = h.to(adapter_dtype)
        
        h_new, state = self.adapter(h_casted)
        
        # Cast back to original dtype to ensure downstream compatibility
        if h_new.dtype != original_dtype:
            h_new = h_new.to(original_dtype)
        
        setattr(h_new, 'ig_state', state)
        
        if isinstance(outputs, tuple):
            return (h_new,) + outputs[1:]
        else:
            return h_new

def wrap_hf_candidate(model, adapter_config):
    """
    Injects IGBundleAdapters into the model.
    """
    # 1. Identify layers
    layers = None
    curr = model
    # Recursive search for 'layers' or 'h'
    for _ in range(5): # Don't go too deep
        if hasattr(curr, 'layers'):
            layers = curr.layers
            break
        elif hasattr(curr, 'model') and curr.model is not curr:
            curr = curr.model
        elif hasattr(curr, 'base_model') and curr.base_model is not curr:
            curr = curr.base_model
        elif hasattr(curr, 'transformer'):
            curr = curr.transformer
        elif hasattr(curr, 'h'):
            layers = curr.h
            break
        else:
            break
            
    if layers is None:
        raise ValueError(f"Could not locate transformer layers in model. Model type: {type(model)}")
        
    # 2. Iterate and wrap
    for i, layer in enumerate(layers):
        adapter = IGBundleAdapter(adapter_config)
        
        # Detect device/dtype from layer
        try:
            target_device = next(layer.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu") # Fallback
            
        # For 4bit/8bit, we want adapter in bf16 or fp32, not 4bit
        # We default to bf16 if available, else fp32
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        adapter.to(device=target_device, dtype=target_dtype)
        
        wrapper = IGBundleBlockWrapper(layer, adapter)
        layers[i] = wrapper
        
    return model

def collect_states(model_output):
    """
    Traverses the output graph or uses hooks to retrieve states? 
    Actually, if we attached 'ig_state' to hidden states, 
    the final hidden state only has the last state.
    Intermediate states are lost unless we accumulated them.
    
    Better approach for `forward`:
    The Wrapper should append to a list if provided in kwargs? 
    HF forward signatures are strict.
    
    Alternative: Register a forward hook on the adapters to capture output states into a list.
    """
    pass

class StateCollector:
    def __init__(self):
        self.states = []
        
    def hook(self, module, input, output):
        # output is (h_new, state) from adapter
        # Wait, wrapper calls adapter(h) -> h_new, state
        # Wrapper returns h_new.
        # So we hook the ADAPTER not the wrapper.
        pass
        
    def attach(self, model):
        self.handles = []
        self.states = []
        
        def get_hook(layer_idx):
            def hk(module, input, output):
                # output is (h, state)
                self.states.append(output[1])
            return hk

        # Find all IGBundleAdapters
        for name, module in model.named_modules():
            if isinstance(module, IGBundleAdapter):
                self.handles.append(module.register_forward_hook(get_hook(len(self.handles))))
                
    def clear(self):
        self.states = []
        
    def detach(self):
        for h in self.handles:
            h.remove()
