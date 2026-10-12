"""Run in a clean process so the two vendored Fizgig packages cannot mix."""
from pathlib import Path
import sys
import tempfile
import torch

NODE = Path(__file__).resolve().parents[1] / 'ComfyUI-DINKIssTyle'
sys.path.insert(0, str(NODE / 'minimax_trainer/vendor/src'))
sys.path.insert(0, str(NODE))
from fizgig.families.registry import FAMILIES
from fizgig.families.lora import FamilyLoRA
from fizgig.minimax.model import MiniMaxH3DiT, MiniMaxH3Config
from minimax_trainer.runner import validate_lora

assert set(FAMILIES) == {'minimax'}
torch.manual_seed(7)
cfg = MiniMaxH3Config(hidden_size=32, num_layers=2, token_refiner_num_layers=1,
    num_attention_heads=2, attention_head_dim=16, ffn_hidden_size=64, text_dim=16,
    timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16, rope_inv_freq_len=2)
model = MiniMaxH3DiT(cfg).to(torch.bfloat16).requires_grad_(False)
base = {k: v.clone() for k, v in model.state_dict().items()}
model.enable_gradient_checkpointing()
driver = FAMILIES['minimax'].load_driver()
driver.options = {'lownoise_pct': '60', 'train_token_refiner': '0'}
net = FamilyLoRA(model, driver, device='cpu')
net.add_trainable(2, 2, blocks={'h3blk_1'})
trainable_names = [name for name, wrapped in net.wrapped.items() if 'lora' in wrapped.adapters]
assert net.parameters() and all(name.startswith('blocks.1.') for name in trainable_names)
assert not any('token_refiner' in name for name in trainable_names)
before = [p.clone() for p in net.parameters()]
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)
latents, cond = torch.randn(1,24,4,4), {'hidden_states':torch.randn(1,3,16)}
losses = []
for _ in range(4):
    optimizer.zero_grad()
    torch.manual_seed(81)
    loss, _ = driver.training_loss(model, latents, cond, torch.Generator().manual_seed(8))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    optimizer.step()
    losses.append(loss.item())
assert any(not torch.equal(a,b) for a,b in zip(before,net.parameters()))
assert losses[-1] < losses[0], losses
for key, expected in base.items():
    name, _, leaf = key.rpartition('.')
    module = model.get_submodule(name)
    actual = getattr(module.base if name in net.wrapped else module, leaf)
    assert torch.equal(expected, actual), key
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'person.safetensors'
    net.save(str(path), dtype=torch.float32, metadata={'modelspec.architecture':'MiniMax-H3'})
    validate_lora(path)
    assert net.add_file(str(path), 'training_adapter') > 0
    assert all(not p.requires_grad for wrapped in net.wrapped.values() if 'training_adapter' in wrapped.adapters
               for p in wrapped.adapters['training_adapter'].parameters())
    net.set_enabled('training_adapter', False)
    net.save(str(path), dtype=torch.float32)
    restored_model = MiniMaxH3DiT(cfg).to(torch.bfloat16).requires_grad_(False)
    restored_model.load_state_dict(base)
    restored = FamilyLoRA(restored_model, driver, device='cpu')
    restored.add_trainable(2,2,blocks={'h3blk_1'})
    assert restored.load_trainable(str(path)) > 0
    for key, tensor in net.state_dict(dtype=torch.float32).items():
        assert torch.equal(tensor,restored.state_dict(dtype=torch.float32)[key]), key
    torch.manual_seed(81)
    first,_ = driver.training_loss(model,latents,cond,torch.Generator().manual_seed(8))
    torch.manual_seed(81)
    second,_ = driver.training_loss(restored_model,latents,cond,torch.Generator().manual_seed(8))
    assert torch.allclose(first,second)
print('Tiny H3: finite flow loss and gradients; only selected LoRA weights change; frozen adapter excluded; Kohya save/load roundtrip', losses)

# Header validation must reject unrelated files before any checkpoint tensor is read.
from unittest.mock import patch
from minimax_trainer.runner import validate_models
from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder, MiniMaxH3VideoVAEDecoder
with torch.device('meta'):
    encoder = MiniMaxH3VideoVAEEncoder()
    decoder = MiniMaxH3VideoVAEDecoder()
vae_shapes = {name:list(p.shape) for name,p in encoder.named_parameters()}
all_vae_shapes = {**vae_shapes, **{name:list(p.shape) for name,p in decoder.named_parameters()}}
shapes = {'dit':{'video_patch_proj.weight':[5376,96],'condition_proj.weight':[5376,5120]},
          'te':{'model.embed_tokens.weight':[151936,5120]},'vae':vae_shapes}
class Header:
    def __init__(self, name, **kwargs): self.shapes=shapes[name]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def keys(self): return self.shapes.keys()
    def get_slice(self,name):
        class Slice:
            def get_shape(_): return self.shapes[name]
        return Slice()
    def get_tensor(self,*args): raise AssertionError('Preflight must not allocate checkpoint tensors')
job={'runtime':{'dit':'dit','text_encoder':'te','vae':'vae'},'options':{'training_base':'fl2va','sample_every':0}}
with patch('safetensors.safe_open',Header):
    validate_models(job)
    job['options']['sample_every']=5
    try:validate_models(job)
    except ValueError as error:assert 'Video VAE' in str(error)
    else:raise AssertionError('Encoder-only VAE must not pass when previews require the decoder')
    shapes['vae']=all_vae_shapes
    validate_models(job)
    shapes['te']['model.embed_tokens.weight']=[151936,4096]
    try:validate_models(job)
    except ValueError as error:assert '32B' in str(error)
    else:raise AssertionError('The Qwen Image encoder must not pass H3 validation')
print('H3 preflight checks model headers and encoder/decoder requirements without loading weights')
