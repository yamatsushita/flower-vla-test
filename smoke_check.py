import sys, os
sys.path.insert(0, '.')
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
from omegaconf import OmegaConf
import torch

opt_cfg = OmegaConf.create({
    '_target_': 'torch.optim.AdamW',
    'transformer_weight_decay': 0.05,
    'learning_rate': 1e-4,
    'betas': [0.9, 0.95]
})
lr_cfg = OmegaConf.create({
    'lr_scheduler': {
        'init_lr': 1e-4, 'init_lr_scale': 0.1, 'final_lr_scale': 0.1,
        'total_steps': 35000, 'phase_ratio': '(0.05,0.15,0.80)', 'lr': 1e-4
    }
})

from flower.models.random_proj_vla import RandomProjVLA
print('Instantiating RandomProjVLA...')
m = RandomProjVLA(optimizer=opt_cfg, lr_scheduler=lr_cfg)
m.print_model_parameters()

B, T, C, H, W = 2, 1, 3, 112, 112
batch = {
    'rgb_obs': {
        'rgb_static':  torch.randn(B, T, C, H, W),
        'rgb_gripper': torch.randn(B, T, C, H, W),
    },
    'lang_text': ['push the red block', 'open the drawer'],
    'actions': torch.randn(B, 10, 7),
}
out = m.encode_observations(batch)
print('features shape:', out['features'].shape)

# Quick forward pass
noise = torch.randn(B, 10, 7)
cond = m.encode_observations(batch)
# skip full sample to stay fast
print('SMOKE TEST PASSED')
