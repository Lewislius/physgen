"""Fix only physical diffusion features to their50-step inference-grid mean."""
import torch
from physgen_native.generator import sinusoidal_embedding_1d

def install_time_mean(model):
    device=next(model.parameters()).device
    grid=torch.linspace(1,0,model.cfg['inference']['steps']+1,device=device)[:-1]
    shift=model.cfg['inference']['shift'];sigmas=shift*grid/(1+(shift-1)*grid)
    features=sinusoidal_embedding_1d(model.wan.freq_dim,sigmas*1000).float().mean(0,keepdim=True)
    def replace_physical_features(module,arguments,keywords):
        if len(arguments)!=7 or arguments[-1].shape!=features.shape:
            raise ValueError('Unexpected physical fusion argument contract')
        return (*arguments[:-1],features),keywords
    handles=[fusion.register_forward_pre_hook(replace_physical_features,with_kwargs=True) for fusion in model.fusions.values()]
    return handles,dict(kind='mean_of_official_sinusoidal_features_on_exact_inference_grid',points=len(sigmas),
        native_Wan_diffusion_time='unchanged',video_presentation_time='unchanged',both_CFG_arms=True,
        physical_queries_and_values='same learned projection, constant mean input; no weights changed')
