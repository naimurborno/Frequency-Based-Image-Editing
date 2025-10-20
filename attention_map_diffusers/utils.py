import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)

from .modules import *


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):

            timestep = module.processor.timestep

            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach \
                else module.processor.attn_map
            
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, JointAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, FluxAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))
    
    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model


# TODO: implement
# def replace_call_method_for_sana(model):
#     if model.__class__.__name__ == 'SanaTransformer2DModel':
#         from diffusers.models.transformers import SanaTransformer2DModel
#         model.forward = SanaTransformer2DModelForward.__get__(model, SanaTransformer2DModel)

#     for name, layer in model.named_children():
        
#         if layer.__class__.__name__ == 'SanaTransformerBlock':
#             from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
#             layer.forward = SanaTransformerBlockForward.__get__(layer, SanaTransformerBlock)
        
#         replace_call_method_for_sana(layer)
    
#     return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        from diffusers.models.transformers import SD3Transformer2DModel
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            from diffusers.models.attention import JointTransformerBlock
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)
        
        replace_call_method_for_flux(layer)
    
    return model


def init_pipeline(pipeline):
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

        # TODO: implement
        # elif pipeline.transformer.__class__.__name__ == 'SanaTransformer2DModel':
        #     from diffusers import SanaPipeline
        #     SanaPipeline.__call__ == SanaPipeline_call
        #     pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn2')
        #     pipeline.transformer = replace_call_method_for_sana(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)


    return pipeline


def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        token, startofword = process_token(token, startofword)
        to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))


def get_attention_map_for_timestep(attn_maps, timestep, unconditional=False):
    """
    Compute a single aggregated attention map for a given denoising timestep.

    Parameters
    ----------
    attn_maps : dict
        Nested dict of attention maps: {timestep: {layer_name: attn_tensor}}
        Each attn_tensor: [batch, num_heads, height, width, num_tokens]
    timestep : int
        Specific timestep to compute the aggregated attention for.
    unconditional : bool, optional
        If True, keep only the conditional half of the batch (default: False)

    Returns
    -------
    torch.Tensor
        Aggregated attention map of shape [batch, height, width],
        normalized to [0, 1].
    """
    if timestep not in attn_maps:
        raise ValueError(f"Timestep {timestep} not found in attn_maps. Available: {list(attn_maps.keys())}")

    timestep_layers = attn_maps[timestep]
    total_map = None
    num_layers = 0

    for layer_name, attn_tensor in timestep_layers.items():
        # attn_tensor shape: [batch, heads, h, w, tokens]
        if unconditional:
            attn_tensor = attn_tensor.chunk(2)[1]  # keep conditional half

        # Combine across heads and tokens
        attn_mean = attn_tensor.mean(dim=1).mean(dim=-1)  # [batch, h, w]

        # Initialize accumulator
        if total_map is None:
            total_map = torch.zeros_like(attn_mean)

        # Resize to match base resolution if needed
        if attn_mean.shape[-2:] != total_map.shape[-2:]:
            attn_mean = F.interpolate(attn_mean.unsqueeze(1),
                                      size=total_map.shape[-2:],
                                      mode="bilinear",
                                      align_corners=False).squeeze(1)

        total_map += attn_mean
        num_layers += 1

    # Average over all layers
    total_map /= max(1, num_layers)

    # Normalize to [0, 1]
    min_val, max_val = total_map.min(), total_map.max()
    if (max_val - min_val) > 1e-8:
        total_map = (total_map - min_val) / (max_val - min_val)

    return total_map  # [batch, height, width]