from allosaurus.am.module.frontend.wav2vec2_model import AudioPretrainingConfig, Wav2Vec2Config, Wav2Vec2Model
from copy import deepcopy
from dataclasses import dataclass, is_dataclass


def merge_with_parent(dc: dataclass, cfg: dict):

    assert is_dataclass(dc)
    assert type(cfg) == dict
    cfg = deepcopy(cfg)

    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys:
                del cfg[k]

    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)



xlsr_model_cfg = merge_with_parent(Wav2Vec2Config, {'_name': 'wav2vec2',
 'extractor_mode': 'layer_norm',
 'encoder_layers': 24,
 'encoder_embed_dim': 1024,
 'encoder_ffn_embed_dim': 4096,
 'encoder_attention_heads': 16,
 'activation_fn': 'gelu',
 'dropout': 0.0,
 'attention_dropout': 0.0,
 'activation_dropout': 0.0,
 'encoder_layerdrop': 0.0,
 'dropout_input': 0.0,
 'dropout_features': 0.0,
 'final_dim': 768,
 'layer_norm_first': True,
 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2',
 'conv_bias': True,
 'logit_temp': 0.1,
 'quantize_targets': True,
 'quantize_input': False,
 'same_quantizer': False,
 'target_glu': False,
 'feature_grad_mult': 1.0,
 'latent_vars': 320,
 'latent_groups': 2,
 'latent_dim': 0,
 'mask_length': 10,
 'mask_prob': 0.65,
 'mask_selection': 'static',
 'mask_other': 0.0,
 'no_mask_overlap': False,
 'mask_min_space': 1,
 'mask_channel_length': 10,
 'mask_channel_prob': 0.0,
 'mask_channel_selection': 'static',
 'mask_channel_other': 0.0,
 'no_mask_channel_overlap': False,
 'mask_channel_min_space': 1,
 'num_negatives': 100,
 'negatives_from_everywhere': False,
 'cross_sample_negatives': 0,
 'codebook_negatives': 0,
 'conv_pos': 128,
 'conv_pos_groups': 16,
 'latent_temp': [2.0, 0.1, 0.999995]
})


xlsr_task_cfg = merge_with_parent(AudioPretrainingConfig, {'_name': 'audio_pretraining',
 'data': '/private/home/aconneau/projects/XLSR/MLS/53bis/',
 'labels': None,
 'sample_rate': 16000,
 'normalize': True,
 'enable_padding': False,
 'max_sample_size': 320000,
 'min_sample_size': 32000
})