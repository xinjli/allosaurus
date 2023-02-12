from allosaurus.am.module.frontend.wav2vec2_model import AudioPretrainingConfig, Wav2Vec2Config, Wav2Vec2Model
from allosaurus.config import allosaurus_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from allosaurus.am.module.frontend import ssl_config


def read_ssl_frontend(config, from_pretrained=False):

    model_cfg = ssl_config.xlsr_model_cfg
    task_cfg = ssl_config.xlsr_task_cfg
    model = Wav2Vec2Model(model_cfg)

    feature_selection = 'hidden_states'

    if from_pretrained:
        # from: https://huggingface.co/s3prl/converted_ckpts/resolve/main/xlsr_53_56k.pt
        ckpt_state = torch.load(allosaurus_config.model_path / 'xlsr_53_56k.pt', map_location="cpu")
        model.load_state_dict(ckpt_state["model_weight"])

    return SSLFrontend(model, task_cfg, config)


class Hook:
    def __init__(self, module_path, transform, unique_identifier=None):
        self.module_path = module_path
        self.transform = transform
        self.unique_identifier = unique_identifier or module_path
        self.handler = None

        assert isinstance(self.module_path, str)
        assert callable(self.transform)
        assert isinstance(self.unique_identifier, str)


class initHook(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for hook in instance.hooks:
            if hook.handler is None:
                instance._register_hook_handler(hook)
        return instance



class SSLFrontend(nn.Module):

    def __init__(self, model, task_cfg, config):
        super().__init__()

        self.model = model
        self.config = config
        self.task_cfg = task_cfg
        self.wav_normalize = task_cfg.normalize

        self.hooks = []
        self._hook_hiddens = []

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        module_name = "self.model.encoder.layers"
        for module_id in range(len(eval(module_name))):
            self.add_hook(
                f"{module_name}[{module_id}]",
                lambda input, output: input[0].transpose(0, 1),
            )
        self.add_hook("self.model.encoder", lambda input, output: output[0])

        def postprocess(xs):
            names, hiddens = zip(*xs)
            unpad_len = min([hidden.size(1) for hidden in hiddens])
            hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
            return list(zip(names, hiddens))

        self.hook_postprocess = postprocess

    def add_hook(self, *args, **kwargs):
        hook = Hook(*args, **kwargs)
        self._register_hook_handler(hook)
        self.hooks.append(hook)


    def _register_hook_handler(self, hook: Hook):
        module = eval(hook.module_path)
        if not isinstance(module, nn.Module):
            show(
                f"[UpstreamBase] - {hook.module_path} is not a valid nn.Module. Skip.",
                file=sys.stderr,
            )
            return

        if callable(hook.handler):
            show(
                f"[UpstreamBase] - Existing hook handler for {hook.unique_identifier} is found. Remove the existing one.",
                file=sys.stderr,
            )
            hook.handler.remove()

        def generate_hook_handler(hiddens, hook):
            def hook_handler(self, input, output):
                hiddens.append((hook.unique_identifier, hook.transform(input, output)))

            return hook_handler

        hook.handler = module.register_forward_hook(
            generate_hook_handler(self._hook_hiddens, hook)
        )


    def forward(self, input_tensor, input_lengths):

        #self.model.eval()

        wavs = [wav[:input_lengths[i]] for i, wav in enumerate(input_tensor)]

        device = wavs[0].device

        if self.wav_normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_padding_mask = ~torch.lt(
            torch.arange(max(input_lengths)).unsqueeze(0).to(device),
            input_lengths.unsqueeze(1),
        )

        padded_wav = pad_sequence(wavs, batch_first=True)
        # print(padded_wav)
        # print(wav_padding_mask)
        result = self.model.extract_features(padded_wav, wav_padding_mask)
        features = result['x']
        mask = result['padding_mask']
        #print(features.shape)
        #return results

        if len(self._hook_hiddens) > 0:
            if (
                result.get("_hidden_states_info") is not None
                or result.get("hidden_states") is not None
                or result.get("last_hidden_state") is not None
            ):
                show(
                    "[UpstreamBase] - If there are registered hooks, '_hidden_states_info', 'hidden_states', and "
                    "'last_hidden_state' are reserved and should not be included in child class's return dict.",
                    file=sys.stderr,
                )
                raise ValueError

            hook_hiddens = self._hook_hiddens.copy()
            self._hook_hiddens.clear()

            if callable(self.hook_postprocess):
                hook_hiddens = self.hook_postprocess(hook_hiddens)

            result["_hidden_states_info"], result["hidden_states"] = zip(*hook_hiddens)
            result["last_hidden_state"] = result["hidden_states"][-1]

            for layer_id, hidden_state in enumerate(result["hidden_states"]):
                result[f"hidden_state_{layer_id}"] = hidden_state

        hidden_states = torch.mean(torch.stack(result["hidden_states"], axis=0), axis=0)

        if mask is None:
             length = hidden_states.shape[1]
             return hidden_states, torch.ones((len(input_lengths), length), dtype=torch.bool).to(hidden_states.device)

        return hidden_states, ~mask