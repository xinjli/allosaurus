from allosaurus.acoustic_model.utils import *
from pathlib import Path
from allosaurus.audio import read_audio
from allosaurus.preprocess_model.factory import read_preprocess_model
from allosaurus.acoustic_model.factory import read_acoustic_model
from allosaurus.language_model.factory import read_language_model
from allosaurus.bin.download_model import download_model
from allosaurus.model import resolve_model_name, get_all_models
from argparse import Namespace
from io import BytesIO

def read_recognizer(inference_config_or_name='latest', alt_model_path=None):
    if alt_model_path:
        if not alt_model_path.exists():
            download_model(inference_config_or_name, alt_model_path)
    # download specified model automatically if no model exists
    if len(get_all_models()) == 0:
        download_model('latest', alt_model_path)

    # create default config if input is the model's name
    if isinstance(inference_config_or_name, str):
        model_name = resolve_model_name(inference_config_or_name, alt_model_path)
        inference_config = Namespace(model=model_name, device_id=-1, lang='ipa', approximate=False, prior=None)
    else:
        assert isinstance(inference_config_or_name, Namespace)
        inference_config = inference_config_or_name

    if alt_model_path:
        model_path = alt_model_path / inference_config.model
    else:
        model_path = Path(__file__).parent / 'pretrained' / inference_config.model

    if inference_config.model == 'latest' and not model_path.exists():
        download_model(inference_config, alt_model_path)

    assert model_path.exists(), f"{inference_config.model} is not a valid model"

    # preprocess(audio) -> features etc..
    preprocess_model = read_preprocess_model(model_path, inference_config)

    # acoustic(features) -> logits
    acoustic_model = read_acoustic_model(model_path, inference_config)

    # language(logits) -> phones
    language_model = read_language_model(model_path, inference_config)

    return Recognizer(preprocess_model, acoustic_model, language_model, inference_config)

class Recognizer:

    def __init__(self, preprocess_model, acoustic_model, language_model, config):

        self.preprocess_model = preprocess_model
        self.acoustic_model = acoustic_model
        self.language_model = language_model
        self.config = config

    def is_available(self, lang_id):
        # check whether this lang id is available

        return self.lm.inventory.is_available(lang_id)

    def recognize(self, filename, lang_id='ipa', topk=1, emit=1.0, timestamp=False):
        # recognize a single file

        # filename check (skipping for BytesIO objects)
        if not isinstance(filename, BytesIO):
            assert str(filename).endswith('.wav'), "only wave file is supported in allosaurus"

        # load wav audio
        audio = read_audio(filename)

        # extract feature
        feat = self.preprocess_model.compute(audio)

        # add batch dim
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        tensor_batch_feat, tensor_batch_feat_len = move_to_tensor([feats, feat_len], self.config.device_id)

        tensor_batch_lprobs = self.acoustic_model(tensor_batch_feat, tensor_batch_feat_len)

        if self.config.device_id >= 0:
            batch_lprobs = tensor_batch_lprobs.cpu().detach().numpy()
        else:
            batch_lprobs = tensor_batch_lprobs.detach().numpy()

        token = self.language_model.compute(batch_lprobs[0], lang_id, topk, emit=emit, timestamp=timestamp)
        return token
