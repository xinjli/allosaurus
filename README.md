# allosaurus_private
Allosaurus is a pretrained universal phone recognizer. 

It can be used to recognize narrow phones in more than 2000 languages.

![Architecture](arch.png?raw=true "Architecture")

## Install
Allosaurus is available from pip
```bash
pip install allosaurus
```
 
You can also clone this repository and install 
```bash
python setup.py install
```

## Tutorial
The basic usage is as follows:
 
```bash
python -m allosaurus.run [--lang <language name>] [--model <model name>] [--device_id <gpu_id>] -i <audio>
```
It will recognize the narrow phones in the audio file and print them in stdout.

Only audio argument is mandatory, other options can ignored. Please refer to following sections for their details. 

### Audio
Audio should be a single input audio file

* It should be a wav file. If the audio is not in the wav format, please convert your audio to a wav format using sox or ffmpeg in advance.

* The sampling rate can be arbitrary, we will automatically resample them based on models' requirements.

* We assume the audio is a mono-channel audio.

### Language
The `lang` option is the language id. It is to specify the phone inventory you want to use.
The default option is `ipa` which tells the recognizer to use the the entire inventory (around 230 phones).

Generally, specifying the language inventory can improve your recognition accuracy.

You can check the full language list with the following command. The number of available languages is around 2000. 
```bash
python -m allosaurus.list_lang
```

To check language's inventory you can use following command
```bash
python -m allosaurus.list_phone [--lang <language name>]
```

For example,
```bash
# to get English phone inventory
# ['a', 'aː', 'b', 'd', 'd̠', 'e', 'eː', 'e̞', 'f', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'l', 'm', 'n', 'o', 'oː', 'p', 'pʰ', 'r', 's', 't', 'tʰ', 't̠', 'u', 'uː', 'v', 'w', 'x', 'z', 'æ', 'ð', 'øː', 'ŋ', 'ɐ', 'ɐː', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔː', 'ɘ', 'ə', 'əː', 'ɛ', 'ɛː', 'ɜː', 'ɡ', 'ɪ', 'ɪ̯', 'ɯ', 'ɵː', 'ɹ', 'ɻ', 'ʃ', 'ʉ', 'ʉː', 'ʊ', 'ʌ', 'ʍ', 'ʒ', 'ʔ', 'θ']
python -m allosaurus.list_phone --lang english

# you can also skip lang option to get all inventory
#['I', 'a', 'aː', 'ã', 'ă', 'b', 'bʲ', 'bʲj', 'bʷ', 'bʼ', 'bː', 'b̞', 'b̤', 'b̥', 'c', 'd', 'dʒ', 'dʲ', 'dː', 'd̚', 'd̥', 'd̪', 'd̯', 'd͡z', 'd͡ʑ', 'd͡ʒ', 'd͡ʒː', 'd͡ʒ̤', 'e', 'eː', 'e̞', 'f', 'fʲ', 'fʷ', 'fː', 'g', 'gʲ', 'gʲj', 'gʷ', 'gː', 'h', 'hʷ', 'i', 'ij', 'iː', 'i̞', 'i̥', 'i̯', 'j', 'k', 'kx', 'kʰ', 'kʲ', 'kʲj', 'kʷ', 'kʷʼ', 'kʼ', 'kː', 'k̟ʲ', 'k̟̚', 'k͡p̚', 'l', 'lʲ', 'lː', 'l̪', 'm', 'mʲ', 'mʲj', 'mʷ', 'mː', 'n', 'nj', 'nʲ', 'nː', 'n̪', 'n̺', 'o', 'oː', 'o̞', 'o̥', 'p', 'pf', 'pʰ', 'pʲ', 'pʲj', 'pʷ', 'pʷʼ', 'pʼ', 'pː', 'p̚', 'q', 'r', 'rː', 's', 'sʲ', 'sʼ', 'sː', 's̪', 't', 'ts', 'tsʰ', 'tɕ', 'tɕʰ', 'tʂ', 'tʂʰ', 'tʃ', 'tʰ', 'tʲ', 'tʷʼ', 'tʼ', 'tː', 't̚', 't̪', 't̪ʰ', 't̪̚', 't͡s', 't͡sʼ', 't͡ɕ', 't͡ɬ', 't͡ʃ', 't͡ʃʲ', 't͡ʃʼ', 't͡ʃː', 'u', 'uə', 'uː', 'u͡w', 'v', 'vʲ', 'vʷ', 'vː', 'v̞', 'v̞ʲ', 'w', 'x', 'x̟ʲ', 'y', 'z', 'zj', 'zʲ', 'z̪', 'ä', 'æ', 'ç', 'çj', 'ð', 'ø', 'ŋ', 'ŋ̟', 'ŋ͡m', 'œ', 'œ̃', 'ɐ', 'ɐ̞', 'ɑ', 'ɑ̱', 'ɒ', 'ɓ', 'ɔ', 'ɔ̃', 'ɕ', 'ɕː', 'ɖ̤', 'ɗ', 'ə', 'ɛ', 'ɛ̃', 'ɟ', 'ɡ', 'ɡʲ', 'ɡ̤', 'ɡ̥', 'ɣ', 'ɣj', 'ɤ', 'ɤɐ̞', 'ɤ̆', 'ɥ', 'ɦ', 'ɨ', 'ɪ', 'ɫ', 'ɯ', 'ɯ̟', 'ɯ̥', 'ɰ', 'ɱ', 'ɲ', 'ɳ', 'ɴ', 'ɵ', 'ɸ', 'ɹ', 'ɹ̩', 'ɻ', 'ɻ̩', 'ɽ', 'ɾ', 'ɾj', 'ɾʲ', 'ɾ̠', 'ʀ', 'ʁ', 'ʁ̝', 'ʂ', 'ʃ', 'ʃʲː', 'ʃ͡ɣ', 'ʈ', 'ʉ̞', 'ʊ', 'ʋ', 'ʋʲ', 'ʌ', 'ʎ', 'ʏ', 'ʐ', 'ʑ', 'ʒ', 'ʒ͡ɣ', 'ʔ', 'ʝ', 'ː', 'β', 'β̞', 'θ', 'χ', 'ә', 'ḁ']
python -m allosaurus.list_phone
```


### Model
The `model` option is to select model for inference.
The default option is `latest`, it is pointing to the latest model you downloaded. 
It will automatically download the latest model during your first inference if you do not have any local models. 


We intend to train new models and continuously release them. The update might include both acoustic model binary files and phone inventory. 
Typically, the model's name indicates its training date, so usually a higher model id should be expected to perform better.

To download a new model, you can run following command.

```bash
python -m allosaurus.download <model>
``` 

Current available models are the followings

| Model | Description |
| --- | --- |
| `200529` | This is the `latest` model |

If you do not know the model name, 
you can just use `latest` as model's name and it will automatically download the latest model.

We note that updating to a new model will not delete the original models. All the models will be stored under `pretrained` directory where you installed allosaurus.
You might want to fix your model to get consistent results during one experiment.  

To see which models are available in your local environment, you can check with the following command
```bash
python -m allosaurus.list_model
```

### Device
`device_id` controls which device to run the inference.

By default, device_id will be -1, which indicates the model will only use CPUs.  

However, if you have GPU, You can use them for inference by specifying device_id to a single GPU id. (note that multiple GPU inference is not supported)


## Acknowledgements
This work uses part of the following codes and inventories.
* AlloVera: https://github.com/dmort27/allovera
* Phoible: https://github.com/phoible/dev
* python_speech_features: https://github.com/jameslyons/python_speech_features
* fairseq: https://github.com/pytorch/fairseq

In particular, we heavily used AlloVera and Phoible to build this model's phone inventory.  

## Reference
Please cite the following paper if you use code in  your work.

If you have any advice or suggestions, please feel free to send email to me (xinjianl [at] cs.cmu.edu) or submit an issue in this repo. Thanks!    

```BibTex
@inproceedings{li2020universal,
  title={Universal phone recognition with a multilingual allophone system},
  author={Li, Xinjian and Dalmia, Siddharth and Li, Juncheng and Lee, Matthew and Littell, Patrick and Yao, Jiali and Anastasopoulos, Antonios and Mortensen, David R and Neubig, Graham and Black, Alan W and others},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8249--8253},
  year={2020},
  organization={IEEE}
}
```