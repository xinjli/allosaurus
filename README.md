# Allosaurus

![CI Test](https://github.com/xinjli/allosaurus/actions/workflows/python.yml/badge.svg)

Allosaurus is a pretrained universal phone recognizer. It can be used to recognize phones in more than 2000 languages.

This tool is based on our ICASSP 2020 work [Universal Phone Recognition with a Multilingual Allophone System](https://arxiv.org/pdf/2002.11800.pdf)

![Architecture](arch.png?raw=true "Architecture")

## Get Started

### Install
Allosaurus is available from pip
```bash
pip install allosaurus
```
 
You can also clone this repository and install 
```bash
python setup.py install
```

### Quick start
The basic usage is pretty simple, your input is an wav audio file and output is a sequence of phones.

```bash
python -m allosaurus.run  -i <audio>
```

For example, you can try using the attached sample file in this repository. Guess what's in this audio file :)
```bash
python -m allosaurus.run -i sample.wav
æ l u s ɔ ɹ s
```

You can also use allosaurus directly in python
```python
from allosaurus.app import read_recognizer

# load your model
model = read_recognizer()

# run inference -> æ l u s ɔ ɹ s
model.recognize('sample.wav')
```

For full features and details, please refer to the following sections.

## Inference 
The command line interface is as follows:
 
```bash
python -m allosaurus.run [--lang <language name>] [--model <model name>] [--device_id <gpu_id>] [--output <output_file>] [--topk <int>] -i <audio file/directory>
```
It will recognize the narrow phones in the audio file(s).
Only the input argument is mandatory, other options can ignored. Please refer to following sections for their details. 

There is also a simple python interface as follows:
```python
from allosaurus.app import read_recognizer

# load your model by the <model name>, will use 'latest' if left empty
model = read_recognizer(model)

# run inference on <audio_file> with <lang>, lang will be 'ipa' if left empty
model.recognize(audio_file, lang)
```

The details of arguments in both interface are as follows:
### Input
The input can be a single file or a directory containing multiple audio files.

If the input is a single file, it will output only the phone sequence; if the input is a directory, it will output both the file name and phone sequence, results will be sorted by file names. 

The audio file(s) should be in the following format:

* It should be a wav file. If the audio is not in the wav format, please convert your audio to a wav format using sox or ffmpeg in advance.

* The sampling rate can be arbitrary, we will automatically resample them based on models' requirements.

* We assume the audio is a mono-channel audio.

### Output
The output is by default stdout (i.e. it will print all results to terminal).

If you specify a file as the output, then all output will be directed to that file.

### Language
The `lang` option is the language id. It is to specify the phone inventory you want to use.
The default option is `ipa` which tells the recognizer to use the the entire inventory (around 230 phones).

Generally, specifying the language inventory can improve your recognition accuracy.

You can check the full language list with the following command. The number of available languages is around 2000. 
```bash
python -m allosaurus.bin.list_lang
```

To check language's inventory you can use following command
```bash
python -m allosaurus.bin.list_phone [--lang <language name>]
```

For example,
```bash
# to get English phone inventory
# ['a', 'aː', 'b', 'd', 'd̠', 'e', 'eː', 'e̞', 'f', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'l', 'm', 'n', 'o', 'oː', 'p', 'pʰ', 'r', 's', 't', 'tʰ', 't̠', 'u', 'uː', 'v', 'w', 'x', 'z', 'æ', 'ð', 'øː', 'ŋ', 'ɐ', 'ɐː', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔː', 'ɘ', 'ə', 'əː', 'ɛ', 'ɛː', 'ɜː', 'ɡ', 'ɪ', 'ɪ̯', 'ɯ', 'ɵː', 'ɹ', 'ɻ', 'ʃ', 'ʉ', 'ʉː', 'ʊ', 'ʌ', 'ʍ', 'ʒ', 'ʔ', 'θ']
python -m allosaurus.bin.list_phone --lang eng

# you can also skip lang option to get all inventory
#['I', 'a', 'aː', 'ã', 'ă', 'b', 'bʲ', 'bʲj', 'bʷ', 'bʼ', 'bː', 'b̞', 'b̤', 'b̥', 'c', 'd', 'dʒ', 'dʲ', 'dː', 'd̚', 'd̥', 'd̪', 'd̯', 'd͡z', 'd͡ʑ', 'd͡ʒ', 'd͡ʒː', 'd͡ʒ̤', 'e', 'eː', 'e̞', 'f', 'fʲ', 'fʷ', 'fː', 'g', 'gʲ', 'gʲj', 'gʷ', 'gː', 'h', 'hʷ', 'i', 'ij', 'iː', 'i̞', 'i̥', 'i̯', 'j', 'k', 'kx', 'kʰ', 'kʲ', 'kʲj', 'kʷ', 'kʷʼ', 'kʼ', 'kː', 'k̟ʲ', 'k̟̚', 'k͡p̚', 'l', 'lʲ', 'lː', 'l̪', 'm', 'mʲ', 'mʲj', 'mʷ', 'mː', 'n', 'nj', 'nʲ', 'nː', 'n̪', 'n̺', 'o', 'oː', 'o̞', 'o̥', 'p', 'pf', 'pʰ', 'pʲ', 'pʲj', 'pʷ', 'pʷʼ', 'pʼ', 'pː', 'p̚', 'q', 'r', 'rː', 's', 'sʲ', 'sʼ', 'sː', 's̪', 't', 'ts', 'tsʰ', 'tɕ', 'tɕʰ', 'tʂ', 'tʂʰ', 'tʃ', 'tʰ', 'tʲ', 'tʷʼ', 'tʼ', 'tː', 't̚', 't̪', 't̪ʰ', 't̪̚', 't͡s', 't͡sʼ', 't͡ɕ', 't͡ɬ', 't͡ʃ', 't͡ʃʲ', 't͡ʃʼ', 't͡ʃː', 'u', 'uə', 'uː', 'u͡w', 'v', 'vʲ', 'vʷ', 'vː', 'v̞', 'v̞ʲ', 'w', 'x', 'x̟ʲ', 'y', 'z', 'zj', 'zʲ', 'z̪', 'ä', 'æ', 'ç', 'çj', 'ð', 'ø', 'ŋ', 'ŋ̟', 'ŋ͡m', 'œ', 'œ̃', 'ɐ', 'ɐ̞', 'ɑ', 'ɑ̱', 'ɒ', 'ɓ', 'ɔ', 'ɔ̃', 'ɕ', 'ɕː', 'ɖ̤', 'ɗ', 'ə', 'ɛ', 'ɛ̃', 'ɟ', 'ɡ', 'ɡʲ', 'ɡ̤', 'ɡ̥', 'ɣ', 'ɣj', 'ɤ', 'ɤɐ̞', 'ɤ̆', 'ɥ', 'ɦ', 'ɨ', 'ɪ', 'ɫ', 'ɯ', 'ɯ̟', 'ɯ̥', 'ɰ', 'ɱ', 'ɲ', 'ɳ', 'ɴ', 'ɵ', 'ɸ', 'ɹ', 'ɹ̩', 'ɻ', 'ɻ̩', 'ɽ', 'ɾ', 'ɾj', 'ɾʲ', 'ɾ̠', 'ʀ', 'ʁ', 'ʁ̝', 'ʂ', 'ʃ', 'ʃʲː', 'ʃ͡ɣ', 'ʈ', 'ʉ̞', 'ʊ', 'ʋ', 'ʋʲ', 'ʌ', 'ʎ', 'ʏ', 'ʐ', 'ʑ', 'ʒ', 'ʒ͡ɣ', 'ʔ', 'ʝ', 'ː', 'β', 'β̞', 'θ', 'χ', 'ә', 'ḁ']
python -m allosaurus.bin.list_phone
```

### Model
The `model` option is to select model for inference.
The default option is `latest`, it is pointing to the latest model you downloaded. 
It will automatically download the latest model during your first inference if you do not have any local models. 


We intend to train new models and continuously release them. The update might include both acoustic model binary files and phone inventory. 
Typically, the model's name indicates its training date, so usually a higher model id should be expected to perform better.

To download a new model, you can run following command.

```bash
python -m allosaurus.bin.download_model -m <model>
```

If you do not know the model name, 
you can just use `latest` as model's name and it will automatically download the latest model.

We note that updating to a new model will not delete the original models. All the models will be stored under `pretrained` directory where you installed allosaurus.
You might want to fix your model to get consistent results during one experiment.  

To see which models are available in your local environment, you can check with the following command
```bash
python -m allosaurus.bin.list_model
```

To delete a model, you can use the following command. This might be useful when you are fine-tuning your models mentioned later. 
```bash
python -m allosaurus.bin.remove_model
```

Current available models are the followings

#### Language Independent Model (Universal Model) 
The universal models predict language-independent phones and covers many languages. This is the default model allosaurus will try to download and use.
 If you cannot find your language on the language dependent models, please use this universal model instead. 

| Model | Target Language | Description |
| --- | --- | --- |
| `uni2005` | Universal | This is the `latest` model (previously named as `200529`) |

#### Language Dependent Model
We are planning to deliver language-dependent models for some widely-used languages. The models here are trained with the target language specifically.
It should perform much better than the universal model for the target language. Those models will not be downloaded automatically. Please use the `download_model` command above to download, and use `--model` flag during inference. 

| Model | Target Language | Description |
| --- | --- | --- |
| `eng2102` | English (eng) | English only model|

### Device
`device_id` controls which device to run the inference.

By default, device_id will be -1, which indicates the model will only use CPUs.  

However, if you have GPU, You can use them for inference by specifying device_id to a single GPU id. (note that multiple GPU inference is not supported)

### Timestamp
You can retrieve an approximate timestamp for each recognized phone by using `timestamp` argument.

```bash 
python -m allosaurus.run --timestamp=True -i sample.wav 
0.210 0.045 æ
0.390 0.045 l
0.450 0.045 u
0.540 0.045 s
0.630 0.045 ɔ
0.720 0.045 ɹ
0.870 0.045 s
```

The format here in each line is `start_timestamp duration phone` where the `start_timestamp` and `duration` are shown in seconds.

Note that the current timestamp is only an approximation. It is provided by the CTC model, which might not be accurate in some cases due to its nature. 

The same interface is also available in python as follows:

```python
model = read_recognizer()
model.recognize('./sample.wav', timestamp=True)
```


### Top K
Sometimes generating more phones might be helpful. Specifying the top-k arg will generate k phones at each emitting frame. Default is 1.
 
```bash
# default topk is 1
python -m allosaurus.run -i sample.wav
æ l u s ɔ ɹ s

# output top 5 probable phones at emitting frame, "|" is used to delimit frames (no delimiter when topk=1)
# probability is attached for each phone, the left most phone is the most probable phone 
# <blk> is blank which can be ignored.
python -m allosaurus.run -i sample.wav --topk=5
æ (0.577) ɛ (0.128) ɒ (0.103) a (0.045) ə (0.021) | l (0.754) l̪ (0.196) lː (0.018) ʁ (0.007) ʀ (0.006) | u (0.233) ɨ (0.218) uː (0.104) ɤ (0.070) ɪ (0.066) | s (0.301) <blk> (0.298) z (0.118) s̪ (0.084) sː (0.046) | ɔ (0.454) ɑ (0.251) <blk> (0.105) ɹ̩ (0.062) uə (0.035) | ɹ (0.867) ɾ (0.067) <blk> (0.024) l̪ (0.018) r (0.015) | s (0.740) z (0.191) s̪ (0.039) zʲ (0.009) sː (0.003)
```

### Phone Emission
You can tell the model to emit more phones or less phones by changing the `--emit` or `-e` argument.

```bash
# default emit is 1.0
python -m allosaurus.run -i sample.wav 
æ l u s ɔ ɹ s

# emit more phones when emit > 1
python -m allosaurus.run -e 1.2 -i sample.wav 
æ l u s f h ɔ ɹ s

# emit less phones when emit < 1
python -m allosaurus.run -e 0.8 -i sample.wav 
æ l u ɹ s
```


## Inventory Customization
The default phone inventory might not be the inventory you would like to use, so we provide several commands here for you to customize your own inventory.

We have mentioned that you can check your current (default) inventory with following command.
```bash
python -m allosaurus.bin.list_phone --lang <language name>
```

The current phone inventory file can be dumped into a file
```bash
# dump the phone file
python -m allosaurus.bin.write_phone --lang <language name> --output <a path to save this file>
```

If you take a look at the file, it is just a simple format where each line represents a single phone. For example, the following one is the English file 
```
a
aː
b
d
...
```

You can customize this file to add or delete IPAs you would like. Each line should only contain one IPA phone without any space. It might be easier to debug later if IPAs are sorted, but it is not required.

Next, update your model's inventory by the following command
```bash
python -m allosaurus.bin.update_phone --lang <language name> --input <the file you customized)
``` 

Then the file has been registered in your model, run the list_phone command again and you could see that it is now using your updated inventory
```bash
python -m allosaurus.bin.list_phone --lang <language name>
```

Now, if you run the inference again, you could also see the results also reflect your updated inventory.

Even after your update, you can easily switch back to the original inventory. In this case, your updated file will be deleted.
```bash
python -m allosaurus.bin.restore_phone --lang <language name>
```

## Prior Customization
You can also change the results by adjusting the prior probability for each phone. 
This can help you reduce the unwanted phones or increase the wanted phones.

For example, in the sample file, we get the output 
```bash
æ l u s ɔ ɹ s
```
Suppose you think the first phone is wrong, and would like to reduce the probability of this phone, you can create a new file ```prior.txt``` as follows
```text
æ -10.0
```

The file can contain multiple lines and each line has information for each phone. 
The first field is your target phone and the second field is the log-based score to adjust your probability. Positive score means you want to boost its prediction, 
negative score will suppress its prediction. In this case, we can get a new result

```bash
python -m allosaurus.run -i=sample.wav --lang=eng --prior=prior.txt 
ɛ l u s ɔ ɹ s
```

where you can see ```æ``` is suppressed and another vowel ```ɛ``` replaced it.

Another application of prior is to change the number of total output phones. You might want more phones outputs or less phones outputs.
In this case, you can change the score for the ```<blk>``` which corresponds to the silence phone.

A positive ```<blk>``` score will add more silence, therefore decrease the number of outputs, similarly, a negative ```<blk>``` will increase the outputs. The following example illustrates this.
```text

# <blk> 1.0
python -m allosaurus.run -i=sample.wav --lang=eng --prior=prior.txt 
æ l u ɔ ɹ s

# <blk> -1.0
$ python -m allosaurus.run -i=sample.wav --lang=eng --prior=prior.txt 
æ l u s f ɔ ɹ s
```

The first example reduces one phone and the second example adds a new phone.

## Fine-Tuning
We notice that the pretrained models might not be accurate enough for some languages, 
so we also provide a fine-tuning tool here to allow users to further improve their model by adapting to their data.
Currently, it is only limited to fine-tuned with one language.

### Prepare
To fine-tune your data, you need to prepare audio files and their transcriptions.
First, please create one data directory (name can be arbitrary), inside the data directory, create a `train` directory and a `validate` directory.
Obviously, the `train` directory will contain your training set, and the `validate` directory will be the validation set. 

Each directory should contain the following two files:
* `wave`: this is a file associating utterance with its corresponding audios 
* `text`: this is a file associating utterance with its phones.

#### wave
`wave` is a txt file mapping each utterance to your wav files. Each line should be prepared as follows:
```txt
utt_id /path/to/your/audio.wav
```

Here `utt_id` denotes the utterance id, it can be an arbitrary string as long as it is unique in your dataset.
The `audio.wav` is your wav file as mentioned above, it should be a mono-channel wav format, but sampling rate can be arbitrary (the tool would automatically resample if necessary)
The delimiter used here is space.

To get the best fine-tuning results, each audio file should not be very long. We recommend to keep each utterance shorter than 10 seconds. 
  
#### text
`text` is another txt file mapping each utterance to your transcription. Each line should be prepared as follows
```txt
utt_id phone1 phone2 ...
```

Here `utt_id` is again the utterance id and should match with the corresponding wav file.
The phone sequences came after utterance id is your phonetic transcriptions of the wav file.
The phones here should be restricted to the phone inventory of your target language.
Please make sure all your phones are already registered in your target language by the `list_phone` command

### Feature Extraction
Next, we will extract feature from both the `wave` file and `text` file.
We assume that you already prepared `wave` file and `text` file in BOTH `train` directory and `validate` directory


#### Audio Feature
To prepare the audio features, run the following command on both your `train` directory and `validate` directory. 

```bash
# command to prepare audio features
python -m allosaurus.bin.prep_feat --model=some_pretrained_model --path=/path/to/your/directory (train or validate)
```

The `path` should be pointing to the train or the validate directory, the `model` should be pointing to your traget pretrained model. If unspecified, it will use the latest model. 
It will generate three files `feat.scp`, `feat.ark` and `shape`.
 
* The first one is an file indexing each utterance into a offset of the second file.

* The second file is a binary file containing all audio features. 

* The third one contains the feature dimension information  

If you are curious, the `scp` and `ark` formats are standard file formats used in Kaldi.

### Text Feature
To prepare the text features, run the following command again on both your `train` directory and `validate` directory. 

```bash
# command to prepare token
python -m allosaurus.bin.prep_token --model=<some_pretrained_model> --lang=<your_target_language_id> --path=/path/to/your/directory (train or validate)
```

The `path` and `model` should be the same as the previous command. The `lang` is the 3 character ISO language id of this dataset.
Note that you should already verify the the phone inventory of this language id contains all of your phone transcriptions.
Otherwise, the extraction here might fail.

After this command, it will generate a file called `token` which maps each utterance to the phone id sequences.


### Training
Next, we can start fine-tuning our model with the dataset we just prepared. The fine-tuning command is very simple.

```bash
# command to fine_tune your data
python -m allosaurus.bin.adapt_model --pretrained_model=<pretrained_model> --new_model=<your_new_model> --path=/path/to/your/data/directory --lang=<your_target_language_id> --device_id=<device_id> --epoch=<epoch>
```
There are couple of other optional arguments available here, but we describe the required arguments. 
 
* `pretrained_model` should be the same model you specified before in the `prep_token` and `prep_feat`.
 
* `new_model` can be an arbitrary model name (Actually, it might be easier to manage if you give each model the same format as the pretrained model (i.e. YYMMDD))

* The `path` should be pointing to the parent directory of your `train` and `validate` directories.

* The `lang` is the language id you specified in `prep_token`

* The `device_id` is the GPU id for fine-tuning, if you do not have any GPU, use -1 as device_id. Multiple GPU is not supported.

* `epoch` is the number of your training epoch


During the training, it will show some information such as loss and phone error rate for both your training set and validation set.
After each epoch, the model would be evaluated with the validation set and would save this checkpoint if its validation phone error rate is better than previous ones.
After the specified `epoch` has finished, the fine-tuning process will end and the new model should be available.

### Testing
After your training process, the new model should be available in your model list. use the `list_model` command to check your new model is available now

```bash
# command to check all your models
python -m allosaurus.bin.list_model
```

If it is available, then this new model can be used in the same style as any other pretrained models. 
Just run the inference to use your new model. 
```bash
python -m allosaurus.run --lang <language id> --model <your new model> --device_id <gpu_id> -i <audio>
```



## Acknowledgements
This work uses part of the following codes and inventories. In particular, we heavily used AlloVera and Phoible to build this model's phone inventory.  

* [AlloVera](https://github.com/dmort27/allovera): For pretraining the model with correct allophone mappings 
* [Phoible](https://github.com/phoible/dev): For language specific phone inventory 
* [python_speech_features](https://github.com/jameslyons/python_speech_features): For mfcc, filter bank feature extraction
* [fairseq](https://github.com/pytorch/fairseq): For some utilities
* [kaldi_io](https://github.com/vesis84/kaldi-io-for-python): For kaldi scp, ark reader and writer

## Reference
Please cite the following paper if you use code in  your work.

If you have any advice or suggestions, please feel free to send email to me (xinjianl [at] cs.cmu.edu) or submit an issue in this repo. Thanks!    

```BibTex
@inproceedings{li2020universal,
  title={Universal phone recognition with a multilingual allophone system},
  author={Li, Xinjian and Dalmia, Siddharth and Li, Juncheng and Lee, Matthew and Littell, Patrick and Yao, Jiali and Anastasopoulos, Antonios and Mortensen, David R and Neubig, Graham and Black, Alan W and Florian, Metze},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8249--8253},
  year={2020},
  organization={IEEE}
}
```
