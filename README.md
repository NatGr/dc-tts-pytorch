# pytorch implementation of DC_TTS 
This repo contains a Pytorch implementation of [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](http://arxiv.org/abs/1710.08969).  
DC_TTS is [relatively lightweight to train](http://arxiv.org/abs/1710.08969) and [performs better than Tacotron](http://arxiv.org/abs/1903.11269).  
It is inspired from the [tensorflow implementation](https://github.com/Kyubyong/dc_tts).  
Since the tf implementation provides models in [english](https://github.com/Kyubyong/dc_tts) as well as in [german, greek, spanish, finnish, french, hungarian, japanese, dutch, russian and chinese](https://github.com/Kyubyong/css10), this repo is designed to init its pytorch models from the corresponding tf weights.
When using tf-weights to init a model, do not forget to set the vocab argument to the same value as was used for training, for [english](https://github.com/Kyubyong/dc_tts/blob/master/hyperparams.py) or [other languages](https://github.com/Kyubyong/css10/blob/master/dc_tts/hyperparams.py).  

## Audio samples
Here are audio samples in [french](https://soundcloud.com/user-1954917/french), [english](https://soundcloud.com/user-1954917/sets/english), [spanish](https://soundcloud.com/user-1954917/sets/spanish) and [german](https://soundcloud.com/user-1954917/sets/german).  
These were generated with this repo's code but with https://github.com/Kyubyong/dc_tts and https://github.com/Kyubyong/css10 weights.  

## Installation
You will need to install [pytorch on gpu](https://pytorch.org/get-started/locally/), as well as TorchAudio
```
conda install torchaudio -c pytorch
```
.This version of master uses Pytorch 1.6 to benefit from AMP automatically, postprocessing is faster with Pytorch 1.7. So we recommend pytorch version >= 1.7.  
as well as the other requirements  
```
pip install -r requirements.txt
```

### Loading tf weights
To load tensorflow weights, you will need to install a version of tensorflow 1 posterior to 1.3 (don't panic, you don't need gpu support for this). For example:
```
pip install tensorflow==1.15.3
```

## Usage
If you are not using pretrained models, you should train both text2Mel and SSRN separately for example by using max_num_samples_to_train_on (numbers here are just examples and depends on dataset)
```
python train.py --name my_model --net Text2Mel --train_data_folder folder --max_num_samples_to_train_on 10000000
```
and
```
python train.py --name my_model --net SSRN --train_data_folder folder --max_num_samples_to_train_on 2000000
```
see ```python train.py -h``` for additionnal arguments.

Then, to perform inference, you can use:
```
python synthesize.py --ttm_ckpt text2mel.ckpt --ssrn_ckpt ssrn.ckpt --sentences_file my_sentences.txt
```
where my_sentences.txt contains the sentences to utter, each on one line. If sentences are too long, they will be splitted and the output of the models concatenated.  
The wav audio files should be in a folder named audio next to my_sentences.txt

### data format
train.py's "--train_data_folder" flag should contain an audio folder containing the data and a transcript.csv file.  
The trascript.csv file should contain two columns named file and sentence. File being the file name in the audio folder and sentence the sentence spoken within the file. These values must be separated by a ";" and sentence should contain no ";".
One of the first things done in train.py is to compute and store the mel and mag spectrograms and store them on disk, this takes a few minites and requires quite a lot of disk space (around 25GB for 5GB of audio (20 hours)).

## Fine tuning from a TF model
This works in the exact same way as fine-tune from pytorch, except that the argument "--base_model" should point to a folder containing the tf checkpoint files instead of a pytorch checkpoint file.
I also made some helper scripts to [collect data with your voice](https://github.com/NatGr/easy_voice_registration) or to [label audio files](https://github.com/NatGr/annotate_audio).

## Differences with paper
Since the code can load the weights from the [tensorflow implementation](https://github.com/Kyubyong/dc_tts) and I inspired myself from it, I reproduced their architectural specificities:  
 - layer normalization (only operating over the channels axis)  
 - dropout layers  
 - in SSRN, the transposed convolution layers have a kernel size of 3 and not 2  
 - we clip gradient whose value is > 1 or < -1  
 - one can optionnally use Noam LR scheduler as in tf implementation  

Unlike the tensorflow implementation but similar to what was reported in [the paper introducing the multilingual datasets (at the end of section 4.4)](http://arxiv.org/abs/1903.11269). We have weird mumbling once the model said what it has to do, to neutralize that, we remove the outputs to Text2Mel where the attention is "looking" over padding chars.

