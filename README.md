# InstaLearn
#### Description

The user will first type in training text with some words prefixed with * and !. The model will train on this text. Then, the user will enter test text without the prefixes. The program will tag same or related words and output a color coded version of the evaluation text with accuracy metrics. 

This program involves [bert-as-service](https://github.com/hanxiao/bert-as-service) which uses [BERT](https://github.com/google-research/bert) as a sentence encoder and hosts it as a service.

<h2 align="center">Getting Started</h2>
#### 1. Install
Install the server and client via `pip`. They can be installed separately or even on *different* machines:
```bash
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
```
Note that the server MUST be running on **Python >= 3.5** with **Tensorflow >= 1.10** (*one-point-ten*). Again, the server does not support Python 2!
#### 2. Download a Pre-trained BERT Model
Download a model listed below, then uncompress the zip file into some folder, say `/tmp/uncased_L-12_H-768_A-12/`. In my program, I use [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) for the text, but if you have a powerful hardware, strongly recommend you to use [BERT-Large, Uncased (Whole Word Masking)](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)

<details>
 <summary>List of released pretrained BERT models (click to expand...)</summary>


<table>
<tr><td><a href="https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip">BERT-Large, Uncased (Whole Word Masking)</a></td><td>24-layer, 1024-hidden, 16-heads, 340M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip">BERT-Large, Cased (Whole Word Masking)</a></td><td>24-layer, 1024-hidden, 16-heads, 340M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip">BERT-Base, Uncased</a></td><td>12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip">BERT-Large, Uncased</a></td><td>24-layer, 1024-hidden, 16-heads, 340M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip">BERT-Base, Cased</a></td><td>12-layer, 768-hidden, 12-heads , 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip">BERT-Large, Cased</a></td><td>24-layer, 1024-hidden, 16-heads, 340M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip">BERT-Base, Multilingual Cased (New)</a></td><td>104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip">BERT-Base, Multilingual Cased (Old)</a></td><td>102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
<tr><td><a href="https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip">BERT-Base, Chinese</a></td><td>Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters</td></tr>
</table>

</details>


#### 3. Start the BERT service
After installing the server, you should be able to use `bert-serving-start` CLI as follows:
```bash
bert-serving-start -pooling_strategy NONE -model_dir /tmp/uncased_L-12_H-768_A-12 -max_seq_len 512
```
This will start a service with sequence strategy, meaning that it can output **sequence** embedding rather than single **pooling** embedding. More concurrent requests will be queued in a load balancer. Details can be found in our [FAQ](#q-what-is-the-parallel-processing-model-behind-the-scene) and [the benchmark on number of clients](#speed-wrt-num_client).

#### 4. Use Client to Do the InstaLearn
Now you can InstaLearn sentences simply as follows:
```bash
python InstalLearn.py
```
First input your training sentence:
```bash
input training data: > I lived in *Munich last summer. *Germany has a relaxing, slow summer lifestyle. One night, I got food poisoning and couldn't find !Tylenol to make the pain go away, they insisted I take !aspirin instead.
```
Then you will get a print like following:
