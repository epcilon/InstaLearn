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
Download a model listed below, then uncompress the zip file into some folder, say `/tmp/uncased_L-12_H-768_A-12/`. In my program, I use [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) for the test, but if you have a powerful hardware, strongly recommend you to use [BERT-Large, Uncased (Whole Word Masking)](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)

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
This will start a service with sequence strategy, meaning that it can output **sequence** embedding rather than single **pooling** embedding. More concurrent requests will be queued in a load balancer. Details can be found in the original [FAQ](https://github.com/hanxiao/bert-as-service#q-what-is-the-parallel-processing-model-behind-the-scene) and [the benchmark on number of clients](https://github.com/hanxiao/bert-as-service#speed-wrt-num_client).

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

![image](https://github.com/epcilon/InstaLearn/blob/master/images/1.png)

Then input your inference sentence:
```bash
input inference data: > When I lived in Paris last year, France was experiencing a recession. The night life was too fun, I developed an addiction to Adderall and Ritalin.
```
Then you will get following result:

![image](https://github.com/epcilon/InstaLearn/blob/master/images/2.png)

More examples:

Training sentence: `*Sauropods first appeared in the late !Triassic Period,[7] where they somewhat resembled the closely related ( and possibly ancestral) group *Prosauropoda. By the Late !Jurassic (150 million years ago), *sauropods had become widespread (especially the *diplodocids and *brachiosaurids).`

Inference sentence: `In the Late Cretaceous, the hadrosaurs, ankylosaurs, and ceratopsians experienced success in Western North America and eastern Asia. Tyrannosaurs were present in Asia. Pachycephalosaurs were also present in both North America and Asia.`

Output:

![image](https://github.com/epcilon/InstaLearn/blob/master/images/3.png)

Training sentence: `For the past month the two brightest planets, *Venus and *Jupiter, have been an eye-catching duo in the western sky after sunset. *Venus appear as a brilliant yellow planet many times brighter than any other star in the sky. It is ~18 times brighter than the brightest star !Sirius (located in the southeast) and ~75 times brighter than !Capella (the bright star located nearly over head in the evening).`

Inference sentence: `To the right of the Moon is the Pleiades star cluster. Above and to the right is Mars. And above and to the left is the red giant star Aldebaran. By the next evening, the Moon has moved a bit higher in the sky and hangs here, above Aldebaran. The two stars that make up the front side of the pot are called "pointer stars" because they point toward the star Polaris.`

Output:

![image](https://github.com/epcilon/InstaLearn/blob/master/images/4.png)

#### Use run InstaLearn
One may also start the service on one (GPU) machine and call it from another (CPU) machine as follows:

```python
# on another CPU machine
from InstaLearn import InstaLearn
il = InstaLearn(ip='xx.xx.xx.xx')  # ip address of the GPU machine
il.train('The *cat is playing the !ball.')
il.inference('The dog is tracing the frisbee')
```

Note that you only need `pip install -U bert-serving-client` in this case, the server side is not required. You may also [call the service via HTTP requests.](https://github.com/hanxiao/bert-as-service#using-bert-as-service-to-serve-http-requests-in-json)


