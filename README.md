# Fine-graned analysis using Transformers and Checklist

This repo uses [Huggingface Transformers](https://github.com/huggingface/transformers) git repo 
as well as [Checklist](https://github.com/marcotcr/checklist) testing according to the ACL2020
[paper](https://arxiv.org/abs/2005.04118). It uses the provided pipelines in transformers with
extension to support batches. It also extends both transformers and checklist to work efficiently 
on a single pipeline, using data parallel for multiple GPUs and hidden layers caching.


The purpose of this package is to enable ablation study (for now) on NLP models on checklist 
tasks. The enviroment currently supports:
+ Models from transformers like:
```
'Bert', 'RoBerta', 'DistillBert' 
```
and more...
+ Agregation on per layer, per timestep and per neuron with zeroing out nurons (and other methods) 

Available checklist tasks include:
```
'Negation', 'Vocabulary', 'Taxonomy', 'Robustness', 'NER',  'Fairness', 'Temporal', 'Coref', 'SRL', 'Logic' 
```

## Installation
+ Install checklist from source
```
git clone git@github.com:marcotcr/checklist.git
cd checklist
pip install -e .
```
+ Clone this package and cd to it and install 
```
git clone 
cd 
pip install -e .
```
+ Copy and extract the suits tar file from checklist into this package
```
cp -rf ../checklist/release_data.tar.gz .
tar xvzf release_data.tar.gz
```

 
Example for efficient multi-gpu run on a subset of neurons:
 ```
 python main.py --device-ids 0 1 2 3 --batch-size 1996 --study-num-layer 6 --study-num-ts 16 --study-num-neurons 32
```
For all neurons omit the --study_* flags


The outputs of the main script are two heatmaps of the nurons. Specifically,
the error rate on the checklist task and the confidence by score, when ablating each and one
of the neurons by layer, timestep and neuron inside the hidden dimension. 
Example run Jupyter notebook on a subset of all neurons can be found in [here]().


