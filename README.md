This is a PyTorch implementation of [QANet](https://openreview.net/pdf?id=B14TlG-RW) for SQuAD 2.0. This repository is built upon the [starter code](https://github.com/minggg/squad.git) for Stanford's CS224N Default Project, Winter 2019-2020. Some modules are written based on the implementations in [NLPLearn/QANet](https://github.com/NLPLearn/QANet) and [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

A short setup instruction can be found below. For mode details, please see default-final-project-handout.pdf

Most hyper-parameters are stored in args.py. Some of them are not the same as in the original paper, mostly because we only have 8GB GPU memory.

## Performance

After 27 epochs, this model reaches 66.63 F1 & 63.43 EM on the dev set and 64.35 F1 & 60.96 EM on the test set of CS224N. (Note that these dev and test sets are different from the official ones - see section 3.1 of default-final-project-handout.pdf for more details). This performance is in line with the remark by [Rajpurkar et al](https://arxiv.org/abs/1806.03822) that "a strong neural system that gets 86% F1 on SQuAD 1.1 achieves only 66% F1 on SQuAD 2.0."

The training speed is extremely slow compared to what was observed in the original paper. On an Azure NV6 virtual machine, our model takes about 45-50 minutes to go through each epoch. Both our model and the BiDAF model in the starter code take about 8 hours to reach 64.01 F1 & 60.85 EM on the dev set, which is the peak performance of the latter. In contrast, the original paper reported that their QANet model equaled the peak F1 score of BiDAF after one-fifth of the training time.

## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `source activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
