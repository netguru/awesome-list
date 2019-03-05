# awesome-list [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

The research awesome list can be found [here](research/readme.md)

List of machine learning frameworks and tools. Inspired by
[`awesome-machine-learning`](https://github.com/josephmisiti/awesome-machine-learning)

If you want to contribute to this list (please do), just open a pull request.
Also, a listed repository should be deprecated if:

- Repository's owner explicitly say that "this library is not maintained".
- Not committed for long time (2~3 years).

# Table of Contents

- [Environments](#environments)

    - [Graphical](#graphical)
    - [Hybrid](#hybrid)

- [Libraries](#libraries)

    - [Deep Learning](#deep-learning)
    - [Dimensionality Reduction](#dimensionality-reduction)
    - [Multipurpose](#multipurpose)
    - [Natural Language Processing](#natural-language-processing)
    - [Optimization](#optimization)
    - [Probabilistic Programming](#probabilistic-programming)
    - [Speech Processing](#speech-processing)

- [Tools](#tools)

    - [Compilers](#compilers)
    - [Data Adapters](#data-adapters)
    - [Data Gathering](#data-gathering)
    - [Data Management](#data-management)
    - [Job Management](#job-management)
    - [Parallelization](#parallelization)
    - [Data Visualization](#data-visualization)
    - [Reporting](#reporting)

# Environments

## Graphical

- [AI-Blocks](https://github.com/MrNothing/AI-Blocks) - a powerful and intuitive
WYSIWYG interface that allows anyone to create Machine Learning models.

## Hybrid

- [Luna Studio](https://github.com/luna/luna-studio) - Hybrid textual and visual
functional programming

# Libraries

## Deep Learning

- [fast.ai](https://github.com/fastai/fastai) - The fastai library simplifies
training fast and accurate neural nets using modern best practices
- [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural
networks in Python with strong GPU acceleration
- [tensorflow](https://github.com/tensorflow/tensorflow) - An Open Source
Machine Learning Framework for Everyone by Google
- [neon](https://github.com/NervanaSystems/neon) - Intel® Nervana™ reference
deep learning framework committed to best performance on all hardware
- [cleverhans](https://github.com/tensorflow/cleverhans) - An adversarial
example library for constructing attacks, building defenses, and benchmarking
both
- [Netron](https://github.com/lutzroeder/Netron) - a viewer for neural network,
deep learning and machine learning models.
[Online viewer](https://lutzroeder.github.io/netron/)

## Dimensionality Reduction

- [umap](https://github.com/lmcinnes/umap) - dimension reduction technique that
can be used for visualisation similarly to t-SNE, but also for general
non-linear dimension reduction

## Multipurpose

- [numpy](https://github.com/numpy/numpy) - The fundamental package for
scientific computing with Python.
- [RAPIDS](https://github.com/rapidsai) - Open GPU Data Science. More
[here](https://rapids.ai/) or in [cheatsheet](https://rapids.ai/assets/files/cheatsheet.pdf)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - machine
learning in Python
- [scipy](https://github.com/scipy/scipy) - open-source software for
mathematics, science, and engineering. It includes modules for statistics,
optimization, integration, linear algebra, Fourier transforms, signal and image
processing, ODE solvers, and more.
- [SymPy](https://github.com/sympy/sympy) - A computer algebra system written in pure Python, library for symbolic mathematics

## Natural Language Processing

- [Allen NLP](https://github.com/allenai/allennlp) - An open-source NLP research
library, built on PyTorch.
- [PyText](https://github.com/facebookresearch/pytext) - A natural language
modeling framework based on PyTorch by Facebook Research
- [flair](https://github.com/zalandoresearch/flair) - A very simple framework
for state-of-the-art Natural Language Processing (NLP) by Zalando Research

## Optimization

- [nevergrad](https://github.com/facebookresearch/nevergrad) - A Python toolbox
for performing gradient-free optimization by Facebook Research

## Probabilistic Programming

- [pyro](https://github.com/uber/pyro) - Deep universal probabilistic
programming with Python and PyTorch by Uber
- [pgmpy](https://github.com/pgmpy/pgmpy) - Python Library for Probabilistic
Graphical Models

## Speech Processing

- [warp-ctc](https://github.com/baidu-research/warp-ctc) - loss function to
train on misaligned data and labels by Baidu Research
- [DeepSpeech](https://github.com/mozilla/DeepSpeech) - A TensorFlow
implementation of Baidu's DeepSpeech architecture
- [speech-to-text-wavenet](https://github.com/buriburisuri/speech-to-text-wavenet) -
End-to-end sentence level English speech recognition based on DeepMind's WaveNet
and tensorflow
- [pykaldi](https://github.com/pykaldi/pykaldi) - A Python wrapper for Kaldi -
a toolkit for speech recognition
- [pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi) - pytorch-kaldi
is a project for developing state-of-the-art DNN/RNN hybrid speech recognition
systems. The DNN part is managed by pytorch, while feature extraction, label
computation, and decoding are performed with the kaldi toolkit.

# Tools

## Compilers

- [glow](https://github.com/pytorch/glow) - Compiler for Neural Network hardware
accelerators by PyTorch
- [jax](https://github.com/google/jax) - Composable transformations of
Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more by
Google
- [numba](https://github.com/numba/numba) - NumPy aware dynamic Python compiler
using LLVM

## Data Adapters

- [csvkit](https://github.com/wireservice/csvkit) - A suite of utilities for
converting to and working with CSV, the king of tabular file formats.
- [redash](https://github.com/getredash/redash) - Connect to any data source,
easily visualize, dashboard and share your data.
- [odo](https://github.com/blaze/odo/blob/master/docs/source/index.rst) - Odo migrates between many formats. These include in-memory structures like list, pd.DataFrame and np.ndarray and also data outside of Python like CSV/JSON/HDF5 files, SQL databases, data on remote machines, and the Hadoop File System.

## Data Gathering

- [scrapy](https://github.com/scrapy/scrapy) - high-level library to write
crawlers and spiders.

## Data Management

- [Quilt](https://github.com/quiltdata/quilt) - Quilt versions and deploys data

## Job Management

- [luigi](https://github.com/spotify/luigi) - Luigi is a Python module that
helps you build complex pipelines of batch jobs. It handles dependency
resolution, workflow management, visualization etc. It also comes with Hadoop
support built in. By Spotify.

## Parallelization

- [pywren](https://github.com/pywren/pywren) - parfor on AWS Lambda
- [horovod](https://github.com/uber/horovod) - Distributed training framework
for TensorFlow, Keras, PyTorch, and MXNet by Uber.
- [dask](https://github.com/dask/dask) - library for parallel computing in Python with dynamic task scheduling: numpy computation graphs.

## Data Visualization

- [matplotlib](https://github.com/matplotlib/matplotlib) - plotting with Python
- [bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python
- [plotly](https://github.com/plotly/plotly.py) - An open-source, interactive
graphing library for Python
- [dash](https://github.com/plotly/dash) - Analytical Web Apps for Python. No
JavaScript Required.
- [Jupyter Dashboards](https://github.com/jupyter/dashboards) - Jupyter layout
extension
- [redash](https://github.com/getredash/redash) - Connect to any data source,
easily visualize, dashboard and share your data.
- [vega](https://github.com/vega/vega) - visualization grammar, a declarative
format for creating, saving, and sharing interactive visualization designs.
With Vega you can describe data visualizations in a JSON format, and generate
interactive views using either HTML5 Canvas or SVG.
- [scikit-plot](https://github.com/reiinakano/scikit-plot) - sklearn wrapper to automate
frequently used  machine learning visualizations.

## Reporting

- [tensorboardX](https://github.com/lanpa/tensorboardX) - tensorboard for
pytorch (and chainer, mxnet, numpy, ...)
- [Weights and Biases](https://www.wandb.com) - Experiment Tracking for Deep
Learning
