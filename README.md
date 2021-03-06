# awesome-list [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

The research awesome list can be found [here](research/readme.md)

List of curated machine learning frameworks and tools, inspired by
[`awesome-machine-learning`](https://github.com/josephmisiti/awesome-machine-learning).

# Contribute

Contributions welcome! Read the [contribution guidelines](CONTRIBUTING.md) first.

# Table of Contents
- [Datasets](#datasets)

    - [Speach](#speach)

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
    - [Recommender Systems](#recommender-systems)
    - [Speech Processing](#speech-processing)
    - [Model deployment](#model-deployment)

- [Tools](#tools)

    - [Compilers](#compilers)
    - [Data Adapters](#data-adapters)
    - [Data Gathering](#data-gathering)
    - [Data Management](#data-management)
    - [Feature engineering](#feature-engineering)
    - [Hardware Management](#hardware-management)
    - [Job Management](#job-management)
    - [Parallelization](#parallelization)
    - [Data Visualization](#data-visualization)
    - [Reporting](#reporting)

- [License](#license)
  
# Datasets

## Speach

- [Common-Voice](https://voice.mozilla.org/en) - Multi language, open source database with voice samples that anyone can use to train speech-enabled applications.

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
- [Online viewer](https://lutzroeder.github.io/netron/) -
- [List of conversion tools for DNN models](https://github.com/ysh329/deep-learning-model-convertor) - list of many libraries (github projects) that provides options for converting DNN models between different frameworks

## Dimensionality Reduction

- [umap](https://github.com/lmcinnes/umap) - dimension reduction technique that
can be used for visualisation similarly to t-SNE, but also for general
non-linear dimension reduction

## Multipurpose

- [DALI](https://github.com/NVIDIA/DALI) - A library containing both highly optimized building blocks and an execution engine for data pre-processing in deep learning applications [docs](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)
- [gin-config](https://github.com/google/gin-config) - Gin provides
a lightweight configuration framework for Python, by Google.
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - A python package offering a number of re-sampling techniques. Compatible with scikit-learn, is part of scikit-learn-contrib projects.
- [mlxtend](https://github.com/rasbt/mlxtend) - A library of extension and
helper modules for Python's data analysis and machine learning libraries.
- [numpy](https://github.com/numpy/numpy) - The fundamental package for
scientific computing with Python.
- [PyOD](https://github.com/yzhao062/pyod) - Outlier detection library
- [RAPIDS](https://github.com/rapidsai) - Open GPU Data Science. More
[here](https://rapids.ai/) or in [cheatsheet](https://rapids.ai/assets/files/cheatsheet.pdf)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - machine
learning in Python
- [scikit-learn-laboratory (SKLL)](https://github.com/EducationalTestingService/skll) - CLI
for sklearn, working with configuration files
- [scipy](https://github.com/scipy/scipy) - open-source software for
mathematics, science, and engineering. It includes modules for statistics,
optimization, integration, linear algebra, Fourier transforms, signal and image
processing, ODE solvers, and more.
- [statsmodels](https://github.com/statsmodels/statsmodels/) - statistical
modeling and econometrics in Python: time-series analysis, survival analysis,
discrete models, Generalized Linear Models
- [SymPy](https://github.com/sympy/sympy) - A computer algebra system written in pure Python, library for symbolic mathematics
- [Vaex](https://github.com/vaexio/vaex/) - Out-of-Core DataFrames for Python, visualize and explore big tabular data at a billion rows per second. [Project page](https://vaex.io/)

## Natural Language Processing

- [Allen NLP](https://github.com/allenai/allennlp) - An open-source NLP research
library, built on PyTorch.
- [PyText](https://github.com/facebookresearch/pytext) - A natural language
modeling framework based on PyTorch by Facebook Research
- [pytorch-transformers](https://github.com/huggingface/transformers) -
A library of state-of-the-art pretrained models for (NLP) including BERT, GPT,
GPT-2, Transformer-XL, XLNet and XLM with multiple pre-trained model weights 
- [flair](https://github.com/flairNLP/flair) - A very simple framework
for state-of-the-art Natural Language Processing (NLP) by Zalando Research
- [gensim](https://github.com/RaRe-Technologies/gensim) - Topic modeling 
for humans. Enables analysis of plain-text documents for semantic structure. 
Compatible with Word2Vec, FastText and other NLP models.
- [spaCy](https://github.com/explosion/spaCy) - spaCy is a library for advanced 
Natural Language Processing in Python and Cython. spaCy comes with pretrained 
statistical models and word vectors, and supports tokenization for 50+ languages.
- [TextBlob](https://github.com/sloria/TextBlob/) - Simple, Pythonic, text
processing--Sentiment analysis, part-of-speech tagging, noun phrase extraction,
translation, and more.

## Optimization

- [hyperopt](https://github.com/hyperopt/hyperopt) - Distributed Asynchronous 
Hyperparameter Optimization in Python
- [nevergrad](https://github.com/facebookresearch/nevergrad) - A Python toolbox
for performing gradient-free optimization by Facebook Research

## Probabilistic Programming

- [pyro](https://github.com/pyro-ppl/pyro) - Deep universal probabilistic
programming with Python and PyTorch by Uber
- [pgmpy](https://github.com/pgmpy/pgmpy) - Python Library for Probabilistic
Graphical Models

## Recommender systems

- [surprise](https://github.com/NicolasHug/Surprise) - A Python scikit for
building and analyzing recommender systems

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

## Model Deployment

- [gradio](https://gradio.app/) - Library to easily integrate models into existing python (web) apps.


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

## Data Annotation

- [doccano](https://github.com/doccano/doccano) - Open source text annotation tool for machine learning practitioners.
- [snorkel](https://github.com/snorkel-team/snorkel) - A system for
quickly generating training data with weak supervision

## Data Gathering

- [scrapy](https://github.com/scrapy/scrapy) - high-level library to write
crawlers and spiders.

## Data Management

- [Quilt](https://github.com/quiltdata/quilt) - Quilt versions and deploys data

## Data Visualization

- [matplotlib](https://github.com/matplotlib/matplotlib) - plotting with Python
- [bokeh](https://github.com/bokeh/bokeh) - Interactive Web Plotting for Python
- [plotly](https://github.com/plotly/plotly.py) - An open-source, interactive
graphing library for Python
- [dash](https://github.com/plotly/dash) - Analytical Web Apps for Python. No
JavaScript Required.
- [Jupyter Dashboards](https://github.com/jupyter-attic/dashboards) - Jupyter layout
extension
- [vega](https://github.com/vega/vega) - visualization grammar, a declarative
format for creating, saving, and sharing interactive visualization designs.
With Vega you can describe data visualizations in a JSON format, and generate
interactive views using either HTML5 Canvas or SVG.
- [schema crawler](https://github.com/schemacrawler/SchemaCrawler) - a tool
to visualize database schema
- [scikit-plot](https://github.com/reiinakano/scikit-plot) - sklearn wrapper to automate
frequently used  machine learning visualizations.

## Feature engineering

- [featuretools](https://github.com/FeatureLabs/featuretools) - an open source python framework for automated feature engineering.

## Hardware Management

- [nvtop](https://github.com/Syllo/nvtop) - a (h)top like task monitor for NVIDIA GPUs. 
It can handle multiple GPUs and print information about them in a htop familiar way.

- [s2i](https://github.com/openshift/source-to-image) - Source-to-Image (S2I) is a toolkit and workflow for building reproducible container images from source code. S2I produces ready-to-run images by injecting source code into a container image and letting the container prepare that source code for execution. By creating self-assembling builder images, you can version and control your build environments exactly like you use container images to version your runtime environments.

## Job Management

- [luigi](https://github.com/spotify/luigi) - Luigi is a Python module that
helps you build complex pipelines of batch jobs. It handles dependency
resolution, workflow management, visualization etc. It also comes with Hadoop
support built in. By Spotify.

## Parallelization

- [pywren](https://github.com/pywren/pywren) - parfor on AWS Lambda
- [horovod](https://github.com/horovod/horovod) - Distributed training framework
for TensorFlow, Keras, PyTorch, and MXNet by Uber.
- [dask](https://github.com/dask/dask) - library for parallel computing in Python with dynamic task scheduling: numpy computation graphs.

## Reporting

- [shap](https://github.com/slundberg/shap) - A unified approach to explain the output of any machine learning model
- [tensorboardX](https://github.com/lanpa/tensorboardX) - tensorboard for
pytorch (and chainer, mxnet, numpy, ...)
- [Weights and Biases](https://www.wandb.com) - Experiment Tracking for Deep
Learning
- [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) - tool for generating exploratory data 
analysis for the provided DataFrame - presenting results in the form of HTML report


# License

[![CC0][CC0-badge]][CC0-link]

To the extent possible under law, Netguru has waived all copyright
and related or neighboring rights to this work. See [LICENSE](LICENSE).

<!-- Links -->
[CC0-badge]: http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg
[CC0-link]: https://creativecommons.org/publicdomain/zero/1.0/
