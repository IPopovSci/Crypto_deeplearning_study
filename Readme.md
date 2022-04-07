# Crypto_deeplearning_study

A python-based project based on quantative approach of predicting returns of financial (primarly crypto) assets.

## Table of contents

* [Introduction](#Introduction)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Author](#Author)

### Introduction

The aim of this project is to study, and improve deep-learning techniques for financial analysis. Project implements ways to easially train new models, predict returns,
and evaluate the results. Custom data-pipeline allows easy implementation of additional data-manipulation modules. This project is created as a training in quantative 
financial analysis, and practice with machine learning and deep networks in particular. Future goals of the project is to use the models and techniques created here for
crypto portfolio managment.

To learn more about this project and its usage, visit my medium.com article: (link here)

### Technologies

* Anaconda 3
* Python 3.9
* Tensorflow 
* Keras
* Pandas
* Docker
* Numpy
* Scikit-learn

### Setup

To setup, please clone the repository.
There are 2 ways to run this project: via docker or by using Main.py file directly.
If using a docker file, enviromental parameters need to be setup via docker-compose.yml
```
batch_size: The batch size to use for the data (First dimension)
time_steps: Amount of time steps to use for the data (Second dimension)
mode: Either training, prediction or continue. Continue will continue the training of specified model.
model_type: A model to use. Currently supports: Dense, Conv1d, Conv2d, lstm, Convlstm.
model_load_name: If using prediction or continue, specifies which model will be acted upon.
ticker: Crypto ticker to use. If using training, make sure the csv is in data/interval/ticker location, in OHLCV format.
interval: The data interval to use.
```


You can build the docker image by running
```
$ docker build -t name .
```
Note: Currently image is setup with a bind-mount. If you would like to use a named volume, create one, and change the volumes: parameter in docker-compose.yml

After this you can launch the app via
```
$ docker-compose up -d
```

If running through Main.py directly, above enviromental parameters need to be setup in .env file.

If you'd like to modify the structure of a network to use, you will need to acess Networks/structures and modify the desired network directly.

If you'd like to resample OHLCV data to a lower resolution, place your higher resolution data in data/interval folder and run Data_Processing/resample_data.py script
with desired in/out intervals. All the csv files in original interval folder will be converted, without modifying them.

### Features

* Acquisition of crypto data through Rest-API
* Resampling of existing data to a lower-resolution
* Data feature enchansment via technical analysis
* Data rescaling and PCA (Principal Component Analysis)
* Several deep learning networks to train and experiment with
* Custom data pipeline with ease of adding and removing features
* Custom losses and metrics, created to enhance predictive power of networks for financial data
* Tools to evaluate the predictive power of the model

### Project Status
This project is not fully completed.
Future additions will include:

* More various networks, such as Conv3D
* Better data processing, including noise removal, additional time-series data augmentation through feature extractors, and addition of OHLCV candle analysis.
* Introduction of an investment universe, working with multiple assets at once
* Prediction of OHLCV candles directly
* Losses that are better suited for financial tasks, implemented via Keras custom layers
* Event-driven backtesting
* Ensembling techniques
* Addition of order-book data
* Qunatile-based portfolio managment tools

### Author

Created by Ivan Popov

Organisational help by Alon Rodovinsky
