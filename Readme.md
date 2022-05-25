# Crypto_deeplearning_study

A python project based on quantative approach of predicting returns of financial (primarly crypto) assets.

## Table of contents

* [Introduction](#Introduction)
* [Technologies](#Technologies)
* [Setup](#setup)
* [Features](#features)
* [Author](#Author)

### Introduction

The aim of this project is to study and improve deep-learning techniques for financial analysis. Project implements ways to easily train new models, predict returns,
and evaluate the results. Custom data-pipeline allows easy implementation of additional data-manipulation modules. This project is created as a training exercise in quantitative 
financial analysis, and practice with machine learning and deep networks. Future goals of the project are to use the models and techniques created here for
crypto portfolio management.

To learn more about this project and its usage, visit my medium.com article: https://medium.com/@ipopovca/using-deep-learning-to-predict-crypto-markets-here-we-go-again-93c82361d0e9

### Technologies and libraries

* Anaconda 3
* Python 3.9
* Tensorflow 
* Keras
* Pandas
* Docker
* Numpy
* Scikit-learn
* Matplotlib

### Setup

To set up the project, please clone the repository.
There are 2 ways to run this project: via docker or by running the Main.py file directly.
If using a docker file, environmental parameters need to be set up via docker-compose.yml.
```
batch_size: The batch size to use for the data (First dimension)
time_steps: Amount of time steps to use for the data (Second dimension)
mode: Either training, prediction,continue or data_resample. Continue will continue the training of specified model.
model_type: A model to use. Currently supports: Dense, Conv1d, Conv2d, lstm, Convlstm.
model_load_name: If using prediction or continue, specifies which model will be acted upon.
ticker: Crypto ticker to use. If using training, make sure the csv is in data/interval/ticker location, in OHLCV format.
interval: The data interval to use.
cryptowatch_key: the api key for cryptowatch api to grab data
interval_from: used for data resampling, interval and name of the folder from which to resample
interval_to: used for data resampling, interval and name of the folder to which to resample
```


You can build the docker image by running
```
$ docker build -t mlfinancial .
```
Note: Currently the image is setup with a bind-mount. If you would like to use a named volume, create one, and change the volumes parameter in docker-compose.yml
You will also need to add 'Copy .' line to Dockerfile in order to copy the project onto the image

After this you can launch the app via
```
$ docker-compose up -d
```

If running through Main.py directly, the above environmental parameters need to be setup in .env file.

If you'd like to modify the structure of a network to use, you will need to access the Networks/structures directory and modify the desired networks directly.

If you'd like to resample OHLCV data to a lower resolution, place your higher resolution data in data/interval folder and run Data_Processing/resample_data.py script
with desired in/out intervals. All the csv files in original interval folder will be converted, without modifying the originals. 
Alternatively, resampling can be done by setting enviromental variable 'mode' to 'data_resample' and executing main.py or spinning up the docker container.

To utilize SQL model tracking:
If being deployed on a new enviroment,
1) Run create_adam_opt_table, create_model_table
2) Run insert_init_db if there are already existing models you would like to index

For existing system:
1) Once the model been trained and approved, use save_model_to_db to add it to database
2) To add same model to ensemble, use move_model_to_ensembly (Note: This will create a copy)
3) To delete a model from both hard-drive and the database, use delete_model

Currently SQL managment is only done using the respective files/functions in sql folder. 




### Features

* Acquisition of crypto data through Rest-API
* Resampling of existing data to a lower-resolution
* Data feature enhancement via technical analysis
* Data rescaling and PCA (Principal Component Analysis)
* Several deep learning networks to train and experiment with
* Custom data pipeline with ease of adding and removing features
* Custom losses and metrics, created to enhance predictive power of networks for financial data
* Tools to evaluate the predictive power of the model
* SQL database to store model parameters

### Project Status
This project is not fully completed.
Future additions will include:

* More networks, such as Conv3D
* Better data processing, including noise removal, additional time-series data augmentation through feature extractors, and addition of OHLCV candle analysis.
* Introduction of an investment universe, working with multiple assets at once
* Prediction of OHLCV candles directly
* Losses that are better suited for financial tasks, implemented via Keras custom layers
* Event-driven backtesting
* Ensembling techniques
* Addition of order-book data
* Quantile-based portfolio managment tools

### Author

Created by Ivan Popov

Organisational help by Alon Rodovinsky
