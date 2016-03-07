
# Program usage

## Overview

`smart-aircon` is a small python program whose main job is to predict air conditioner controlling actions from user. The prediction is based on user context information collected through sensor system.

In order to fulfil such a job, it firstly needs to build a prediction model from previously observed data, and certainly before that, it also needs to perform data pre-processing. So, basically `smart-aircon.py` will do these following tasks:

+ Process raw data. Sensor data in raw format can be processed for constructing the prediction model.
+ Train model from processed data; the model are later used for generating prediction.
+ Make a prediction given a sensor data input with the trained model.
+ Evalate prediction performance of the model (for analysis purpose, not yet implemented).
+ Reinforce model. Improve model with user's feedback data (more research needs).

The program source code can be downloaded from [this github repository](https://github.com/reogac/smart-home.git).

You will need [git](https://en.wikipedia.org/wiki/Git_(software)) to clone the repository

```bash
    git clone https://github.com/reogac/smart-home.git
```
Several python packages are requred for running the program, including [scikit-learn](http://scikit-learn.org) (for classification model learning), [pickle](https://docs.python.org/3/library/pickle.html) (for python objsect serialization) and [pandas](http://pandas.pydata.org/) (for data frame processing).

If you are not familiar with python package installation, perhaps it is a better idea to try the program on a ready-to-run computer. I have installed the program as well as neccesary packages on a Beaglebone black (BBB) machine. The machine can be accessed using ssh:

```bash
    ssh debian@reogac.iptime.org -p 2200
```

Once you are at the terminal console of the BBB, go the folder that has the program:

```bash
    cd ~/smart-home
```

From there, follows below instructions for running the program.


## Usage

Program usage will be shown up with this command:


```python
!./smart-aircon.py --help
```

    usage: smart-aircon.py command [options]
    
    Home air conditioner controller smart engine
    
    positional arguments:
      {process,train,predict,reinforce,evaluate}
                            tell the program what to do
    
    optional arguments:
      -h, --help            show this help message and exit
      -c {tree,forest}, --classifier {tree,forest}
                            Select classification model (Decision tree or Random
                            forest)
      -t CSV_FILE, --csv_file CSV_FILE
                            file containing the original sensor data, default name
                            = 'sensor.csv'
      -d DATA_FILE, --data_file DATA_FILE
                            file to save/load the processed sensor data, default
                            name = 'sensor.pk'
      -m MODEL_FILE, --model MODEL_FILE
                            file to save/load the prediction model, default name =
                            'model.pk'
      -s SENSORS, --sensors SENSORS
                            sensor data for which prediction should be made, here
                            is a sample format: "ac_status=1, temp=37,
                            humidity=50, dust=100, CO2=1000, light=30, day=2,
                            hour=19.5"
      -l LOG, --log LOG     where to save log messages


## <a id="training-model"/>Training-model


Before prediction can be made, a model must be trained from observed data. Usually the training should be performed on a computer with high computation power. However, some algorithms such as decision tree, random forest, can perform reasonably comfortable on lightweight machines such as Beaglebone black, as long as the data size is **reasonable small**.

The input for training model can be raw data in csv format or processed data which was the output of [data processing step](#data-processing).

For input is raw data, the command should be:
```bash
    smart-aircon train -t raw-data-file -m model-file -c classification-method
```

For input is processed data, the command should be:
```bash
    smart-aircon train -d processed-data-file -m model-file -c classification-method
```

Here, the classification method can either be `tree` for Desison tree or `forest` for Random forest. Default value for the parameter is `tree`.

If the training succeed, it should save the model into the specified `model-file`.

Training should be performed periodically. It can be done easily on any linux system by using a time-based scheduler such as [crontab](https://en.wikipedia.org/wiki/Cron).

### Note
Currently a model trained on 64 bit computer can not be loaded on BBB (32 bit machine). The reason is that `scikit-learn` package use different data types on the two architectures. A work around solution is to install a 32bit linux OS for training the model.


## <a id="data-processing"/>Data pre-processing

Training model consists of two consecutive steps: pre-processing raw data and learning model from the processed data. Both steps are computational extensive on lightweight machines, thus it is advisible to perform one or both of them on more powerful computer then copy the output to the target machine. While different algorithms may requires diffrent levels of computational power, the requirment for data pre-processing stays the same, mostly depending on the size of the data. Breaking down the training into two threfore makes deployment more flexible.

Pre-processing raw data can be perform with this follwoing command:

``` bash
    smart-aircon process -t raw--data-file -d processed-data-file
```
Here `raw-data-file` is the sensor data collected in a csv format and `processed-data-file` is the output file. The processed data then can be used as input for [model training](#training-model)


## Making a prediction

Once the model is trained, the program can make prediction of user action based on the input data from sensor system. Inputs should be all avaiable information from sensors at the time of predition. To make a prediction, the following command should be entered:

```bash
    smart-aircon predict --sensors sensor-data
```
where the `sensor-data` format should be followed the below example:
```bash
"ac_status=1, temp=37, humidity=50, dust=100, CO2=1000, light=30, day=2,hour=19.5"
```

The command return the predicted action of user which can be one of three possible outcomes: `TURN-ON`, `TURN-OFF` or `DO-NOTHING`.

### When to make prediction?

Let supposed that we have a module called `Action Recommender` that interact with user and recommend him which action he should take then act accordingly to the his decision (turn on/off the air conditioner). Basically, the module should interface with three other modules:

 + interacts with user mobile application (it can be `restful` or any kind of message-based interface)
 + interacts with prediction module (this program, aka `smart-aircon`; interface is command line)
 + interacts with controller module (to control air conditioner)
 
The module should not wait for user to initiate a prediction request, instead it has to make prediction pediodically. In order to do so, it should read sensor data, for example, in every 3 minutes, then call the prediction command to get the predicted user action. Depending on the prediction outcome, it can choose either to ignore (do nothing) or recommend user to take a action (turn on or turn off).

Therefore, the `Action Recommender` should be implemented as a daemon or it can be a script associated with a crontab job that runs periodically (3 minutes).

## Model reinforcemence

Model reinforcement is the process of re-training the model upon receiving user's feedback, with expectation of improving prediction quality. Currently we are not i



```python

```
