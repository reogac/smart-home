#!/usr/bin/python

import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
#from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
import pickle as pk
#from sklearn.externals import joblib

predictors = ('ac_status', 'temp', 'humidity', 'light', 'CO2', 'dust', 'day', 'hour')
label = 'action'

power_cut = 0.01 #cut value for positive power consumption

class EngineError(Exception):
    def _init_(self, msg):
        self.__message = msg
    def _str_(self):
        return repr(self.__message)

def load_data(filename):
    return pk.load(filename)

def save_data(data, filename):
    pk.dump(data, filename, 2)

def process_data(filename):
    print "loading sensor data ..."
    df = pd.read_csv(filename)
    df = df.loc[0:5000,]
    print "completed\n"
    print "now pre-processing the data..."
    nrows = len(df)
    # convert string to time object
    df['time'] = df.time.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # add two columns to the data frame to represent hour of the the day, and day of the week
    df['hour'] = df.time.apply(lambda x: x.hour + x.minute/60.0)
    df['day'] = df.time.apply(lambda x: x.day)

    df['ac_status'] = np.zeros(nrows, int)
    df.loc[df.power > power_cut, 'ac_status'] = 1
    df['action'] = np.zeros(nrows, int)
    for i in range(1, nrows):
        if (df.ac_status[i] == 1) and (df.ac_status[i-1]) == 0:
            df.loc[i, 'action'] = 1 #TURN ON
        elif (df.ac_status[i] == 0) and (df.ac_status[i-1]) == 1:
            df.loc[i, 'action'] = -1 #TURN OFF

    print "completed!\n"
    return df

def save_model(model, filename):
    print "completed\nsaving model...\n"
    pk.dump(model, open(file_name, 'wb'), 2)
    print "completed!\n"


def load_model(filename):
    print "loading the model..."
    model = pk.load(open(filename, 'rb'))
    print "completed!\n"
    return model

def train_model(data):
    print "building the model..."
    #model = RandomForestClassifier(n_estimators=100)
    model = tree.DecisionTreeClassifier()
    x = data.as_matrix(predictors)
    y = data.loc[:,label]
    model = model.fit(x, y)
    return model

def test_model(model, data):
    x = data.as_matrix(predictors)
    y = data.loc[:, label]
    pred_y = model.predict(x)
    con_mat = confusion_matrix(y, pred_y)
    return con_mat

def predict(model, inputs):
    input_vect = np.array(inputs)
    return model.predict(input_vect)


def reinforce():
    pass

def parse_sensors(sensors):
    pass

def process(args):
    if (args.command == 'process'):
        print "process"
        if not args.csv_file:
            raise EngineError("original data file not found")
        df = process_data(args.csv_file)
        save_data(df, args.data_file)
    elif (args.command == 'train'):
        print "train"
        #df = process_data(args.csv_file)
        #model = train_model(df)
        #save_model(model)
    elif (args.command == 'predict'):
        print "predict"
        #inputs = parse_sensors(args.sensors)

    elif (args.command == 'evaluate'):
        print "evaluate"
        #df = process_data()
        #model = load_model()
        #print test_model(model, df)
    elif (args.command == 'reinforce'):
        print "reinformance learning"
        #reinforce()
    else:
        raise EngineError("unknown command")

def main():
  """"
  this program is to do the following job:
    - train model from data
    - collect sensors data and store to database
    - predict user action taken on air conditioner
    - collect user feed back
  :return:
  """

  parser = argparse.ArgumentParser(description="home air conditioner controller smart engine")
  command = parser.add_mutually_exclusive_group()
  command.add_argument("-c", "--command", choices=['train','reinforce', 'predict','evaluate','process'],
                       help="tell the engine to train model or make prediction")
  command.add_argument("--train", dest="command", help="train prediction model", action="store_const",
                       const="train")
  command.add_argument("--predict", dest="command", help="make a prediction", action="store_const",
                       const="predict")
  command.add_argument("--process", dest="command", help="process raw data", action="store_const",
                       const="process")
  command.add_argument("--evaluate", dest="command", const = "evaluate",
                       help="evaluate the prediction performance with cross validation "
                            "<for analysis purpose only>", action="store_const")

  command.add_argument("--reinforce", dest="command", help="reinforce the model with user feedbacks",
                       action="store_const", const="reinforce")

  parser.add_argument("-d", "--data_file", action="store_true",
                      dest="data_file", help="file to save/load the processed sensor data")
  parser.add_argument("-t", "--csv_file",
                      dest="csv_file", help="file containing the original sensor data")

  parser.add_argument("-m", "--model", dest="model_file",
                      help="file to save/load the prediction model")

  parser.add_argument("-s", "--sensors", dest="sensors",
                      help="data from sensors for that acts as input of prediction")


  args = parser.parse_args()

  try:
      process(args)
  except EngineError as e:
      print e

if __name__ == "__main__" : main()
