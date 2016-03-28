#!/usr/bin/python


import argparse
import logging
import pickle as pk
import sys
from modeling import EngineError

SENSOR_CSV = "sensor.csv"
SENSOR_PICKLE = "sensor.pk"
MODEL_PICKLE = "model.pk"
log_to_file = False

predictors = ('ac_status', 'temp', 'humidity', 'light', 'CO2', 'dust', 'day', 'hour')
label = 'action'

power_cut = 0.01 #cut value for positive power consumption



def load_data(filename):
    import pandas as pd
    logging.info("load processed data from \'"+ filename + "\'")
    try:
        df = pk.load(open(filename, 'rb'))
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to load data")
    return df

def save_data(data, filename):
    """
    TODO: convert data frame to numpy matrix then save to file
    """
    logging.info("saving the data into \'"+ filename + "\'")
    try:
        pk.dump(data, open(filename, 'wb'), 2)
    except Exception as e:
        logging.error(e)
        raise EngineError("Failed to save data")


"""def process_data(filename):
    from datetime import datetime
    import numpy as np
    import pandas as pd

    logging.info("load sensor data from \'" + filename + "\'")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to read csv file")
    df = df.loc[0:5000,]

    logging.info("pre-processing the data")
    try:
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
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to process data")
    return df
"""

def save_model(model, filename):
    logging.info("save model to \'" + filename + "\'")
    try:
        pk.dump(model, open(filename, 'wb'), 2)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to save model")

def load_model(filename):

    logging.info("load model from \'" + filename + "\'")
    try:
        model = pk.load(open(filename, 'rb'))
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to load model")
    return model

def predict(model, inputs):
    import numpy as np
    logging.info("make a prediction")
    try:
        input_vect = np.array(inputs)
        pred = model.predict(input_vect)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to make a prediction")
    return pred


def reinforce():
    logging.info("reinformance learning has not been implemented yet!")
    raise EngineError("reinformance learning has not been implemented yet!", False)

def parse_sensors(sensors):
    import numpy as np
    if not sensors:
        raise EngineError("Sensor data is not provided", False)
    sensors = sensors.replace(' ', '') #remove all whitespaces
    assignments = sensors.split(",") #retrieve sensor value assignments
    pairs = {}
    for a in assignments:
        pair = a.split("=")
        if len(pair) != 2:
            raise EngineError("Failed to parse the sensor data: " + a, False)
        p = pair[0]
        v = float(pair[1])
        if p not in predictors:
            raise EngineError(p + " is not a predictor", False)
        pairs[p] = v

    sensor_values = np.zeros(len(predictors), float)


    for i in range(len(predictors)):
        p = predictors[i]

        if not pairs.has_key(p):
            raise EngineError("the \'" + p + "\' variable is missing", False)
        sensor_values[i] = pairs[p]

    return sensor_values.reshape(1,-1)

def process(args):
    global log_to_file
    global SENSOR_CSV

    if args.log:
        log_to_file = True
        logging.basicConfig(filename=args.log, level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    if (args.command == 'process'):

        if not args.csv_file:
            args.csv_file = SENSOR_CSV

        df = process_data(args.csv_file)
        save_data(df, args.data_file)

    elif (args.command == 'train'):
        if (args.csv_file):
            df = process_data(args.csv_file)
            save_data(df, args.data_file)
        else:
            df = load_data(args.data_file)
        model = train_model(df, args.classifier)
        save_model(model, args.model_file)
    elif (args.command == 'predict'):

        from sklearn import tree
        from sklearn.ensemble import RandomForestClassifier

        inputs = parse_sensors(args.sensors)
        model = load_model(args.model_file)
        action = predict(model, inputs)
        if action==1:
            print "TURN_ON"
        elif action==-1:
            print "TURN_ON"
        else:
            print "DO_NOTHING"

    elif (args.command == 'evaluate'):
        if (args.csv_file):
            df = process_data(args.csv_file)
        else:
            df = load_data(args.data_file)
        model = load_model(args.model_file)
        print evaluate_model(df, model)
    elif (args.command == 'reinforce'):
        reinforce()
    else:
        raise EngineError("unknown command")

def main():
    """"
      this program can do following tasks:
    - read raw data in csv file, process then save to file for training model
    - read raw data in csv file or processed data then train model; save the trained model to file
    - read trained model from file then make a prediction of user's action given input sensor data
    - reinforce model with feedbacks from user (not implement yet)
    - prediction evaluation with cross validation (not implement yet)
    """

    global SENSOR_PICKLE
    global MODEL_PICKLE
    try:
        parser = argparse.ArgumentParser(description="Home air conditioner controller smart engine",
                                         usage='%(prog)s command [options]')
        parser.add_argument("command", choices=['process', 'train', 'predict', 'reinforce','evaluate'],
                            help="tell the program what to do")
        parser.add_argument("-c", "--classifier", choices=['tree', 'forest'], dest="classifier", default="tree",
                            help="Select classification model (Decision tree or Random forest)")
        parser.add_argument("-t", "--csv_file", dest="csv_file",
                            help="file containing the original sensor data, default name = \'" + SENSOR_CSV + "\'")
        parser.add_argument("-d", "--data_file", dest="data_file", default=SENSOR_PICKLE,
                            help="file to save/load the processed sensor data, default name = \'" + SENSOR_PICKLE + "\'")
        parser.add_argument("-m", "--model", dest="model_file", default=MODEL_PICKLE,
                            help="file to save/load the prediction model, default name = \'" + MODEL_PICKLE + "\'")
        parser.add_argument("-s", "--sensors", dest="sensors",
                            help="sensor data for which prediction should be made, here is a sample format: "
                                 "\"ac_status=1, temp=37, humidity=50, dust=100, CO2=1000, light=30, day=2, hour=19.5\"")
        parser.add_argument("-l", "--log", dest="log", help="where to save log messages")
        args = parser.parse_args()

        process(args)
    except EngineError as e:
        sys.stderr.write("engine error \n")
    except Exception as e:
        sys.stderr.write("unknown exception \n")

if __name__ == "__main__" : main()
