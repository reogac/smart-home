#!/usr/bin/python


import argparse
import logging
import numpy as np
import pickle as pk
import sys
from modeling import EngineError
from  modeling import log_to_file

SENSOR_CSV = "sensor.csv"
SENSOR_PICKLE = "sensor.pk"
MODEL_PICKLE = "model.pk"

def load_data(filename):
    logging.info("load processed data from \'"+ filename + "\'")
    try:
        df = pk.load(open(filename, 'rb'))
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to load data")
    return df

def save_data(data, filename):
    """save data to file
    """
    logging.info("saving the data into \'"+ filename + "\'")
    try:
        pk.dump(data, open(filename, 'wb'), 2)
    except Exception as e:
        logging.error(e)
        raise EngineError("Failed to save data")


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
    logging.info("make a prediction")
    try:
        #input_vect = np.array(inputs)
        pred = model.predict(inputs)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to make a prediction")
    return pred


def reinforce():
    logging.info("reinformance learning has not been implemented yet!")
    raise EngineError("reinformance learning has not been implemented yet!", False)

def parse_sensors(sensors):
    from modeling import POWER_CUT
    predictors = ('temp', 'humidity', 'light', 'co2', 'dust', 'day', 'hour')

    if not sensors:
        raise EngineError("Sensor data is not provided", False)
    sensors = sensors.replace(' ', '') #remove all whitespaces
    sensors = sensors.lower()
    assignments = sensors.split(",") #retrieve sensor value assignments
    pairs = {}
    for a in assignments:
        pair = a.split("=")
        if len(pair) != 2:
            raise EngineError("Failed to parse the sensor data: " + a, False)
        p = pair[0]
        v = float(pair[1])
        if (p!="power") and (p not in predictors):
            raise EngineError(p + " is not a predictor", False)
        pairs[p] = v

    sensor_values = np.zeros(len(predictors), float)

    power = 0 #STATUS_OFF
    if pairs.has_key('power'):
        if float(pairs['power']) > POWER_CUT:
            power = 1 #STATUS_ON

    for i in range(len(predictors)):
        p = predictors[i]
        if not pairs.has_key(p):
            raise EngineError("the \'" + p + "\' variable is missing", False)
        sensor_values[i] = pairs[p]

    return [power, sensor_values.reshape(1,-1)]

def print_confusion_matrix(c, labels):
    n = c.shape[0]
    s = "act\pred"
    for l in labels:
        s += "\t" + l
    print s
    for i in range(n):
        s=labels[i] + "\t"
        for j in range(n):
            s += "\t" + str(c[i][j])
        print s

def process(args):
    global log_to_file
    global SENSOR_CSV

    if args.log:
        log_to_file = True
        logging.basicConfig(filename=args.log, level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    if (args.command == 'process'):
        from modeling import process_data

        if not args.csv_file:
            args.csv_file = SENSOR_CSV
        #process the raw data
        df = process_data(args.csv_file)
        #save data to file
        save_data(df, args.data_file)

    elif (args.command == 'train'):
        from modeling import process_data
        from modeling import train_model

        if (args.csv_file): #input is csv file
            #process the raw data
            df = process_data(args.csv_file)
            #save data to file
            save_data(df, args.data_file)
        else: #input is processed data file
            #load the processed data
            df = load_data(args.data_file)

        #for aircon in ON status, build model to predict TURN OFF action
        on_model = train_model(df[0], df[1], args.classifier)

        #for aircon in OFF status, build model to predict TURN ON action
        off_model = train_model(df[2], df[3], args.classifier)

        #save models to file
        save_model([on_model, off_model], args.model_file)

    elif (args.command == 'predict'):
        from modeling import ACTION_TURN_OFF
        from modeling import ACTION_TURN_ON
        from modeling import ACTION_NOTHING
        #parse the input values from sensors
        status, inputs = parse_sensors(args.sensors)

        #load prediction models
        on_model, off_model = load_model(args.model_file)

        if status == 0:     #aircon is OFF, predict TURN ON
            action = predict(off_model, inputs)
        else:               #aircon is ON, predict TURN OFF
            action = predict(on_model, inputs)

        if action==ACTION_TURN_ON:
            print "TURN_ON"
        elif action==ACTION_TURN_OFF:
            print "TURN_OFF"
        else:
            print "DO_NOTHING"

    elif (args.command == 'evaluate'):
        import modeling

        if (args.csv_file):     #input is raw data
            #process raw data
            df = modeling.process_data(args.csv_file)
        else:                   #input is processed data
            #load processed data
            df = load_data(args.data_file)


        print "\n\nPerformance for TURN ON prediction"

        con_mats =  modeling.evaluate_model(df[2], df[3], args.classifier)
        fold = 1
        for c1, c2 in con_mats:
            print "\nPrediction performance for fold " + str(fold)
            print "\n... on training data"
            print_confusion_matrix(c1, ("NOTHING", "TURN-ON"))
            print "\n... on testing data"
            print_confusion_matrix(c2, ("NOTHING", "TURN-ON"))
            fold += 1

        print  "\n\nPerformance for TURN OFF prediction"

        #cross validation evaluation
        con_mats = modeling.evaluate_model(df[0], df[1], args.classifier)

        #report the performance
        fold = 1
        for c1, c2 in con_mats:
            print "\nPrediction performance for fold " + str(fold)
            print "\n... on training data"
            print_confusion_matrix(c1, ("DO-NOTHING", "TURN-OFF"))
            print "\n... on testing data"
            print_confusion_matrix(c2, ("DO-NOTHING", "TURN-OFF"))
            fold += 1

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
        parser.add_argument("-c", "--classifier", choices=['tree', 'forest'], dest="classifier", default="forest",
                            help="Select classification model (Decision tree or Random forest)")
        parser.add_argument("-t", "--csv_file", dest="csv_file",
                            help="file containing the original sensor data, default name = \'" + SENSOR_CSV + "\'")
        parser.add_argument("-d", "--data_file", dest="data_file", default=SENSOR_PICKLE,
                            help="file to save/load the processed sensor data, default name = \'" + SENSOR_PICKLE + "\'")
        parser.add_argument("-m", "--model", dest="model_file", default=MODEL_PICKLE,
                            help="file to save/load the prediction model, default name = \'" + MODEL_PICKLE + "\'")
        parser.add_argument("-s", "--sensors", dest="sensors",
                            help="sensor data for which prediction should be made, here is a sample format: "
                                 "\"power=1.2, temp=37, humidity=50, dust=100, CO2=1000, light=30, day=2, hour=19.5\"")
        parser.add_argument("-l", "--log", dest="log", help="where to save log messages")
        args = parser.parse_args()

        process(args)
    except EngineError as e:
        sys.stderr.write("Engine Error: " + str(e) + "\n")
    except Exception as e:
        sys.stderr.write("Internal Error: " + str(e) + "\n")

if __name__ == "__main__" : main()
