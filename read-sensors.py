#!/usr/bin/python

from threading import Thread
import time
import serial
import Queue
from datetime import datetime
import pickle as pk
import numpy as np

PROBING_INTERVAL = 1 #reading at 1sec interval. It should be not to small
SAVING_INTERVAL = 300 
FW_KEY = "(02)"
NUM_SENSORS = 5
PREDICTION_INTERVAL = 600
SENSOR_DATA_BUFFER_SIZE = 10
SENSOR_DATA_FILE_NAME = "sensor-raw-data.csv"
MODEL_FILE_NAME = "model.pk"

class MyThread(Thread):
    def __init__(self, my_name):
        self.alive = True
        self.my_name = my_name
        Thread.__init__(self)

    def set_kill(self):
        print "kill " + self.my_name + " thread!"
        self.alive = False

class ReaderWriter(MyThread):
    def __init__(self, data):
        self.port = serial.Serial(
                    port='/dev/ttyO2',
                    baudrate=9600, timeout=0,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
        self.queue = data
        MyThread.__init__(self, "Port reader/writer")

    def run(self):
        self.port.flush() #clearing input buffer
        bf = ""
        while self.alive:

            #Tell the port that I want more data
            self.port.write(FW_KEY)

            while self.port.inWaiting():
                bf += self.port.read()
            l = len(bf)
            if (l >= 32) :

                self.queue.put(bf[0:32])
                bf = bf[32:]
            else:
                #print "got this: " + bf
                pass
            time.sleep(PROBING_INTERVAL)
        self.port.close()
        print self.my_name + " says BYE"


class Predictor:
    PREDICTORS = ('temp', 'humidity', 'light', 'CO2', 'dust')
    NUM_FEATURES = 7
    def __init__(self):
        self.last_prediction_time = datetime.now()
        self.on_model, self.off_model = pk.load(open(MODEL_FILE_NAME, 'rb'))
        self.input = np.zeros(Predictor.NUM_FEATURES, float)

    def handle_data(self, data):
        current_time = datetime.now()
        time_diff = current_time - self.last_prediction_time
        if time_diff.total_seconds() >= PREDICTION_INTERVAL:
            for i in range(len(Predictor.PREDICTORS)):
                self.input[i] = data[Predictor.PREDICTORS[i]]
            pred_time = data["time"]
            self.input[Predictor.NUM_FEATURES-2] = pred_time.weekday()
            self.input[Predictor.NUM_FEATURES-2] = pred_time.hour + pred_time.minute/60.0 + pred_time.second/3600.0

            print "Make a prediction with inputs = " + str(self.input)
            #make a prediciton here
            if data["power"] > 0.0:
                pred_label = self.on_model.predict(self.input)
            else:
                pred_label = self.off_model.predict(self.input)
            if pred_label == 0:
                print "DO_NOTHING"
            elif pred_label == 1:
                print "TURN_ON"
            else:
                print "TURN_OFF"

            self.last_prediction_time = current_time


class SensorDataManager:
    def __init__(self):
        self.last_saving_time = datetime.now()
        self.buffer=[]
        with open(SENSOR_DATA_FILE_NAME, "w") as f:
            #f.write("time,power,temp,humidity,light,CO2,dust\n")
            f.write("time,temp,humidity,light,dust\n")

    def handle_data(self, data):
        current_time = datetime.now()
        time_diff = current_time - self.last_saving_time
        if time_diff.total_seconds() >= SAVING_INTERVAL:
            print "collecting data: " + str(data)
            self.buffer.append(data)
            self.last_saving_time = current_time

    def save(self):
        if len(self.buffer) >= SENSOR_DATA_BUFFER_SIZE:
            self.flush()

    def flush(self): #save data to disk
        print "dump data to disk"
        with open(SENSOR_DATA_FILE_NAME, "a") as f:
            for data in self.buffer:
                #f.write(str(data["time"])+","+str(data["power"])+","
                #        + str(data["temp"])+","+str(data["humidity"])+","
                #        + str(data["light"]) + "," + str(data["CO2"])+","+str(data["dust"]) + "\n")
                f.write(str(data["time"]) + str(data["temp"])+","+str(data["humidity"])+","
                        + str(data["light"]) + "," + str(data["dust"]) + "\n")

        self.buffer = []


class Framework:
    def __init__(self):
        self.sensor_data = Queue.Queue(100) #queue to get sensor data

        self.port_reader_writer = ReaderWriter(self.sensor_data) #port read/write thread
        self.predictor = Predictor() #predictor
        self.data_manager = SensorDataManager()

    def parse_sensor_data(self, data):
        sensors={}
        try:
            sensors["time"] = datetime.now()
            sensors["dust"] = 0.01*int(data[10:15])
            sensors["temp"] = 0.1*int(data[17:21])
            sensors["humidity"] = int(data[23:26])
            sensors["light"] = int(data[28:31])
            sensors["CO2"] = 50
            sensors["power"] = 0.0
        except ValueError as e:
            print e

        return sensors

    def run(self):
        self.port_reader_writer.start()
        while True:
            try:
                data = self.sensor_data.get(timeout=PROBING_INTERVAL)
                data = self.parse_sensor_data(data)
                self.data_manager.handle_data(data) #save data to disk
                self.predictor.handle_data(data) #make prediction

            except (KeyboardInterrupt, SystemExit):
                print "user interupt"
                self.data_manager.flush()
                self.port_reader_writer.set_kill() #must be killed before the command sender thread
                break
            except Queue.Empty as e:
                pass

        print "BYE"


framework = Framework()
framework.run()
