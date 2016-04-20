from threading import Thread
import time
import serial
import Queue
from datetime import datetime
import numpy as np

PROBING_INTERVAL = 1 #reading at 1sec interval. It should be not to small
SAVING_INTERVAL_NUM = 5
FW_KEY = "(02)"
NUM_SENSORS = 5
AMBIENT_SENSORS = 1

def parse_data(data):
    sensors={}
    try:
        sensors["dust"] = 0.01*int(data[10:15])
        sensors["temp"] = 0.1*int(data[17:21])
        sensors["humidity"] = int(data[23:26])
        sensors["light"] = int(data[28:31])
        sensors["time"] = str(datetime.now())
        print sensors
    except ValueError as e:
        print e
    return sensors

class MyThread(Thread):
    def __init__(self, my_name):
        self.alive = True
        self.my_name = my_name
        Thread.__init__(self)

    def set_kill(self):
        print "kill " + self.my_name + " thread!"
        self.alive = False

class CommandSender(MyThread):
    def __init__(self, command):
        self.command = command
        MyThread.__init__(self, "Command sender")

    def run(self):
        while self.alive:
            self.command.put(AMBIENT_SENSORS)
            time.sleep(PROBING_INTERVAL)
        print self.my_name + " says BYE"

class ReaderWriter(MyThread):
    def __init__(self, command, data):
        self.port = serial.Serial(
                    port='/dev/ttyO2',
                    baudrate=9600, timeout=0,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
        print self.port.isOpen()
        self.queue = data
        self.command = command
        MyThread.__init__(self, "Port reader/writer")

    def run(self):
        self.port.flush() #clearing input buffer
        bf = ""
        while self.alive:
            c = self.command.get()
            if c==AMBIENT_SENSORS:
                #Tell the port that I want more data
                self.port.write(FW_KEY)

            while self.port.inWaiting():
                bf += self.port.read()
            l = len(bf)
            if (l >= 32) :
                print bf
                self.queue.put(bf[0:32])
                bf = bf[32:]
            else:
                print "got this: " + bf

        self.port.close()
        print self.my_name + " says BYE"


class Predictor(MyThread):
    def __init__(self, input_queue):
        self.input_queue = input_queue
        MyThread.__init__(self, "Predictor")

    def run(self):
        while self.alive:
            try:
                sensors = self.input_queue.get(timeout=1)
                #now make a prediction
                print "predict this: " + str(sensors)
            except Queue.Empty as e:
                pass

        print self.my_name + " says BYE"

class SensorDataManager:
    def __init__(self):
        pass
    def handle_data(selfdata):
        #store data?
        #preprocess?
        #prepare
        pass


class Framework:
    def __init__(self):
        self.sensor_data = Queue.Queue(100) #queue to get sensor data
        self.port_command = Queue.Queue(100) #queue of commands to be sent to port
        self.command_sender = CommandSender(port_command) #port command sender thread
        self.port_reader_writer = ReaderWriter(port_command, sensor_data) #port read/write thread
        self.pred_input = Queue.Queue(100)
        self.pred = Predictor(pred_input) #prediction thread
    def run(self):
        self.command_sender.start()
        self.port_reader_writer.start()
        self.pred.start()

        while True:
            print "hahah"
            try:
                data = self.sensor_data.get(timeout=PROBING_INTERVAL)
                self.pred_input.put(parse_data(data))
            except (KeyboardInterrupt, SystemExit):
                print "user interupt"
                self.pred.set_kill() #kill this thread first
                self.port_reader_writer.set_kill() #must be killed before the command sender thread
                time.sleep(1)
                self.command_sender.set_kill()
                break
            except Queue.Empty as e:
                pass

        print "BYE"


framework = Framework()
framework.run()
