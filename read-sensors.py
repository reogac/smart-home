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

class MyThread(Thread, my_name):
    def __init__(self, name):
        self.alive = True
        self.my_name = my_name
        Thread.__init__(self)

    def set_kill(self):
        print "kill " + self.my_name + " thread!"
        self.alive = False

class CommandSender(MyThread):
    def __init__(self, command):
        self.command = command
        MyThread.__init__(self)

    def run(self):
        while self.alive:
            self.command.put(AMBIENT_SENSORS)
            time.sleep(PROBING_INTERVAL)

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
        MyThread.__init__(self)

    def run(self):
        self.port.flush() #clearing input buffer
        bf = ""
        while (self.alive):
            c = self.command.get()
            if c==AMBIENT_SENSORS:
                #Tell the port that I want more data
                self.port.write(FW_KEY)
                print "write"
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


class Predictor(MyThread):
    def __init__(self):
        self.data = Queue.Queue(100)
        command = Queue.Queue(100)
        self.command_sender = CommandSender(command)
        self.port_reader_writer = ReaderWriter(command, self.data)
        MyThread.__init(self)

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
    def run(self):
        self.command_sender.start()
        self.port_reader_writer.start()

        while self.alive:
            try:
                msg = self.data.get(timeout=1)
                sensors = self.parse_data(msg)

            except (KeyboardInterrupt, SystemExit):
                print "interupted"
                self.command_sender.set_kill()
                time.sleep(1)
                self.port_reader_writer.set_kill()
                break
            except Queue.Empty as e:
                print "pass the empty queue exception"
                pass


pred = Predictor()
pred.start()
pred.join()


