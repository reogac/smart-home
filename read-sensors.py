from threading import Thread
import time
import serial
import Queue
from datetime import datetime
PROBING_INTERVAL = 1 #reading at 1sec interval. It should be not to small
SAVING_INTERVAL_NUM = 5
FW_KEY = "(02)"

class CommandSender(Thread):
    def __init__(self, command):
        self.command = command
        self.alive = True
        Thread.__init__(self)

    def run(self):
        while self.alive:
            self.command.put(1)
            time.sleep(PROBING_INTERVAL)
    def set_kill(self):
        print "kill command sender thread"
        self.alive = False

class ReaderWriter(Thread):
    def __init__(self, command, data):
        self.port = serial.Serial(
                    port='/dev/ttyO2',
                    baudrate=9600, timeout=0,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
        print self.port.isOpen()
        self.alive = True
        self.queue = data
        self.command = command
        Thread.__init__(self)

    def run(self):
        self.port.flush() #clearing input buffer
        bf = ""
        while (self.alive):
            c = self.command.get()
            if c==1:
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

    def set_kill(self):
        print "kill reader-writer thread"
        self.alive = False
 
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


data = Queue.Queue(100)
command = Queue.Queue(100)
cs = CommandSender(command)
rw = ReaderWriter(command, data)
rw.start()
cs.start()


while True:
    try:
        msg = data.get(timeout=1)       
        sensors = parse_data(msg)
    except (KeyboardInterrupt, SystemExit):
        print "interupted"
        rw.set_kill()
        time.sleep(1)
        cs.set_kill()
        break
    except Queue.Empty as e:
        print "pass the empty queue exception"
        pass

