import threading
from networktables import NetworkTables
import cv2

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

NetworkTables.initialize(server='10.99.97.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

# Insert your processing code here
print("Connected!")

table = NetworkTables.getTable('SmartDashboard')

foo = table.putBoolean('target', True)

print("its on the table")
	
while(True):
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break