"""
@author: Kojungbeom
    
"""

import sys
import os
from threading import Event
import time
import usb.core


def usb_detection_thread(ent, email_deq) :
    """This thread detect usb installation, mounts the usb,
    reads wifi information and user email from the "config.txt" file.
    user email is sent to the firebase_thread

    Args:
        ent(event): event object indicates wifi is connected.
        email_deq(deque): deque to send e-mail address to firebase thread.
        
    """
    conf = {}
    busses = usb.busses()
    while True:
        #dev = usb.core.find(idVendor=0x0781, idProduct=0x5590)
        busses = usb.busses()
        devices = [i for i in (bus.devices for bus in busses)]
        usb_num = 0
        for i in devices:
            usb_num += len(i)

        if usb_num==5 and not ent.isSet():
            # mount usb
            print("Start Mount")
            time.sleep(5)
            os.system("usb_detect/scripts/usb_mount.sh")
            time.sleep(5)
            # read config file
            with open('/home/pi/usb/config.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split('=')
                    if line[0] == "wifi_id":
                        conf["wifi_id"] = line[1][:-1]
                    elif line[0] == "wifi_pw":
                        conf["wifi_pw"] = line[1][:-1]
                    elif line[0] == "email":
                        conf["email"] = line[1][:-1]
                    elif line[0] == "command":
                        conf["command"] = line[1][:-1]
            # write new wpa_supplicant.conf to connect with wifi.          
            with open('wpa_supplicant.conf', 'w') as f:
                f.write("ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n")
                f.write("update_config=1\n")
                f.write("country=KR\n")
                f.write("network={\n")
                f.write('\tssid="' +  conf["wifi_id"] + '"\n')
                f.write('\tpsk="' + conf["wifi_pw"] + '"\n')
                f.write("}")
            # append email address to send
            email_deq.append(conf["email"])    
            
            # wifi connection
            os.system("usb_detect/scripts/wifi.sh")
            time.sleep(15)
            print("Network Connected")
            ent.set()
        else:
            time.sleep(1)

        if usb_num == 4:
            ent.clear()
            

