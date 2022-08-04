"""
@author: Google
    Edited by Jihwan
    
reference
"""

from threading import Thread, Lock, Event
from firebase_admin import credentials
from firebase_admin import firestore
import firebase_admin
import collections
import time

update_ent = Event()
tri_deq = collections.deque()




mode_ent = Event()          # mode 바뀐거 인식
mode_record = Event()        # mode record 실시

mode_deq = collections.deque()      # mode security & normal 이름 받기





def get_command_thread(ent, trigword_deq, email_deq, device_mode) :
    # condition
    #
    ent.wait()
    email_id = email_deq[-1]
    doc_ref = get_doc_ref(email_id)
    doc_watch = doc_ref.on_snapshot(on_snapshot)
    while True:
        if email_deq[-1] != email_id:
            email_id = email_deq[-1]
            doc_ref = get_doc_ref(email_id)
            doc_watch = doc_ref.on_snapshot(on_snapshot)
        
        if update_ent.isSet():
            new_trigger = tri_deq.popleft()
            if len(trigword_deq) > 50:
                trigword_deq.clear()
            trigword_deq.append(new_trigger)
            update_ent.clear()
        else:
            time.sleep(2)



        if mode_ent.isSet() :
            if mode_deq[-1] == "security" :
                print("security mode!")
                device_mode.put('security')
                mode_ent.clear()
                mode_record.set()
            elif mode_deq[-1] == 'normal' :
                print("normal mode!")
                device_mode.put('normal')
                mode_ent.clear()
            elif mode_deq[-1] == 'training':
                print("training mode!")
                device_mode.put('training')
                mode_ent.clear()
            else:
                mode_ent.clear()
            
            n = device_mode.qsize()
            last_mode = device_mode.queue[-1]
            if n > 100:
                device_mode.queue.clear()
                device_mode.put(last_mode)
        time.sleep(0.001)

def get_doc_ref(email_id):
    """
    initialize Firebase Admin SDK
    set return value as user's document root
    """
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase/cred/alpha.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db.collection(u'setting').document(email_id)



pre_mode = "normal"
pre_command = "eddy"

def on_snapshot(doc_snapshot, changes, read_time):
    """
    Create a callback on_snapshot function to capture changes
    """
    global pre_mode
    global pre_command
    
    for doc in doc_snapshot:
        cmd_txt = doc.to_dict()["Command_text"]
        cmd_mode = doc.to_dict()["Mode"]
        print(f'cmd_txt : {cmd_txt}, cmd_mode : {cmd_mode}')

        if pre_mode != cmd_mode:
            print("Mode change!")
            mode_deq.append(cmd_mode)
            pre_mode = cmd_mode
            mode_ent.set()

        if pre_command != cmd_txt:
            print(f'Command changed : {cmd_txt}')
            mode_deq.append("training")
            mode_ent.set()
            tri_deq.append(cmd_txt)
            pre_command = cmd_txt
            update_ent.set()
