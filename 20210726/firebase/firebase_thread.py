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


def get_command_thread(ent, to_td_deq, email_deq) :
    # condition
    #
    ent.wait()
    email_id = email_deq.pop()
    doc_ref = get_doc_ref(email_id)
    doc_watch = doc_ref.on_snapshot(on_snapshot)
    while True:
        if update_ent.isSet():
            print("update_ent is set!")
            new_trigger = tri_deq.popleft()
            to_td_deq.append(new_trigger)
            update_ent.clear()
        else:
            time.sleep(2)

def get_doc_ref(email_id):
    """
    initialize Firebase Admin SDK
    set return value as user's document root
    """
    cred = credentials.Certificate("firebase/cred/alpha.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db.collection(u'setting').document(email_id)

def on_snapshot(doc_snapshot, changes, read_time):
    """
    Create a callback on_snapshot function to capture changes
    """
    for doc in doc_snapshot:
        cmd_txt = doc.to_dict()["Command_text"]
        print(f'Command changed : {cmd_txt}')
        tri_deq.append(cmd_txt)
        update_ent.set()
