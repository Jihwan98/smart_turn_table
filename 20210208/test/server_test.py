import socket
from threading import Thread
from threading import Lock
from queue import Queue
import time
from utils import draw_logo, svrt_thread


if __name__ == '__main__' :
    
    email_id = "test@test.com"
    # doc_ref = svrt_thread.get_doc_ref(email_id)
    
    changed = [[0]]
    
    ds2odas_lock = Lock()
    odas2ds_lock = Lock()
    
    ds2odas = Queue()
    odas2ds = Queue()
    
    host = '127.0.0.1'
    port = 40000
    
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    
    ds2odas_sock, ds2odas_addr = server_sock.accept()
    print('DeepSpeech accessed...')
    #odas2ds_sock, odas2ds_addr = server_sock.accept()
    # print('ODAS accessed...')
    
    print('Access Complete...')
    draw_logo.draw_logo()

    recv_from_d = Thread(target=svrt_thread.recv_thread, args=(ds2odas_sock, ds2odas, ds2odas_lock))
    send_d2o = Thread(target=svrt_thread.send_thread, args=(ds2odas_sock, ds2odas, ds2odas_lock))
    # recv_from_o = Thread(target=svrt_thread.recv_thread, args=(odas2ds_sock, odas2ds, odas2ds_lock))
    # send_o2d = Thread(target=svrt_thread.send_thread, args=(odas2ds_sock, odas2ds, odas2ds_lock))
    
    usb_detection = Thread(target=svrt_thread.usb_detection_thread, args=(changed))
    get_command = Thread(target=svrt_thread.get_command, args=(changed, email_id))

    recv_from_d.start()
    send_d2o.start()
    # recv_from_o.start()
    # send_o2d.start()
    usb_detection.start()
    # doc_watch = doc_ref.on_snapshot(svrt_thread.on_snapshot)
    get_command.start()
    
    recv_from_d.join()
    send_d2o.join()
    # recv_from_o.join()
    # send_o2d.join()
    usb_detection.join()
    get_command.join()
    
    odas2ds_sock.close()
    ds2odas_sock.close()
    server_sock.close()
    
    print("Quit SVRT")

