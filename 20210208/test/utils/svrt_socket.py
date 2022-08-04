from threading import Thread
from threading import Lock
from queue import Queue
import socket
import time

def send_thread(sock, q, lock) :
    """
    """
    while True:
        lock.acquire()
        if q.empty():
            lock.release()
            time.sleep(0.1)
        else:
            lock.release()
            sendData = q.get()
            sock.send(sendData.encode())
            time.sleep(0.1)

def recv_thread(sock, q, lock) :
    """
    """
    while True:
        recvData = sock.recv(1024)
        print('client :', recvData.decode())
        lock.acquire()
        q.put(recvData)
        lock.release()
        time.sleep(0.1)

def error_handling(problem):
    """
    """
    raise Exception(problem, "Erorr")
