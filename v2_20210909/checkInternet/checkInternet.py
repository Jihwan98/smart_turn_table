import socket
import time

def checkInternetSocket(host="8.8.8.8", port=53, timeout=3):
	try:
	    socket.setdefaulttimeout(timeout)
	    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
	    return True
	except socket.error as ex:
	    print(ex)
	    return False
def checkInternet_thread(internet, check_internet_ent, internet_ent):
    while True:
        if checkInternetSocket():
            internet.put(1)
            internet_ent.set()
        else:
            internet.put(0)
        check_internet_ent.set()
        n = internet.qsize()
        flag = internet.queue[-1]
        if n > 1000:
            internet.queue.clear()
            internet.put(flag)
        time.sleep(1)

if __name__ == "__main__":
    if checkInternetSocket():
        print("Connected")
    else:
        print("Not")
