import socket
import time

try:
    import httplib
except:
    import http.client as httplib
def checkInternetHttplib(url="www.google.com", timeout=3):
    conn = httplib.HTTPConnection(url, timeout=timeout)

    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except Exception as e:
        print(e)
        return False
def checkInternet_thread(internet, check_internet_ent, internet_ent):
    while True:
        if checkInternetHttplib():
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
    if checkInternetHttplib():
        print("Connected")
    else:
        print("Not")






