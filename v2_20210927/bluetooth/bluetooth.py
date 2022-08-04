import os
import time

def bluetooth_thread(email_deq, loading):
    try:
        with open("user_email.txt", 'r') as f:
            past_email = f.readline().split("=")[-1]
            email_deq.append(past_email)
    except:
        past_email = ""
    
    start_time = time.time()
    while True:
        if time.time() - start_time > 170:
            os.system("bluetooth/scripts/discoverable.sh")
            start_time = time.time()
        try:
            with open("user_email.txt",'r') as f:
                email_id = f.readline().split("=")[-1]

            if email_id != past_email:
                print("new email detected")
                email_deq.append(email_id)
                loading.put(1)
                os.system("bluetooth/scripts/wifi.sh")
                past_email = email_id
                
                if len(email_deq) > 5000:
                    email_deq.clear()
                    email_deq.append(email_id)
                time.sleep(5)
                loading.put(0)

        except KeyboardInterrupt:
            break

        except:
            pass

def start_bluetooth():
    os.system("bluetooth/scripts/discoverable.sh")
    os.system("bluetooth/scripts/bt")
