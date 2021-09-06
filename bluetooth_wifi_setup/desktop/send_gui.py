import serial
import serial.tools.list_ports as lp
import subprocess
import sys
import time

import tkinter.messagebox as msgbox
from tkinter import *

def getstatusoutput(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    out, _ = process.communicate()

    if out[-1:] == b'\n':
        out = out[:-1]

    out = out.decode('cp949')

    return (process.returncode, out)

def btnSend(event):
    try:
        email_id = email_ent.get().strip()
        print("email id :",email_id)
        if email_id:
            # wifi id 가져오기
            _, interface_output = getstatusoutput("netsh wlan show interface")
            
            profile = interface_output.split("프로필")[-1].split("\n")[0].split(":")[-1].strip()
            print("wifi id :", profile)
            
            # wifi pw 가져오기
            _, password_output = getstatusoutput("netsh wlan show profiles {} key=clear".format(profile))
            password = password_output.split("키 콘텐츠")[-1].split("\n")[0].split(":")[-1].strip()  
            print("wifi pw :", password)


            #wifi = [profile, password]
            data = [email_id, profile, password]
            # 포트 확인
            ports = lp.comports()

            if not ports:
                sys.exit("활성화 된 COM포트가 없습니다")

            #for i in ports:
                # print(i)

            # 사용할 포트
            port = ports[0].device
            print("사용할 포트 :", port)

            # 시리얼 통신
            ser = serial.Serial(port, 9600, timeout = 1)
            for i in data:
                ser.write(i.encode())
                # print("R: ", ser.readline())

            print("전송 완료")
            ser.close()
            msgbox.showinfo("알림", "전송 완료")
        else:
            print("email id를 입력해주세요")
            msgbox.showinfo("알림", "email id를 입력해주세요")
    except Exception as ex:
        print(ex)
        msgbox.showinfo("알림", "에러 발생 : {}".format(ex))

width = 200
height = 200

root = Tk()
root.title("Device Connection Setting")
root.geometry("{}x{}+{}+{}".format(width,height,int(1920/2-width/2),int(1080/2-height/2)))
root.resizable(False, False)

label1 = Label(root, text="email id를 입력해주세요")
label1.place(x=width/2-75, y=height/2-50)

email_ent = Entry(root, width=20)
email_ent.bind("<Return>", btnSend)
email_ent.place(x=width/2-75, y=height/2-25)

send_btn = Button(root, width=5, text="전송", command=btnSend)
send_btn.place(x=width/2-50, y=height-100)
quit_btn = Button(root, width=5, text="close", command=root.destroy)
quit_btn.place(x=width/2, y=height-100)

root.mainloop()
