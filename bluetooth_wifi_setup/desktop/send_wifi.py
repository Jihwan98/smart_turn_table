from typing import final
import serial
import serial.tools.list_ports as lp
import subprocess
import sys
import time

try:
    # wifi id 가져오기
    _, interface_output = subprocess.getstatusoutput("netsh wlan show interface")
    
    profile = interface_output.split("프로필")[-1].split("\n")[0].split(":")[-1].strip()
    print("wifi id :", profile)
    
    # wifi pw 가져오기
    _, password_output = subprocess.getstatusoutput("netsh wlan show profiles {} key=clear".format(profile))
    password = password_output.split("키 콘텐츠")[-1].split("\n")[0].split(":")[-1].strip()  
    print("wifi pw :", password)

    wifi = [profile, password]
    
    # 포트 확인
    ports = lp.comports()

    if not ports:
        sys.exit("활성화 된 COM포트가 없습니다")

    for i in ports:
        print(i)

    # 사용할 포트
    port = ports[0].device
    print("사용할 포트 :", port)

    # 시리얼 통신
    ser = serial.Serial(port, 9600, timeout = 1)
    for i in wifi:
        ser.write(i.encode())
        # print("R: ", ser.readline())

    print("전송 완료")
    ser.close()
    
except KeyboardInterrupt:
    print("Keyboard Interrupt")
    
except Exception as ex:
    print(ex)
    
finally:
    print("3초 후 창이 닫힙니다.")
    time.sleep(3)