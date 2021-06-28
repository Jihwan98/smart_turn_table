import serial
import serial.tools.list_ports as lp
import subprocess
import sys

try:
    # wifi id 가져오기
    interface_output = subprocess.getstatusoutput("netsh wlan show interface")
    index1 = interface_output[1].index("SSID")
    ssid = interface_output[1][index1:].split(":")[1].split("\n")[0].strip()

    print("wifi id :", ssid)

    # wifi pw 가져오기
    password_output = subprocess.getstatusoutput("netsh wlan show profiles {} key=clear".format(ssid))
    index2 = password_output[1].index("키 콘텐츠")
    password = password_output[1][index2:].split(":")[1].split("\n")[0].strip()

    print("wifi pw :", password)

    wifi = [ssid, password]

    # 포트 확인
    ports = lp.comports()

    if not ports:
        sys.exit("활성화 된 COM포트가 없습니다")

    for i in ports:
        print(i)

    # 사용할 포트
    port = ports[0].device
    print("사용할 포트 :", port)

    ser = serial.Serial(port, 9600, timeout = 1)
    for i in wifi:
        ser.write(i.encode())
        # print("R: ", ser.readline())

    print("serial close")
    ser.close()

except KeyboardInterrupt:
    print("Keyboard Interrupt")
    ser.close()

except Exception as ex:
    print(ex)
    ser.close()