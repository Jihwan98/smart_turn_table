import serial

ser = serial.Serial("COM3", 9600, timeout = 1)
while True:
    print("insert op :", end=' ')
    op = input()
    ser.write(op.encode())
    print("R: ", ser.readline())

    if op == 'q':
        print("serial close\n")
        ser.close()
        break