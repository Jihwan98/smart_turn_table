import subprocess

interface_output = subprocess.getstatusoutput("netsh wlan show interface")
index1 = interface_output[1].index("SSID")
ssid = interface_output[1][index1:].split(":")[1].split("\n")[0].strip()

print(ssid)

password_output = subprocess.getstatusoutput("netsh wlan show profiles {} key=clear".format(ssid))
index2 = password_output[1].index("키 콘텐츠")
password = password_output[1][index2:].split(":")[1].split("\n")[0].strip()

print(password)

wifi = [ssid, password]
print(wifi)