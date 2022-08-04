sudo chmod 755 wpa_supplicant.conf
sudo chown root wpa_supplicant.conf
sudo mv wpa_supplicant.conf /etc/wpa_supplicant/wpa_supplicant.conf
wpa_cli -i wlan0 reconfigure
sudo umount /dev/sda1
