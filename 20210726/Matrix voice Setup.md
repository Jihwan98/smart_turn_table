# Matrix voice Setup

```
curl https://apt.matrix.one/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.matrix.one/raspbian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/matrixlabs.list
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install matrixio-creator-init libmatrixio-creator-hal libmatrixio-creator-hal-dev
sudo reboot

sudo apt-get install python3-pip
python3 -m pip install --upgrade pip
sudo python3 -m pip install matrix-lite
sudo reboot
```

```
pip3 install --upgrade firebase-admin
pip install scikit-learn==0.22.2.post1
sudo apt-get install exfat-fuse exfat-utils
pip3 install pyusb
pip3 install numpy==1.19.5
pip3 install pandas==1.1.5
```

```
pip3 install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl
sudo apt-get install libopenblas-base
export LD_LIBRARY_PATH=/usr/lib/openblas-base/
```

