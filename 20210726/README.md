# SVRT PROJECT (Python version)

- 이전 버전이 설치되어 있다면 맨 아래로



## Matrix Kernel Modules

```bash
sudo apt install vim
sudo raspi-config
sudo apt-get update
sudo apt-get upgrade

wget -q --show-progress -O rpi_kernel_5-4.deb http://archive.raspberrypi.org/debian/pool/main/r/raspberrypi-firmware/raspberrypi-kernel_1.20201126-1_armhf.deb
wget -q --show-progress -O rpi_kernel_headers_5-4.deb http://archive.raspberrypi.org/debian/pool/main/r/raspberrypi-firmware/raspberrypi-kernel-headers_1.20201126-1_armhf.deb
sudo apt purge raspberrypi-kernel-headers
sudo apt install ~/rpi_kernel_5-4.deb
sudo apt install ~/rpi_kernel_headers_5-4.deb
sudo apt-mark hold raspberrypi-kernel
sudo apt-mark hold raspberrypi-kernel-headers
sudo reboot

curl https://apt.matrix.one/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.matrix.one/raspbian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/matrixlabs.list
sudo apt update
sudo apt install matrixio-kernel-modules
sudo reboot
```



## DeepSpeech

```bash
sudo apt install git python3-pip python3-scipy python3-numpy python3-pyaudio libatlas3-base
pip3 install deepspeech --upgrade
pip3 install halo webrtcvad --upgrade
sudo apt install portaudio19-dev

curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
mv deepspeech-0.9.3-models.tflite deepSpeech/
mv deepspeech-0.9.3-models.tflite deepSpeech/
```



## Firebase & Pyusb & Scikit-Learn

```bash
pip3 install --upgrade firebase-admin
pip3 install pyusb
pip3 install sklearn
sudo apt-get install exfat-fuse exfat-utils
```



## Matrix Library

```bash
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



## Pytorch

```bash
wget https://wintics-opensource.s3.eu-west-3.amazonaws.com/torch-1.3.0a0%2Bdeadc27-cp37-cp37m-linux_armv7l.whl
pip3 install torch-1.3.0a0%2Bdeadc27-cp37-cp37m-linux_armv7l.whl
```



## 이전 버전이 설치되어 있는 경우

```bash
cd alpha_project
git pull

cd 20210324
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite
#curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

mv deepspeech-0.9.3-models.tflite deepSpeech/
#mv deepspeech-0.9.3-models.scorer deepSpeech/

pip3 install --upgrade firebase-admin
pip3 install pyusb
pip3 install pandas

cd ~/
mkdir usb
pip3 install sklearn
sudo apt-get install exfat-fuse exfat-utils
sudo python3 -m pip install matrix-lite

wget https://wintics-opensource.s3.eu-west-3.amazonaws.com/torch-1.3.0a0+deadc27-cp37-cp37m-linux_armv7l.whl
pip3 install torch-1.3.0a0%2Bdeadc27-cp37-cp37m-linux_armv7l.whl
sudo reboot
```

```bash
cd alpha_project/20210324

# 아래의 명령으로 프로그램 실행
./svrt.sh
# 종료할 때는 터미널에 1을 입력
```

