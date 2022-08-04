# SVRT System Raspberry Pi Setup

1. [ODAS](#odas)
2. [DeepSpeech](#deepspeech)
3. [Matrix](#matrix)
4. [Master](#master)

<br>

## ODAS

### Install Matrix Software

```batch
# Add repo and key
$ curl https://apt.matrix.one/doc/apt-key.gpg | sudo apt-key add -
$ echo "deb https://apt.matrix.one/raspbian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/matrixlabs.list

# Update packages and install
$ sudo apt-get update
$ sudo apt-get upgrade

# Installation
$ sudo apt install matrixio-creator-init
$ sudo apt install libmatrixio-creator-hal
$ sudo apt install libmatrixio-creator-hal-dev
$ sudo reboot
```

After reboot, install the MATRIX Kernel Modules as follows:

```batch
$ sudo apt install matrixio-kernel-modules
$ sudo reboot
```

<br>

### Install External Libraries

```batch
$ sudo apt-get install g++ git cmake
$ sudo apt-get install libfftw3-dev
$ sudo apt-get install libconfig-dev
$ sudo apt-get install libasound2-dev
$ sudo apt install libjson-c-dev
```

<br>

### Compile ODAS

Clone our repository.

```batch
$ git clone https://github.com/hyunchul78/alpha_project.git
```

Create a folder to build the system and build it

```batch
$ cd alpha_project/20210208/odas
$ mkdir build
$ cd build
$ cmake ..
$ make
```



## DeepSpeech

### Install DeepSpeech package

```batch
$ sudo apt install git python3-pip python3-scipy python3-numpy python3-pyaudio libatlas3-base
$ pip3 install deepspeech --upgrade
$ pip3 install halo webrtcvad --upgrade
```

<br>

### Download DeepSpeech Model

```batch
$ cd 20210208/deepSpeech
$ curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite
$ curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

<br>

## Matrix 

### Compile matrix code

```batch
$ cd 20210208/matrix
$ make
```

<br>

## Master

### Compile Master code

```batch
$ cd 20210208/master
$ make
```

<br>



## How to Execute

