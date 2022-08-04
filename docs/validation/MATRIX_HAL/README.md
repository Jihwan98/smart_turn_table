# MATRIX Hardware Abstraction Layer #

MATRIX Hardware Abstraction Layer (HAL) is an open source library for directly interfacing with the MATRIX device. MATRIX HAL consists of driver files written in C++ which enable the user to write low level programs in C++.

## Installing MATRIX HAL From Package ##

```
curl https://apt.matrix.one/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.matrix.one/raspbian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/matrixlabs.list

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install matrixio-creator-init libmatrixio-creator-hal libmatrixio-creator-hal-dev

sudo reboot
```

## Installing MATRIX HAL From Source ##

```
curl https://apt.matrix.one/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.matrix.one/raspbian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/matrixlabs.list

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install cmake g++ git libfftw3-dev wiringpi libgflags-dev matrixio-creator-init

cd ~/
git clone https://github.com/matrix-io/matrix-creator-hal.git
cd matrix-creator-hal
mkdir build
cd build
cmake ..
make -j4 && sudo make install

sudo reboot
```
## ODAS: servo-matrix-demo branch ##
```
cd ~/
git clone https://github.com/matrix-io/odas.git -b servo-matrix-demo
cd odas
mkdir build
cd build
cmake ..
make

cd ~/odas/bin
./matrix-odas &
./odaslive -vc ../config/matrix-demo/matrix_voice.cfg
```

## TEST EXAMPLE ##

* Everloop: LED interface.

* Humidity: Humidity and temperature measurement.

* IMU: Inertial Measurement Unit.

* Pressure: Pressure, altitude and temperature measurement.

* UV: Ultraviolet light sensor.

* GPIO: General Purpose Input/Output.

* Microphone: Microphone Array. [example](https://github.com/matrix-io/matrix-hal-examples)

<< requirement >>
```g++ -o YOUR_OUTPUT_FILE YOUR_CPP_FILE -std=c++11 -lmatrix_creator_hal -lgflags```

<< download package >>
```
sudo apt-get install cmake g++ git
cd ~/
git clone https://github.com/matrix-io/matrix-hal-examples.git
cd matrix-hal-examples
mkdir build
cd build
cmake ..
make -j4
```

  * [Microphone Array Record to File](https://github.com/matrix-io/matrix-hal-examples/blob/master/microphone_array/mic_record_file.cpp)
   To convert the .raw files outputted by this example to playable .wav files run these commands, replacing 16000 with selected sampling rate.
   
   
  ```
  sudo apt-get install sox alsa-utils
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_0.raw channel_0.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_1.raw channel_1.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_2.raw channel_2.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_3.raw channel_3.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_4.raw channel_4.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_5.raw channel_5.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_6.raw channel_6.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_7.raw channel_7.wav
  sox -r 16000 -c 1 -e signed -c 1 -e signed -b 16 mic_16000_s16le_channel_8.raw channel_8.wav
  ```


  * [Microphone Array Record to Pipe](https://github.com/matrix-io/matrix-hal-examples/blob/master/microphone_array/mic_record_pipe.cpp) - mySQL, mariaDB, PostgreSQL에 저장 가능
  
* NFC: Near field communication.
