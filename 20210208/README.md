# SVRT PROJECT

## 작성 및 수정한 소스파일 리스트

### Master 코드

: Matrix-odas코드와 DeepSpeech사이에서 메세지를 전달하는 코드, 후에 USB연결 감지 및 데이터베이스와의 연동부분 구현이 필요함

- master/server.cpp

```c++
// line 31-32, matrix-odas코드나 DeepSpeech코드로부터 오는 메세지를 저장하는 Queue
std::queue<std::string> m_queue1;
std::queue<std::string> m_queue2;

// line 61~65, 메세지를 주고받는 Thread를 생성
std::thread ds_t_send(senda, c_sock1, std::ref(m_queue2), std::ref(m2));
std::thread ds_t_recv(recva, c_sock1, std::ref(m_queue1), std::ref(m1));

std::thread mv_t_send(senda, c_sock2, std::ref(m_queue1), std::ref(m1));
std::thread mv_t_recv(recva, c_sock2, std::ref(m_queue2), std::ref(m2));

// line 79, matrix-odas코드나 DeepSpeech코드로부터 오는 메세지를 받는 Thread
void recva(int sock, std::queue<std::string> &m_queue, std::mutex &m)

// line 97, matrix-odas코드나 DeepSpeech코드로부터 받은 메세지를 전달하는 Thread
void senda(int sock, std::queue<std::string> &m_queue, std::mutex &m)
```



### Matrix-odas 코드

- matrix/matrix-odas.cpp

```c++
void angle_calculate_thread(hal::MatrixIOBus &bus, hal::EverloopImage &image1d, hal::Everloop &everloop)
/* line 183, odas로부터 Sound source의 좌표값을 받아 Angle을 계산하고 저장하는 Thread */

void recva(int sock, std::queue<std::string> &m_queue, std::mutex &m)
/* line 340, Master코드로부터 오는 메세지를 받는 Thread */
    
void senda(int sock, std::queue<std::string> &m_queue, hal::GPIOControl &gpio, std::mutex &m)
/* line 373, Master로부터 받은 메세지에 따라 행동하는 Thread
| 1. Message가 만약 Voice가 감지됨을 뜻하는 "Detection"일 경우:
|     Angle buffer에 저장된 각도를 카피한다.
| 2. Message가 만약 감지된 Voice가 Trigger임을 뜻하는 "Trigger"일 경우:
|     1에서 카피된 각도로 모터를 돌린 후, Master에 "Finish Rotate"라는 메세지를 보낸다.
*/
```



### Angle buffer Class

: Angle이 저장되는 객체

- matrix/angle_buffer.h
- matrix/angle_buffer.cpp

```c++
// In angle_buffer.cpp
void angle_buffer::push_buffer(int angle){
    /* 들어온 angle값이 이전에 들어온 값과 다를경우 해당 Angle을 버퍼에 저장 */
    
    if(buffer_len ==0){
	angles.push(angle);
	buffer_len++;
	count++;
	std::cout << "pushed: " << angle << std::endl;
	return;
    }
    if(angle == angles.back()){
        return;
    }
    else{
        angles.push(angle);
	buffer_len++;
	count++;
	if(buffer_len > 10){
	    angles.pop();
	    buffer_len--;
	}
	//std::cout << "In Buffer: ";
	//show_elements(angles);
	//std::cout << std::endl;
    }
}
```



### DeepSpeech

: 실시간으로 STT (Speech-To-Text) 하면서 Voice가 Detection되었거나 혹은 해당 Voice가 Trigger일 경우에 Master code로 메세지를 보내는 코드

- deepSpeech/streaming_sst.py

```python
# line 199, Voice가 Detection되면 메세지를 보냄
sendData = str("Detection")

# line 207~216, Trigger인지 아닌지 확인하고 Trigger라면 메세지를 보냄
text = stream_context.finishStream()
print("Recognized: %s" % text)
if ('f' in text) and ('r' in text) :
    tc = str("Trigger")
    clientSock.send(tc.encode())
    recvData = clientSock.recv(1024)
    print("Finish Rotate")
else:
    tc = str("0")
    print("0")
```



# 실행방법

## 1. ODAS Setup [링크](https://github.com/matrix-io/odas/tree/master/demo/matrix-demos)

```
# Add repo and key
curl https://apt.matrix.one/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.matrix.one/raspbian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/matrixlabs.list

# Update packages and install
sudo apt-get update
sudo apt-get upgrade

# Installation
sudo apt install matrixio-creator-init
sudo apt install libmatrixio-creator-hal
sudo apt install libmatrixio-creator-hal-dev
sudo reboot

sudo apt install matrixio-kernel-modules
sudo reboot

sudo apt-get install g++ git cmake
sudo apt-get install libfftw3-dev
sudo apt-get install libconfig-dev
sudo apt-get install libasound2-dev
sudo apt install libjson-c-dev

cd odas
mkdir build
cd build
cmake ..
make
```



## 2. DeepSpeech Setup [링크](https://github.com/touchgadget/DeepSpeech)

```
sudo apt install git python3-pip python3-scipy python3-numpy python3-pyaudio libatlas3-base
pip3 install deepspeech --upgrade
pip3 install halo webrtcvad --upgrade

cd deepSpeech
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```



## 3. Matrix-odas 코드 빌드

```
$ cd matrix
$ make
```



## 4. Master 코드 빌드

```
$ cd master
$ make
```



## 5. 실행파일 실행

```
$ ./svrt.sh
```

