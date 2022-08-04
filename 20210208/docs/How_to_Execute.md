# How to Execute

Raspberrypi id & pw for SSH

id: pi

pw: raspberry


#### 1. 실행 이전 마스터 코드 수정

수정사항이 있어서 아래와 같이 마스터코드 수정을 요청드립니다.

프로그램을 종료한 뒤에 바로 다시 프로그램을 실행시키려고 할 때 socket의 bind error가 발생하는 현상을 막기 위한 코드입니다. 

`/* 이 부분을 추가 */`와 `/* /////////// */`사이에 있는 코드를 `s_sock = socket(AF_INET, SOCK_STREAM, 0);`과 `if (s_sock == -1)`사이에 입력해주시면 됩니다.

```batch
$ cd 20210208/master
$ vim server.cpp
```

```c++
s_sock = socket(AF_INET, SOCK_STREAM, 0);

/* 이 부분을 추가 */

int enable = 1;
if (setsockopt(s_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
    error_handling("setsockopt(SO_REUSEADDR) failed");

/* /////////// */

if (s_sock == -1)
    error_handling("socket() error");

# esc -> :wq 입력 후 탈출 
```

```
$ make
$ cd ..
```



#### 2. Matrix voice와 USB 마이크의 번호 확인

마이크의 Index가 바뀌어져 있을 수 있기 때문에 아래의 명령어를 입력하여 번호를 확인합니다.

```batch
$ arecord -l
```

`card 2:`에 Matrix voice가 있고, `card 3:`에 USB 마이크가 있으면 정상입니다.

혹시 다르다면 USB마이크를 뺀 후, Reboot 후에 다시 마이크를 꽂아주시면 됩니다.



#### 3. bash파일 실행

```
$ cd 20210208/
$ ./svrt.sh
```

bash 파일 실행 후에 몇초 뒤 다음과 같은 문자가 터미널에 입력됩니다.

```
|	 Run Master 	|
[Hello, This is your friend]
|	 DeepSpeech [Ready] 	|
TensorFlow: v2.3.0-6-g23ad988
DeepSpeech: v0.9.3-0-gf2e9c85
// 다른 내용이 더 나올 수 있습니다. //
Enter 1 to stop:
```

우선 가까이에서 "Hello Friend"나 "Hey Friend"라고 불러보면서 방향에 따라 정상적으로 돌아가는지 확인합니다.

종료하시려면 터미널에 `1`을 입력하면 됩니다. `ctrl + c`로 종료할 시에는 문제가 발생할 수 있습니다.

`Sink hops: Cannot connect to server`와 같은 오류가 발생했다면 우선 프로그램을 종료한 뒤에 아래의 명령어를 입력해주시면 됩니다.

```
$ killall -9 odaslive
$ killall -9 mo
$ killall -9 python3
$ killall -9 master
```

그 다음 일정시간 기다렸다가 다시 bash파일을 실행하시면 됩니다.

