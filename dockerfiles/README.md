# Docker Installation (사진첨부 필요)

[참고자료](https://www.boolsee.pe.kr/installation-and-running-of-docker-in-raspberry-pi-buster/)

## Step 1: Update and Upgrade

```output
sudo apt-get update && sudo apt-get upgrade
```

This ensures you install the latest version of the software.



## Step 2: Download the Script and Install Docker

```
$ curl -fsSL https://test.docker.com -o test-docker.sh
```

Download "test-docker.sh".



## Step 3: Add a Non-root User to the Docker Group

```
$ sudo usermod -aG docker pi
$ sudo reboot
```

위의 Command 입력 후, reboot하면 sudo없이 Docker 명령을 사용할 수 있다.



## Step 4 : Install Docker with "test-docker.sh" 

```
$ bash test-docker.sh
$ ps auwx|grep docker
$ docker ps
$ docker run hello-world
```

Docker 설치 후, 제대로 설치되었는 지 확인



## Step 5: Build Container with Dockerfile

```
$ docker build -t alpha -f Dockerfile.alpha1.0
```

Dockerfile이 있는 폴더에서 위의 명령어 입력 후, Build가 완료되면

```
$ docker run -it --device=/dev/matrixio_regmap --device /dev/snd:/dev/snd alpha /bin/bash
```

위의 명령어를 입력하면 Docker Container에 접속한다.

```
$ ./matrix-odas & ./odaslive -vc ../config/matrix-demo/matrix_voice.cfg
```

위의 명령어를 입력하면 ODAS demo가 돌아간다.





