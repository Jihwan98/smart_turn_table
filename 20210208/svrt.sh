#!/bin/bash

THREAD()
{
    while [ 1 ]
    do
	echo -n "Enter 1 to stop: "
	read num
	if [ $num -eq 1 ];then
	    echo "It takes a minute to stop whole process $num"
	    killall -9 odaslive
	    killall -9 mo
	    killall -9 python3
	    killall -9 master
            break
	fi
    done
}


./master/master &\
sleep 3;
python3 deepSpeech/streaming_stt.py --nospinner -m deepSpeech/deepspeech-0.9.3-models.tflite -s deepSpeech/deepspeech-0.9.3-models.scorer &\
sleep 5;
./matrix/mo & ./odas/bin/odaslive -vc ./matrix/matrix_voice.cfg &\
sleep 2;
THREAD getInput
echo "Finish, byebye"

