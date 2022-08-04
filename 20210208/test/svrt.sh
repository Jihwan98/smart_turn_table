THREAD()
{
    while [ 1 ]
    do
        echo -n "Enter 1 to stop: "
	read num
	if [ $num -eq 1 ];then
	    echo "It takes a minute"
	    killall -9 python3
	    break
	fi
    done
}

python3 server_test.py &\
sleep 3;
python3 streaming_stt.py -m deepspeech-0.9.3-models.tflite -s deepspeech-0.9.3-models.scorer -d 6 -r 16000 &\

THREAD getInput
echo "Finish, ByeBye"
