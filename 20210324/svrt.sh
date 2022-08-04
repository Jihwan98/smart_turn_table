#!/bin/bash

THREAD()
{
    while [ 1 ]
    do
	echo -n -e "[Enter 1 to stop]"
	read num
	if [ $num -eq 1 ];then
	    echo "It takes a minute to stop whole process"
	    killall -9 python3
            break
	fi
    done
}

python3 master.py &\
THREAD getInput

echo "Finish, byebye"

