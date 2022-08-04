# ODAS

ODAS stands for Open embeddeD Audition System. This library offers sound source localization, tracking, separation and post-filtering.

[ref.] (https://medium.com/@bharathsudharsan023/odas-open-embedded-audition-system-sound-source-localization-tracking-separation-and-c29d54390137)

## Step 1: Get the ODAS demo file

After you connected to the raspberry Pi,  change the directory

```
cd odas
```

Bring the servo-market-demo file

```
git checkout servo-market-demo
```

make the make file 

```
mkdir build
cd build
cmake ..
make
```

 #### CMakeLists.txt 

```cmake
//adding odas library including variables set, 'SRC'
add_library(odas SHARED
	${SRC}
)
    target_link_libraries(odas
	${PC_FFTW3_LIBRARIES}
	${PC_ALSA_LIBRARIES}
    ${PC_LIBCONFIG_LIBRARIES}
	m
    pthread
)
add_executable(odaslive
    demo/odaslive/main.c
    demo/odaslive/configs.c
    demo/odaslive/objects.c
    demo/odaslive/parameters.c
    demo/odaslive/profiler.c
    demo/odaslive/threads.c
)
```

## Step 2: Run the demo

make sure you are still in the odas directory

```
cd ~/odas/bin
./matrix-odas &
./odaslive -vc ../config/matrix-demo/matrix_voice.cfg
```

If you run the demo file, you can see the angle showing up on the page.

<img width="285" alt="angle out" src="https://user-images.githubusercontent.com/67047653/107333780-46134100-6af9-11eb-9e97-d9e90aa6b1b4.PNG">




