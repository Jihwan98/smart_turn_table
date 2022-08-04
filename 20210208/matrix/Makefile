CXX=g++ -std=c++11
LIBS= -Ijson-c
LDFLAGS= -pthread -I/usr/include/json-c
CFLAGS= -L/usr/share/doc -ljson-c -L/usr/include -lmatrix_creator_hal
OBJS=matrix-odas.o angle_buffer.o

TARGET=mo

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(CFLAGS)

matrix_odas.o: angle_buffer.h matrix-odas.cpp
angle_buffer.o: angle_buffer.h angle_buffer.cpp
