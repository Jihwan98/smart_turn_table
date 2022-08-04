#include "angle_buffer.h"

namespace hal = matrix_hal;

angle_buffer::angle_buffer(){
    buffer_len = 0;
    status = 0;
}

angle_buffer::~angle_buffer(){}

int angle_buffer::get_buffer_len(){ return buffer_len; }
void angle_buffer::set_buffer_len(int len){ this->buffer_len = len; }

std::queue<int> angle_buffer::get_angles(){ return angles; }
void angle_buffer::set_angles(std::queue<int> an){ this->angles = angles; }

int angle_buffer::get_status(){ return status; }
void angle_buffer::set_status(int stat){ this->status = stat; }

void angle_buffer::show_elements(std::queue<int> an){
    std::queue<int> q = an;
    while(!q.empty()){
        int tmp = q.front();
	std::cout << tmp << ", ";
	q.pop();
    }
}

std::queue<hal::EverloopImage> angle_buffer::get_images(){ return images; }
void angle_buffer::set_images(std::queue<hal::EverloopImage> images){ this->images = images; }



void angle_buffer::push_buffer(int angle){
    if(buffer_len ==0){
	angles.push(angle);
	buffer_len++;
	count++;
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

    }
}


void angle_buffer::pop_buffer(){
    if(buffer_len != 0){
        int angle = angles.front();
        angles.pop();
        buffer_len++;
    }
    else{
        std::cout << "Buffer is empty!" << std::endl;
    }
}

