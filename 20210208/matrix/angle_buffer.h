#include <iostream>
#include <queue>
#include <matrix_hal/everloop_image.h>

namespace hal = matrix_hal;
class angle_buffer{
    public:
    int buffer_len;
    std::queue<int> angles;
    int status;
    std::queue<hal::EverloopImage> images;
    int count;

    angle_buffer();
    ~angle_buffer();

    int get_buffer_len();
    void set_buffer_len(int len);
    std::queue<int> get_angles();
    void set_angles(std::queue<int> an);
    int get_status();
    void set_status(int stat);    
    std::queue<hal::EverloopImage> get_images();
    void set_images(std::queue<hal::EverloopImage> images);

    void show_elements(std::queue<int> an);
    //void push_buffer(int angle, hal::EverloopImage image);
    void push_buffer(int angle);
    void pop_buffer();
    
};




