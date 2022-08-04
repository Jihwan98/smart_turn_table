#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <queue>
#include <string>
#include <thread>
#include <iostream>
#include <mutex>
#include <chrono>

#define MSG_SIZE 1024

using std::thread;

void error_handling(char *message);
void recva(int sock, std::queue<std::string> &m_queue, std::mutex &m);
void senda(int sock, std::queue<std::string> &m_queue, std::mutex &m);

char recv_buf[MSG_SIZE];
char send_buf[MSG_SIZE];

int main()
{

    int s_sock;
    int c_sock1, c_sock2;
    std::mutex m1, m2;
    std::queue<std::string> m_queue1;
    std::queue<std::string> m_queue2;

    struct sockaddr_in sa, ca1, ca2;

    socklen_t ca1_size, ca2_size;

    s_sock = socket(AF_INET, SOCK_STREAM, 0);

    int enable = 1;
    if (setsockopt(s_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
        error_handling("setsockopt(SO_REUSEADDR) failed");

    if (s_sock == -1)
	error_handling("socket() error");

    memset(&sa, 0, sizeof(sa));

    sa.sin_family = AF_INET;
    sa.sin_port = htons(40000);
    sa.sin_addr.s_addr = INADDR_ANY;

    if (bind(s_sock, (struct sockaddr*)&sa, sizeof(sa)) == -1) 
        error_handling("bind() error");

    if (listen(s_sock, 5) == -1)
        error_handling("listen() error");

    ca1_size = sizeof(ca1);
    ca2_size = sizeof(ca2);
    std::cout << "|\t Run Master \t|" << std::endl;
    std::cout << "[Hello, This is your friend]" << std::endl;
    c_sock1 = accept(s_sock, (struct sockaddr*)&ca1, &ca1_size); 
    c_sock2 = accept(s_sock, (struct sockaddr*)&ca2, &ca2_size); 

    std::thread ds_t_send(senda, c_sock1, std::ref(m_queue2), std::ref(m2));
    std::thread ds_t_recv(recva, c_sock1, std::ref(m_queue1), std::ref(m1));

    std::thread mv_t_send(senda, c_sock2, std::ref(m_queue1), std::ref(m1));
    std::thread mv_t_recv(recva, c_sock2, std::ref(m_queue2), std::ref(m2));

    ds_t_send.join();
    ds_t_recv.join();
    mv_t_send.join();
    mv_t_recv.join();

    close(c_sock1);
    close(c_sock2);
    close(s_sock);

    return 0;
}

void recva(int sock, std::queue<std::string> &m_queue, std::mutex &m){
    int ms_len;
    while(1){
        ms_len = read(sock, recv_buf, MSG_SIZE-1);
        std::string str = recv_buf;

	m.lock();
        m_queue.push(str); 
	m.unlock();

        memset(recv_buf, '\0', sizeof(recv_buf));
        sleep(0.1);  
    }    
}

void senda(int sock, std::queue<std::string> &m_queue, std::mutex &m){
    while(1){
        m.lock();
	if (m_queue.empty()){
	    m.unlock();
	    std::this_thread::sleep_for(std::chrono::milliseconds(10));
	    continue;   
	}
	else {
	    m.unlock();
            memset(send_buf, '\0', sizeof(send_buf));
            strcpy(send_buf, m_queue.front().c_str());
            m_queue.pop();
            write(sock, send_buf, strlen(send_buf));      
	    sleep(0.1);
        }
    }
}



void error_handling(char *message){
        fputs(message, stderr);
        fputc('\n', stderr);
        exit(1);
}
