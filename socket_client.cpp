#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>




//写在main函数内
#include <signal.h>
#include<iostream>
#include<string>
void handle_pipe(int sig)
{
}



using namespace std;
int main(){
 
    
    struct sigaction action;
    action.sa_handler = handle_pipe;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGPIPE, &action, NULL);



    //创建套接字
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    //向服务器（特定的IP和端口）发起请求
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));  //每个字节都用0填充
    serv_addr.sin_family = AF_INET;  //使用IPv4地址
    serv_addr.sin_addr.s_addr = inet_addr("127.0.0.1");  //具体的IP地址
    serv_addr.sin_port = htons(1234);  //端口
    
   
    //读取服务器传回的数据
    char buffer[40];

  
    connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

   
  

    while(true)
    {
    
        char str[] = "ni hao hee";
    
        int x=write(sock, str, sizeof(str));

        if(x!=-1)

        {
            
            read(sock, buffer, sizeof(buffer)-1);
        
            cout<<"Message form server: "<<buffer<<endl;
            
            sleep(1);
        }
        else 
        {
            std::cout<<x<<"error"<<std::endl;
            close(sock);
            sleep(3);

 
            
            sock = socket(AF_INET, SOCK_STREAM, 0);
            connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
           

            sleep(1);

            x=write(sock, str, sizeof(str));
            std::cout<<"new x is"<<x<<std::endl;
          
            
        }




    }
    close(sock);
   
    return 0;
}
