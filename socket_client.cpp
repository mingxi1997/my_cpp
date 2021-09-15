#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
int main(){
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
    char str[] = "ni hao";
    connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    while(true){
    
    
    
    write(sock, str, sizeof(str));

  


    read(sock, buffer, sizeof(buffer)-1);
   
    printf("Message form server: %s\n", buffer);
    
    sleep(1);
    }
    close(sock);
   
    return 0;
}
