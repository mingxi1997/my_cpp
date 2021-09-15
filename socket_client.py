
import socket
import time
s = socket.socket()
port = 1234
s.bind(('', port))
s.listen(5)
c, addr = s.accept()
print ("Socket Up and running with a connection from",addr)
while True:
    s=time.time()
    sendData = "OK"
    c.send(sendData.encode())
    rcvdData = c.recv(1024).decode()
    print ("received : "+rcvdData)
    print(time.time()-s)
    
    
    if(sendData == "Bye" or sendData == "bye"):
        break
c.close()
