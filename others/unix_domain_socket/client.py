
import socket
import sys

server_addr = "./uds_socket"

def clientSocket():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if sock < 0:
        print("socket error")

    try:
        sock.connect(server_addr)
    except socket.error, msg:
        print("exception: %s" % msg)
        sys.exit(1)

    message = "GET /abcdes HTTP/1.1\r\n\
    Host: 47.104.234.135:6100\r\n\
    Connection: keep-alive\r\n\
    Cache-Control: max-age=0\r\n\
    Upgrade-Insecure-Requests: 1\r\n\
    User-Agent: Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36\r\n\
    Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3\r\n\
    Accept-Encoding: gzip, deflate\r\n\
    Accept-Language: zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7\r\n\r\n"

    sock.sendall(message)
    print("send: %s" % message)

    data = sock.recv(100)
    print("recieved: %s" % data)
    
    sock.close()

if __name__ == "__main__":
    clientSocket()
