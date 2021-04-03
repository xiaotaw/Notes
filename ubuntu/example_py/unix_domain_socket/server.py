
import os
import re
import socket
import sys

server_addr = "./uds_socket"

def serverSocket():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if sock < 0:
        #print("socket error", file=sys.stderr)
        print("socket error")
    
    if os.path.exists(server_addr):
        os.unlink(server_addr)

    if sock.bind(server_addr):
        #print("socket.bind error", file=sys.stderr)
        print("socket.bind error")

    if sock.listen(3):
        #print("socket.listen error", file=sys.stderr)
        print("socket.listen error")

    print("Server start. \n")
    while(1):
        conn, client_addr = sock.accept()
        print("connected by: %s" % str(client_addr))
        conn.settimeout(5)

        try:
            request_data = conn.recv(1500)
            if request_data:
                print("    recieved: %s" % request_data)
                request_start_line = request_data.splitlines()[0]
                info = re.match(r"\w+ +(/[^ ]*) ", request_start_line.decode("utf-8")).group(1)

                response_start_line = "HTTP/1.1 200 OK\r\n"
                response_head_line = "Server: My server\r\n"
                response_body = ("I recieved: " + info).decode("utf-8")
            else:
                response_start_line = "HTTP/1.1 404 Not Fount\r\n"
                response_head_line = "Server: My server\r\n"
                response_body = "The corresponding info is not Found!"
            
            response = response_start_line + response_head_line + "\r\n" + response_body
            conn.sendall(bytes(response))
            print("    responsed: xxx")
        except Exception as e:
            print(e)
        finally:
            conn.close()

if __name__ == "__main__":
    serverSocket()
