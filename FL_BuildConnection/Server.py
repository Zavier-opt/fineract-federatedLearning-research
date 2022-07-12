#  === TCP server.py ===

from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread
import syft as sy
import sys
from time import sleep
import asyncio

from utilities.connection_tools import *

def duetIDHandler(f,savedStdout,dataSocket):
    sleep(10)
    sys.stdout = savedStdout
    f.close()
    new_Duet_ID = parseID('temp_for_parseID_server.txt')
    print("Send the ID:" + new_Duet_ID+" to client")
    dataSocket.send(new_Duet_ID.encode())


def connect(Duet_ID ,f,savedStdout, dataSocket):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    th2 = Thread(target=duetIDHandler, args=(f,savedStdout,dataSocket))
    th2.start()
    Duet = sy.join_duet(Duet_ID+"")
    return Duet

# Use a new thread to handle this connection with a client;
def clientHandler(dataSocket,addr,BUFLEN):

    # step1: receive Duet ID from Client;
    #       - if return == null : Client close this connection
    # step2: Join the Duet and generate a new ID return to Client;


    # step 1: Get the Duet ID from client
    receive = dataSocket.recv(BUFLEN)
    if not receive:
        print(f'Client: {addr} close this connection')
        return
    Duet_ID = receive.decode()
    print(f'Receive Duet ID from from {addr}： {Duet_ID}')

    # step 2: Join the Duet and return to Client
    savedStdout = sys.stdout
    f = open('temp_for_parseID_server.txt', 'w+',encoding='utf-8')
    sys.stdout = f # redirect the output steam to the file for parsing the ID

    duet = connect(Duet_ID, f, savedStdout,dataSocket)


    print("Duet connect successfully")


    dataSocket.close()


if __name__ == '__main__':
   IP = '127.0.0.1'
   PORT = 5000
   BUFLEN = 1024

   # create a listen socket for server
   listenSocket = socket(AF_INET, SOCK_STREAM)
   listenSocket.bind((IP, PORT))

   listenSocket.listen(8)
   print(f'Sever launch successfully, wait for a Client to connect at port :{PORT}...')

   while True:
       dataSocket, addr = listenSocket.accept()  # Establish connection with client.
       addr = str(addr)
       print(f'Client at {addr} , connect successfully')

       # create new a new thread to handle this connection
       th = Thread(target=clientHandler, args=(dataSocket, addr, BUFLEN))
       th.start()

   listenSocket.close()










