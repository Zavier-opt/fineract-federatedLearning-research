#  === TCP client.py ===
import os
from threading import Thread
from time import sleep
from socket import socket, AF_INET, SOCK_STREAM
import sys
import syft as sy

from utilities.connection_tools import *

def serverHandler(dataSocket,f,savedStdout,BUFLEN):

    # con't Step 1: Launch Duet and send the sever ID to server
    sleep(5) # wait the main thread to generate the output
    sys.stdout = savedStdout
    f.close()

    Duet_ID = parseID('temp_for_parseID_client.txt')
    dataSocket.send(Duet_ID.encode())


    # Step 2: receive the new Duet ID from server
    new_Duet_ID = dataSocket.recv(BUFLEN)
    print("This is the received Duet ID:")
    print(new_Duet_ID.decode())     # Print the received Duet ID from user to input
    print("Input this received Duet ID for connection")



if __name__ == '__main__':
    IP = '127.0.0.1'
    SERVER_PORT = 5000
    BUFLEN = 1024

    # Build and connect a socket
    dataSocket = socket(AF_INET, SOCK_STREAM)
    try:
        dataSocket.connect((IP, SERVER_PORT))
    except:
        print("Unsuccessfully connect to the server, check the address and port")
        sys.exit(1)


    # step 1: Launch Duet and send the sever ID to server
    savedStdout = sys.stdout
    f = open('temp_for_parseID_client.txt', 'w+',encoding='utf-8')
    sys.stdout = f  # redirect the output steam to the file for parsing the ID

    # Create a new thread for parse and send Duet ID to server
    # This new thread will also receive the new Duet ID from server

    th = Thread(target=serverHandler, args=(dataSocket, f, savedStdout,BUFLEN))
    th.start()

    duet = sy.launch_duet() # wait from the server to return the new Duet ID, user need to input this new server ID


    print('Duet connect successfully')

    dataSocket.close()