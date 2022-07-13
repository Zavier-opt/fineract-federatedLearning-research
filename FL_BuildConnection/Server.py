#  === TCP server.py ===

from socket import socket, AF_INET, SOCK_STREAM
import torch

from utilities.server_connection_tools import *
from utilities.server_fL_tools import *


def server_train(duet,dataSocket):

    signal = dataSocket.recv(BUFLEN).decode()
    print(signal)
    next = input("next step...")

    FL_Model = FLModel(duet)

    remote_model_dict = FL_Model.train()

    print(remote_model_dict)



def server_predict(duet):
    pass


def clientHandler(dataSocket, addr, BUFLEN):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    receive = dataSocket.recv(BUFLEN).decode()

    duet = build_connection(dataSocket, addr, BUFLEN)

    if receive == "Train":
        server_train(duet,dataSocket)
    elif receive == "Predict":
        server_predict(duet)
    else:
        print("Client choose the invalid option!")

    dataSocket.close()


if __name__ == '__main__':
    IP = '127.0.0.1'
    PORT = 5000
    BUFLEN = 1024

    listenSocket = socket(AF_INET, SOCK_STREAM)
    listenSocket.bind((IP, PORT))

    listenSocket.listen(8)
    print(f'Sever launch successfully, wait for a Client to connect at port :{PORT}...')

    while True:
        dataSocket, addr = listenSocket.accept()
        addr = str(addr)
        print(f'Client at {addr} , connect successfully')

        th = Thread(target=clientHandler, args=(dataSocket, addr, BUFLEN))
        th.start()

    listenSocket.close()
