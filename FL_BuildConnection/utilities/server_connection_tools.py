import re
import syft as sy
import sys
from time import sleep
import asyncio
from threading import Thread

def parseID(temp_file):
    Duet_ID = ""
    for line in open(temp_file, 'r+', encoding='utf-8'):
        #print(line)
        if ("Duet Client ID:" in line or "Duet Server ID:" in line):
            #print(line[line.index(":") + 2:])
            Duet_ID = line[line.index(":") + 2:]
            break
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    Duet_ID_result = ansi_escape.sub('', Duet_ID)
    Duet_ID_result = Duet_ID_result.strip('\n')
    return Duet_ID_result

def duetIDHandler(f, savedStdout, dataSocket):
    sleep(10)
    sys.stdout = savedStdout
    f.close()
    new_Duet_ID = parseID('temp_for_parseID_server.txt')
    print("Send the ID:" + new_Duet_ID + " to client")
    dataSocket.send(new_Duet_ID.encode())


def connect(Duet_ID, f, savedStdout, dataSocket):

    th2 = Thread(target=duetIDHandler, args=(f, savedStdout, dataSocket))
    th2.start()
    Duet = sy.join_duet(Duet_ID + "")
    return Duet

def build_connection(dataSocket, addr, BUFLEN):
    receive = dataSocket.recv(BUFLEN)
    Duet_ID = receive.decode()
    print(f'Receive Duet ID from from {addr}： {Duet_ID}')

    # step 2: Join the Duet and return to Client
    savedStdout = sys.stdout
    f = open('temp_for_parseID_server.txt', 'w+', encoding='utf-8')
    sys.stdout = f

    duet = connect(Duet_ID, f, savedStdout, dataSocket)

    print("Duet connect successfully")
    return duet


