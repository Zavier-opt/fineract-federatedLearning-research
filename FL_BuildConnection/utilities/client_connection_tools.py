import re
from threading import Thread
from time import sleep
import sys
import syft as sy


def parseID(temp_file):
    Duet_ID = ""
    for line in open(temp_file, 'r+', encoding='utf-8'):
        if "Duet Client ID:" in line or "Duet Server ID:" in line:
            Duet_ID = line[line.index(":") + 2:]
            break
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    Duet_ID_result = ansi_escape.sub('', Duet_ID)
    Duet_ID_result = Duet_ID_result.strip('\n')
    return Duet_ID_result


def serverHandler(dataSocket, f, savedStdout, BUFLEN):
    sleep(5)
    sys.stdout = savedStdout
    f.close()

    Duet_ID = parseID('temp_for_parseID_client.txt')
    dataSocket.send(Duet_ID.encode())

    new_Duet_ID = dataSocket.recv(BUFLEN)
    print("This is the received Duet ID:")
    print(new_Duet_ID.decode())
    print("Input this received Duet ID for connection")


def build_connection(dataSocket, BUFLEN):
    savedStdout = sys.stdout
    f = open('temp_for_parseID_client.txt', 'w+', encoding='utf-8')
    sys.stdout = f

    th = Thread(target=serverHandler, args=(dataSocket, f, savedStdout, BUFLEN))
    th.start()

    duet = sy.launch_duet()

    print('Duet connect successfully')

    return duet
