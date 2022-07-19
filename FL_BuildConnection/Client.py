#  === TCP client.py ===

from socket import socket, AF_INET, SOCK_STREAM

from utilities.client_connection_tools import *
from utilities.client_fl_tools import *


def client_train(duet,dataSocket):
    client_data = pd.read_csv("./data/german_data_1.csv")

    duet.requests.add_handler(
        action="accept",
        print_local=True,  # print the result in your notebook
    )
    data_connect = Data_Connection(duet)

    X_train, X_test, y_train, y_test = data_connect.preprocessing("German", client_data, test_size=0.1)

    X_train_ptr, y_train_ptr, X_test_ptr, y_test_ptr = data_connect.send_data_to_duet(X_train, X_test, y_train, y_test)

    dataSocket.send("Data have been sent to Duet".encode())

    print(duet.store.pandas)

    finish_signal = dataSocket.recv(BUFLEN).decode()
    print(">>> " + finish_signal)


def client_predict(duet):
    pass


if __name__ == '__main__':
    IP = '127.0.0.1'
    SERVER_PORT = 5000
    BUFLEN = 1024

    dataSocket = socket(AF_INET, SOCK_STREAM)
    try:
        dataSocket.connect((IP, SERVER_PORT))
    except:
        print("Unsuccessfully connect to the server, check the address and port")
        sys.exit(1)

    option = input("Train or Predict ?: ")
    dataSocket.send(option.encode())

    duet = build_connection(dataSocket, BUFLEN)

    print("For Test:")
    print(duet)

    if option == "Train":
        client_train(duet,dataSocket)
    elif option == "Predict":
        client_predict(duet)
    else:
        print("Wrong input option")



    dataSocket.close()
