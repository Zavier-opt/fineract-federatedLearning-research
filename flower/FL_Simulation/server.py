from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from socket import socket, AF_INET, SOCK_STREAM
import pickle
import torch as t
from collections import OrderedDict


class Net(t.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Net, self).__init__()
        # Our network:
        # Linear1->relu->Batchnorm->Linear2->relu->Batchnorm->Dropout->Linear3->output
        # Softmax is added in the predict function
        # This applies Linear transformation to input data.
        self.fc1 = t.nn.Linear(input_channels, int(1.5 * input_channels))
        self.fc2 = t.nn.Linear(int(1.5 * input_channels), int(1.5 * input_channels))
        self.fc3 = t.nn.Linear(int(1.5 * input_channels), output_channels)

        self.relu = t.nn.ReLU()
        self.dropout = t.nn.Dropout(p=0.1)
        self.batchnorm1 = t.nn.BatchNorm1d(int(1.5 * input_channels))
        self.batchnorm2 = t.nn.BatchNorm1d(int(1.5 * input_channels))
        self.sigmoid = t.nn.Sigmoid()

    # This must be implemented
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def predict(self, x):
        output = self.forward(x)
        prediction = t.argmax(output, 1)
        return prediction


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def update_basic_model(basic_net, received_model_dict, TOTAL_CLIENT_COUNT):
    basic_net_dict = basic_net.state_dict()
    avg_updates = OrderedDict()
    for key in basic_net_dict.keys():
        avg_updates[key] = ((basic_net_dict[key]) * (TOTAL_CLIENT_COUNT - 1) + (
        received_model_dict[key])) / TOTAL_CLIENT_COUNT
    basic_net.load_state_dict(avg_updates)

    # for name, param in basic_net.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)


if __name__ == '__main__':
    IP = '127.0.0.1'
    PORT = 51000
    BUFLEN = 1024

    listenSocket = socket(AF_INET, SOCK_STREAM)
    listenSocket.bind((IP, PORT))

    listenSocket.listen(8)
    print(f'>>> Sever launch successfully, wait for a Client to connect at port :{PORT}...')

    TOTAL_CLIENT_COUNT = 0
    input_channels = 7
    output_channels = 2
    basic_net = Net(input_channels, output_channels)

    while True:
        dataSocket, addr = listenSocket.accept()
        addr = str(addr)
        print(f'>>> Client at {addr} , connect successfully')

        command = dataSocket.recv(BUFLEN).decode()
        if command == "Train":
            TOTAL_CLIENT_COUNT = TOTAL_CLIENT_COUNT + 1

            print(">>> Start Flower server")
            # Define strategy
            strategy = fl.server.strategy.FedAvg(min_fit_clients=1, min_available_clients=1,
                                                 evaluate_metrics_aggregation_fn=weighted_average)
            fl.server.start_server(
                server_address="[::]:8085",
                config={"num_rounds": 1},
                strategy=strategy,
            )

            received_pickle_model = dataSocket.recv(BUFLEN * 100)
            received_model_dict = pickle.loads(received_pickle_model)
            #print(received_model_dict)

            print(">>> Training Finish")

            update_basic_model(basic_net, received_model_dict, TOTAL_CLIENT_COUNT)
            print(">>> Model Updated")

        elif command == "Predict":
            print(">>> Send model to client")
            basic_net_dict_string = pickle.dumps(basic_net.state_dict())
            dataSocket.send(basic_net_dict_string)

        dataSocket.close()
        print(f'>>> Wait for a new Client to connect at port :{PORT}...')

    listenSocket.close()
