import syft as sy
import torch

class SyNet(sy.Module):
    def __init__(self, input_channels, output_channels, torch_ref):
        super(SyNet, self).__init__(torch_ref=torch_ref)
        # Our network:
        # Linear1->relu->Batchnorm->Linear2->relu->Batchnorm->Dropout->Linear3->output
        # Softmax is added in the predict function
        # This applies Linear transformation to input data.
        self.fc1 = self.torch_ref.nn.Linear(input_channels, int(1.5 * input_channels))
        self.fc2 = self.torch_ref.nn.Linear(int(1.5 * input_channels), int(1.5 * input_channels))
        self.fc3 = self.torch_ref.nn.Linear(int(1.5 * input_channels), output_channels)

        self.relu = self.torch_ref.nn.ReLU()
        self.dropout = self.torch_ref.nn.Dropout(p=0.1)
        self.batchnorm1 = self.torch_ref.nn.BatchNorm1d(int(1.5 * input_channels))
        self.batchnorm2 = self.torch_ref.nn.BatchNorm1d(int(1.5 * input_channels))
        self.sigmoid = self.torch_ref.nn.Sigmoid()

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
        prediction = self.torch_ref.argmax(output, 1)
        return prediction




class FLModel:

    def __init__(self, duet):
        self.duet = duet
        self.X_Train_Owner_ptr = duet.store[0]
        self.y_Train_Owner_ptr = duet.store[1]
        self.X_Test_Owner_ptr = duet.store[2]
        self.y_Test_Owner_ptr = duet.store[3]

        self.get_datashape()

        self.model = SyNet(self.input_channels, self.output_channels, torch)

    def get_datashape(self):
        X_Train_Owner1_shape = self.X_Train_Owner_ptr.shape.get(
            reason="To evaluate training progress",
            request_block=True,
            timeout_secs=5,
        )
        self.input_channels = X_Train_Owner1_shape[1]
        self.output_channels = 2

    def classifier_train(self, epochs, model, optimizer, X_ptr, y_ptr, criterion, torch_ref):
        losses = []
        for i in range(epochs):
            # Precit the output for Given input
            y_pred_ptr = model.forward(X_ptr)
            # Compute Cross entropy loss
            loss = criterion(y_pred_ptr, y_ptr)
            loss_item = loss.item()
            # Request to get the loss value
            loss_value = loss_item.get(
                reason="To evaluate training progress",
                request_block=True,
                timeout_secs=5,
            )
            # Add loss to the list
            losses.append(loss_value)
            # Print loss
            if i % 50 == 0:
                print("Epoch:", i, " Loss:", loss_value)

            # Clear the previous gradients
            optimizer.zero_grad()
            # Compute gradients
            loss.backward()
            # Adjust weights
            optimizer.step()
        return losses


    def train(self):
        remote_model = self.model.send(self.duet)
        params = remote_model.parameters()
        remote_torch = self.duet.torch
        criterion = remote_torch.nn.CrossEntropyLoss()
        optimizer = remote_torch.optim.Adam(params=params, lr=0.001)
        epochs = 1000

        losses = self.classifier_train(epochs, remote_model, optimizer, self.X_Train_Owner_ptr, self._Train_Owner_ptr, criterion,
                                  remote_torch)

        remote_model_updates = remote_model.get(
            request_block=True
        ).state_dict()

        return remote_model_updates


