import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch


class Data_Connection:

    def __init__(self, duet):
        self.duet = duet


    def send_data_to_duet(self, X_train, X_test, y_train, y_test):
        X_train.tag("X_Train_Owner")
        X_train_ptr = X_train.send(self.duet, pointable=True)

        y_train.tag("y_Train_Owner")
        y_train_ptr = y_train.send(self.duet, pointable=True)

        X_test.tag("X_Test_Owner")
        X_test_ptr = X_test.send(self.duet, pointable=True)

        y_test.tag("y_Test_Owner")
        y_test_ptr = y_test.send(self.duet, pointable=True)

        return (X_train_ptr, y_train_ptr,X_test_ptr,y_test_ptr )


    def preprocessing(self, dataset, data, test_size):
        """
        Preprocess dataset

        Parameters
        ----------
        data: DataFrame
            Pandas dataframe containing German dataset.
        """

        global processed_data
        global categorical
        global label_encoders

        # Reset global variables

        processed_data = None
        categorical = None
        label_encoders = {}

        if dataset == "German":
            # Drop savings account and checkings account columns as they contain a lot
            # of NaN values and may not always be available in real life scenarios
            data = data.drop(columns=['Saving accounts', 'Checking account'])

        dat_dict = data.to_dict()
        new_dat_dict = {}

        # rename columns(Make them lowercase and snakecase)
        for key, value in dat_dict.items():
            newKey = key
            if type(key) == str:
                newKey = newKey.lower().replace(' ', '_')
            # if newKey != key:
            new_dat_dict[newKey] = dat_dict[key]
        del dat_dict

        data = pd.DataFrame.from_dict(new_dat_dict)
        del new_dat_dict

        # print(data.describe())
        # print(data.describe(include='O'))

        cols = data.columns
        num_cols = data._get_numeric_data().columns
        categorical = list(set(cols) - set(num_cols))

        # Drop null rows
        data = data.dropna()

        # Encode text columns to number values
        for category in categorical:
            le = LabelEncoder()
            data[category] = le.fit_transform(data[category])
            label_encoders[category] = le

        for col in data.columns:
            if (col not in categorical):
                data[col] = (data[col].astype('float') - np.mean(data[col].astype('float'))) / np.std(
                    data[col].astype('float'))

        # print(data.describe())
        # print(data.describe(include='O'))

        processed_data = data

        # Get Training parameters
        if dataset == "German":
            target_col = data.columns[-1]
            x = data.drop(columns=target_col, axis=1)
            y = data[target_col].astype('int')
        elif dataset == "Australian":
            x = data.drop(14, axis=1)
            y = data[14].astype('int')
        elif dataset == "Japanese":
            x = data.drop(15, axis=1)
            y = data[15].astype('int')
        elif dataset == "Taiwan":
            x = data.drop('default_payment_next_month', axis=1)
            y = data['default_payment_next_month'].astype('int')
        elif dataset == "Polish":
            x = data.drop('class', axis=1)
            y = data['class'].astype('int')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        y_train = y_train[y_train.columns[0]].to_numpy()
        y_test = y_test[y_test.columns[0]].to_numpy()

        x_train = torch.FloatTensor(x_train)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_test = torch.FloatTensor(x_test)
        y_test = torch.tensor(y_test, dtype=torch.long)


        return (x_train, x_test, y_train, y_test)
