import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U

#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=(3,3),padding=1) #,stride=2)
        self.conv2 = nn.Conv2d(32,40,kernel_size=(3,3),padding=1)
        self.conv3 = nn.Conv2d(40,40,kernel_size=(3,3),padding=1)
        self.conv4 = nn.Conv2d(40,32,kernel_size=(3,3),padding=1)
        self.dropout = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(64,10)
        self.fc2 = nn.Linear(64,10)

    def forward(self, x):
        # TODO use model layers to predict the two digits
#        x = x.view(-1,1,42,28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,(2,2))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,(2,2))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,(2,2))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,(2,2))
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x1 = F.leaky_relu(x1)
        x2 = F.leaky_relu(x2)
        out_first_digit = x1
        out_second_digit = x2

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()



import gzip, _pickle, numpy as np
num_classes = 10
img_rows, img_cols = 42, 28

def get_data(path_to_data_dir, use_mini_dataset):
	if use_mini_dataset:
		exten = '_mini'
	else:
		exten = ''
	f = gzip.open(path_to_data_dir + 'train_multi_digit' + exten + '.pkl.gz', 'rb')
	X_train = _pickle.load(f, encoding='latin1')
	f.close()
	X_train =  np.reshape(X_train, (len(X_train), 1, img_rows, img_cols))
	f = gzip.open(path_to_data_dir + 'test_multi_digit' + exten +'.pkl.gz', 'rb')
	X_test = _pickle.load(f, encoding='latin1')
	f.close()
	X_test =  np.reshape(X_test, (len(X_test),1, img_rows, img_cols))
	f = gzip.open(path_to_data_dir + 'train_labels' + exten +'.txt.gz', 'rb')
	y_train = np.loadtxt(f)
	f.close()
	f = gzip.open(path_to_data_dir +'test_labels' + exten + '.txt.gz', 'rb')
	y_test = np.loadtxt(f)
	f.close()
	return X_train, y_train, X_test, y_test
