import numpy as np

from dataset.mnist import load_mnist
from neuralnetwork import *
from neuralnetwork.SGD import SGD

from neuralnetwork.st_model import st_model

if __name__ == '__main__':
    dixta = {}
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    st = Stemoid(st_model(784, [50], 10))
    for x in range(20000):
        batch_mask = np.random.choice(60000, 100)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        st.learn(SGD(lr=0.01), x_batch, t_batch)
        if x % 1000 == 0:
            train_acc = st.get_accuracy(x_train, t_train)
            print(train_acc)
