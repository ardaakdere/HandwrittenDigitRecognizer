import pickle
import numpy as np
import os

path = os.path.dirname(__file__)
os.chdir(path)

filename = [
["training_images","train-images.idx3-ubyte"],
["test_images","t10k-images.idx3-ubyte"],
["training_labels","train-labels.idx1-ubyte"],
["test_labels","t10k-labels.idx1-ubyte"]
]

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

save_mnist()