import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--filename',type=str,default='eth_mnist_gpu.npz')
args = parser.parse_args()

fp = open(args.filename,'r')
npzfile = np.load(fp)
data = npzfile['data']
print(np.shape(data))
print(np.mean(data[:,6]))
