import argparse
import os
import re
import numpy as np
parser = argparse.ArgumentParser(description='Extract data from log file.')
parser.add_argument('--filename', type=str, default='eth_mnist_gpu')
args = parser.parse_args()
fp = open(args.filename,'r')

line = fp.readline()
line = fp.readline()
cnt = 0
dataset = np.array([])
while line:
    t1 = re.findall(r'-?\d+\.?\d*e?-?\d*?',line)
    if len(t1)<1:
        print(cnt,line)
    line = fp.readline()
    t2 = re.findall(r'-?\d+\.?\d*e?-?\d*?',line)
    if len(t2)>0:
        t1.append(t2[0])
    if len(t1)==7:
        dataset = np.append(dataset,t1)
    else:
        break
    line = fp.readline()
    cnt += 1
fp.close

dataset = np.float_(dataset)
dataset = np.reshape(dataset,[-1,7])
np.savez(args.filename+'.npz',data=dataset)
