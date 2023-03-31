import re
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import helper.dataset as dataUtils

import torchvision

def tokenize(sentences):
    #2) tokenize sentences (can be done during training, you can also use spacy udpipe)
    print('tokenizing sentences...')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]
    return sentences


def createVocab(sentences,n_vocab):
    #3) create vocab if not already created
    print('creating/loading vocab...')
    pth = 'vocab.txt'
    if not exists(pth):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(n_vocab) #keep the N most frequent words
        vocab = [w[0] for w in vocab]
        open(pth, 'w+').write('\n'.join(vocab))
    else:
        vocab = open(pth).read().split('\n')
    return vocab


def plot_metric(data, metric):
    """Plot accuracy graph or loss graph.
    Args:
        data (list or dict): If only single plot then this is a list, else
            for multiple plots this is a dict with keys containing.
            the plot name and values being a list of points to plot
        metric (str): Metric name which is to be plotted. Can be either
            loss or accuracy.
    """

    single_plot = True
    if type(data) == dict:
        single_plot = False
    
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot data
    if single_plot:
        plt.plot(data)
    else:
        plots = []
        for value in data.values():
            plots.append(plt.plot(value)[0])

    # Set plot title
    plt.title(f'{metric} Change')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    if not single_plot: # Set legend
        location = 'upper' if metric == 'Loss' else 'lower'
        plt.legend(
            tuple(plots), tuple(data.keys()),
            loc=f'{location} right',
            shadow=True,
            prop={'size': 15}
        )

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')
    

def createData(pth,
               train_batch_size=128,
               test_batch_size=10,
               seq_len=20,
               n_vocab=40000
               
               ):
    
    ## readin the sentences
    print("reading the sentences")
    sentences = open(pth).read().lower().split('\n')
    
    totalSentencesCount=len(sentences)
    print("totalNumberOfSentences",totalSentencesCount)
    trainingSentencesCount=int(0.8*totalSentencesCount)
    print("trainingSentencesCount",trainingSentencesCount)
    #testingSentences=0.2*totalSentences
    
    ## tokenize the sentences
    print("tokenizing the sentences")
    sentences =  tokenize(sentences)
    
    ## create the vocabulary
    vocab= createVocab(sentences,n_vocab)
    
    #4) create dataset
    print('creating train & test dataset...')
    train_dataset = dataUtils.SentencesDataset(sentences[0:trainingSentencesCount], vocab, seq_len)
    test_dataset = dataUtils.SentencesDataset(sentences[trainingSentencesCount:], vocab, seq_len)

    train_kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':train_batch_size}
    test_kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':test_batch_size}
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    return train_dataset,test_dataset,train_data_loader,test_data_loader,vocab