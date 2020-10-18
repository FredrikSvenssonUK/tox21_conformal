#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# The script is written to peform prob classification using 4 layer DNN model using pytorch wrapper skorch
# The script is revised by Jin Zhang on 2020-09-05
'''


import os, sys
Ranger_Path='/projects/camde/Project/External/2003_GNN8CP/script/TorchOptimizer/'
#'C:/GDrive/Work_File/03_ScriptCenter/PythonScript/ExternalCode/Optimizer/TorchOptimizer/'

if Ranger_Path not in sys.path: sys.path.append(Ranger_Path)

import numpy as np
import pandas as pd
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.callbacks import EpochScoring,EarlyStopping
from skorch.helper import predefined_split

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

from torchtools.optim import RangerLars
from scipy.special import softmax as softmax4np


def Table_Reader(File):
    '''function to read data separated with tab into DF'''
    print('Input file: %s' %str(File))#list input file
    DF=pd.read_csv(File,sep='\t') #read data into dataframe
    print('Input data set size: %s' %str(DF.shape))
    return DF

def DF_Writer(Name,OutDF):
    'function to output a DF to a tab-txt file'
    OutDF.to_csv('%s.txt'%Name,sep='\t',encoding='utf-8',index=False) #write out DF
    print('Output complete: %s.txt' %Name)
    return

def ProbResultReformat(DF):
    '''funciton to reformat the probability prediction results'''
    DF.rename(columns={'ID':'id','Class':'class',
                       'ProbClass0':'score_low','ProbClass1':'score_high',
                       'Fold':'model','Model':'set'}, inplace=True)
    DF['set'].replace({'Valid':'val','Calib':'cal','Test':'test'},inplace=True)
    ColOrderList=['id','class','score_low','score_high','model','set']
    DF=DF[ColOrderList]
    return DF

def split_dataset(dataset, ratio, model=1):
    seed = 1234 * model # the model number starting from 1 (in our case only 1 if not more models are being built)
    np.random.seed(seed)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]

def MinusBACC(net, X=None, y=None):
    y_true = y
    y_pred = net.predict(X)
    return -balanced_accuracy_score(y_true, y_pred)


class DNN4Layer(nn.Module):
    def __init__(self,InD,OutD):
        super(DNN4Layer,self).__init__()
        self.fc1 = nn.Linear(InD, 1000)
        self.fc2 = nn.Linear(1000, 4000)
        self.fc3 = nn.Linear(4000, 2000)
        self.fc4 = nn.Linear(2000, OutD)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(4000)
        self.bn3 = nn.BatchNorm1d(2000)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


#take file & model input parameters
parser = ArgumentParser()
parser.add_argument('-i','--infile', help='input file')
parser.add_argument('-f','--feature', type=str, choices=['fp','rdkit'], help='feature type')
args = parser.parse_args()

File=args.infile
Feature=args.feature
#'rdkit'#
OutPath='/projects/camde/Project/External/2003_GNN8CP/result/DNN_V3/prob_RLars_4Layer/'

InFile=File
#'H:/02_Project/ExternalColab/1911_RNN4Chem/Dataset/Tox21/nr-ahr.sdf.std.sdf_class.sdf.rdkit.txt'

XCol=2
YCol='class'
ChemDF=Table_Reader(InFile)
print('Input feature type: %s' %Feature)
print('Model type: DNN_4Layer')

X=np.array(ChemDF.iloc[:,XCol:]).astype(np.float32)
y=np.array(ChemDF[YCol]).astype(np.int64)

Fold = 1
FeatureDim=X.shape[1]
ResultList=[]
SKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
for TotalTrain, Test in SKFold.split(X, y):
    print('Running DNN_4Layer fold %s'%Fold)
    # Setup proper training, calibration and validation sets
    Train, Valid = split_dataset(TotalTrain, 0.9) #validation set created
    ProperTrain, Calib = split_dataset(Train, 0.8) #calibr set and proper training set created 
    
    # Scale X values
    if Feature == 'rdkit':
        scalexin =  MinMaxScaler(feature_range=(0, 1)).fit(X[Train])
        X = scalexin.transform(X)
    elif Feature == 'fp':
        X = X
    else:
        sys.exit('feacture type error')
    
    # Convert validation to skorch dataset
    valid_ds = Dataset(X[Valid], y[Valid])
    
    # Calculate number of training examples for each class (for weights)
    train_0 = len([x for x in y[ProperTrain] if x==0])
    train_1 = len([x for x in y[ProperTrain] if x==1])
    
    # Setup for class weights
    class_weights = 1 / torch.FloatTensor([train_0, train_1])
    
    # Define the skorch classifier
    MinusBA = EpochScoring(MinusBACC, name='-BA',on_train=False,
                      use_caching=False, lower_is_better=True)
    EarlyStop = EarlyStopping(patience=3, threshold=0.01,
                              threshold_mode='rel', lower_is_better=True)
    
    Model = NeuralNetClassifier(DNN4Layer,
                                module__InD=FeatureDim,
                                module__OutD=2,
                                batch_size=512,
                                max_epochs=100,
                                train_split=predefined_split(valid_ds), # Use predefined validation set
                                optimizer=RangerLars,
                                optimizer__lr=0.001,
                                optimizer__weight_decay=0.01,
                                criterion=nn.CrossEntropyLoss,
                                criterion__weight=class_weights,
                                callbacks=[MinusBA,EarlyStop])
    
    Model.fit(X[ProperTrain], y[ProperTrain])
    
    datasets = [Valid, Calib, Test]
    types = ["Valid", "Calib", "Test"]
    for ds, i in zip(datasets,types):
        pred = Model.predict_proba(X[ds])
        pred = softmax4np(pred,axis=1)
        pred = np.round(pred,4).astype('str')
        n = len(ds)
        Result = np.insert(pred, 0, values=[[i]*n, [Fold]*n, ds, y[ds]], axis=1)
        ResultList.append(Result)
    Fold+=1

ColName=['Model','Fold','ID','Class','ProbClass0','ProbClass1']
ResultDF=pd.DataFrame(np.concatenate(ResultList),columns=ColName)

ReformatResultDF=ProbResultReformat(ResultDF)

os.chdir(OutPath)
OutName='%s_DNN4-V3_%s_Prob_RLars'%(os.path.splitext(os.path.basename(InFile))[0],Feature)
DF_Writer(OutName,ReformatResultDF)