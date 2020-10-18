#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# The script is written to peform CP classification using random forest & LightGBM
# The script is revised by Jin Zhang on 2020-04-15
'''


import os, sys
#NC_Path='/projects/camde/Project/External/2003_GNN8CP/script/NC_210_v2/'
#if NC_Path not in sys.path: sys.path.append(NC_Path)

import numpy as np
import pandas as pd
from argparse import ArgumentParser

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

#from nonconformist.base import ClassifierAdapter
#from nonconformist.icp import IcpClassifier
#from nonconformist.nc import ClassifierNc, MarginErrFunc


def Table_Reader(File):
    '''function to read data separated with tab into DF'''
    print('Input file: %s' %str(File))#list input file
    DF=pd.read_csv(File,sep='\t') #read data into dataframe
    print('Input data set size: %s' %str(DF.shape))
    return DF

def DF_Writer(Name,OutDF):
    '''function to output a DF to a tab-txt file'''
    OutDF.to_csv('%s.txt'%Name,sep='\t',encoding='utf-8',index=False) #write out DF
    print('Output complete: %s.txt' %Name)
    return

def split_dataset(dataset, ratio, model=1):
    '''function to split set'''
    seed = 1234 * model # the model number starting from 1 (in our case only 1 if not more models are being built)
    np.random.seed(seed)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]

def ClassWeight(Y):
    '''funciton to calculate class weight'''
    CWeights=class_weight.compute_class_weight('balanced',np.unique(Y),Y)
    CWeights=dict(enumerate(CWeights))
    return CWeights

#take file & model input parameters
parser = ArgumentParser()
parser.add_argument('-i','--infile', type=str, help='input file')
parser.add_argument('-f','--feature', type=str, choices=['fp','rdkit'], help='feature type')
parser.add_argument('-al', '--algorithm', type=str, choices=['rf','lgbm'], help='algorithm')
#parser.add_argument('-op','--outpath', help='output path')
args = parser.parse_args()

File=args.infile
Feature=args.feature
Algor=args.algorithm
OutPath='/projects/camde/Project/External/2003_GNN8CP/result/LGBM_RF/' #args.outpath


InFile=File
#'H:/02_Project/ExternalColab/1911_RNN4Chem/Dataset/Tox21/nr-ahr.sdf.std.sdf_class.sdf.rdkit.txt'
XCol=2
YCol='class'
ChemDF=Table_Reader(InFile)
X=np.array(ChemDF.iloc[:,XCol:]).astype(np.float32)
y=np.array(ChemDF[YCol]).astype(np.int64)

Signif=None
ResultList=[]
Fold = 1
SKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
for TotalTrain, Test in SKFold.split(X, y):
    # Setup proper training, calibration and validation sets
    Train, Valid = split_dataset(TotalTrain, 0.9) #validation set created
    ProperTrain, Calib = split_dataset(Train, 0.8) #calibr set and proper training set created 
    
    # Scale X values based on feature type
    if Feature == 'rdkit':
        scalexin =  MinMaxScaler(feature_range=(0, 1)).fit(X[Train])
        X = scalexin.transform(X)
    elif Feature == 'fp':
        X = X
    else:
        sys.exit('feacture type error')
    
    #algorithm choice
    if Algor == 'lgbm':
        LGBMC_Params={'boosting_type':'gbdt', 'n_estimators':600, 'num_leaves':200,
                      'max_depth':10,'learning_rate':0.01, 'min_child_samples':55,
                      'max_bin':400,'objective':'binary','class_weight':'balanced',
                      'subsample':0.8, 'subsample_freq':5,
                      'n_jobs':12, 'random_state':12345, 'verbose':-1}
        Model = lgb.LGBMClassifier(**LGBMC_Params)
    elif Algor== 'rf':
        RFC_Params={'n_estimators':600, 'n_jobs':12, 'class_weight':'balanced', 'random_state':12345}
        Model = RandomForestClassifier(**RFC_Params)
    else:
        sys.exit('algorithm type error')
    
    Model.fit(X[ProperTrain], y[ProperTrain])
    
    datasets = [Valid, Calib, Test]
    types = ["Valid", "Calib", "Test"]
    for ds, i in zip(datasets,types):
        pred = Model.predict_proba(X[ds])
        pred = np.round(pred,4).astype('str')
        n = len(ds)
        Result = np.insert(pred, 0, values=[[i]*n, [Fold]*n, ds, y[ds]], axis=1)
        ResultList.append(Result)
    Fold+=1

ColName=['Model','Fold','ID','Class','ProbClass0','ProbClass1']
ResultDF=pd.DataFrame(np.concatenate(ResultList),columns=ColName)

os.chdir(OutPath)
OutName='%s_%s_%s_Prob'%(os.path.splitext(os.path.basename(InFile))[0],Algor,Feature)
DF_Writer(OutName,ResultDF)