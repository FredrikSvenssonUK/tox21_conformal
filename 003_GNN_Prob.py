#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# The script is written to peform CP classification using GNN model based on SMILES
# The script is revised by Jin Zhang on 2020-04-15
'''

import os, sys
Ranger_Path='/projects/camde/Project/External/2003_GNN8CP/script/TorchOptimizer/'
if Ranger_Path not in sys.path: sys.path.append(Ranger_Path)

import numpy as np
import pandas as pd
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

import dgl
from dgl.data.chem.utils import mol_to_complete_graph
from dgl.model_zoo.chem import GCNClassifier, GATClassifier
from dgl.data.chem import CanonicalAtomFeaturizer

from rdkit.Chem import PandasTools
from torchtools.optim import RangerLars


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

def collate(sample):
    graphs, labels = map(list,zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

def Mol2Graph(mol,atom_featurizer):
    '''function to convert mol to mol graph for GNN'''
    GraphList=[mol_to_complete_graph(m, atom_featurizer=atom_featurizer) for m in mol]
    return GraphList

def LoadData(g_set,y):
    zip_data = list(zip(g_set, y))
    load_data = DataLoader(zip_data, batch_size=256, shuffle=False,
                              collate_fn=collate) #, drop_last=True
    return load_data

def GNN_Train(model,train_loader,loss_fn,optimizer,rnd,eval_loader=None):
    model.train()
    #losses, accuracies = [],[]
    patience=2
    min_val_acc= 0
    epochs_no_improve = 0
    for epoch in range(1,int(rnd)+1):
        train_loss = 0
        train_acc = 0
        valid_acc = 0
        for i, (bg, labels) in enumerate(train_loader):
            atom_feats = bg.ndata.pop('h')
            pred = model(bg, atom_feats)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            pred_cls = pred.argmax(-1).detach().numpy()
            true_label = labels.numpy()
            train_acc += balanced_accuracy_score(true_label, pred_cls)
        valid_acc = GNN_Pred(model,eval_loader)
        train_acc /= (i + 1)
        train_loss /= (i + 1)
        if epoch % 5 == 0:
            print('epoch %.3d | loss %.4f | Valid_BA %.4f |'%(epoch,train_loss,valid_acc))
        if valid_acc >= min_val_acc: 
             #torch.save(model) # Save the model
             epochs_no_improve = 0
             min_val_acc = valid_acc
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience: # Check early stop condition
            print('Early stop at epoch %.3d'%epoch)
            print('epoch %.3d | loss %.4f | Valid_BA %.4f |'%(epoch,train_loss,valid_acc))
            break
        #accuracies.append(train_acc)
        #losses.append(train_loss)
    return model

def GNN_Pred(model,test_load,result=False):
    model.eval()
    PredList = []
    m = nn.Softmax(dim=1)
    with torch.no_grad():
        test_acc = 0
        for i, (bg, labels) in enumerate(test_load):
            atom_feats = bg.ndata.pop('h')
            pred = model(bg, atom_feats)
            pred_cls = pred.argmax(-1).detach().numpy()
            true_label = labels.numpy()
            test_acc += balanced_accuracy_score(true_label, pred_cls)
            PredList.append(m(pred).numpy())
            #PredList.append(np.column_stack((true_label,m(pred).numpy())))
        test_acc /= (i + 1)
    if result:
        return float('%.3f'%test_acc), np.concatenate(PredList)
    else:
        return float('%.3f'%test_acc)


#take file & model input parameters
parser = ArgumentParser()
parser.add_argument('-i','--infile', help='input file')
parser.add_argument('-al', '--algorithm', type=str, choices=['GCN','GAT'], help='algorithm')
parser.add_argument('-t','--target', type=str,choices=['NR-AR', 'NR-AR-LBD', 'NR-AhR',
                                                       'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                                                       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                                                       'SR-HSE', 'SR-MMP', 'SR-p53'],
                    help='endpoint target')

args = parser.parse_args()
File=args.infile
Algor=args.algorithm
Target=args.target
OutPath='/projects/camde/Project/External/2003_GNN8CP/result/GNN/'

print(Target)
print(Algor)

LigFile = File
LigDF = PandasTools.LoadSDF(LigFile)
YCol = Target
XCol = 'ROMol'
IDCol = 'ID'
ClassNo = 2

LigDF.dropna(subset=[YCol],inplace=True)
y=np.array(LigDF[YCol]).astype('int64')
X=np.array(LigDF[XCol])
ID=np.array(LigDF[IDCol])


atom_featurizer = CanonicalAtomFeaturizer()
n_feats = atom_featurizer.feat_size('h') # check feature size
#print(n_feats)

ResultList=[]
Fold = 1
SKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
for TotalTrain, Test in SKFold.split(X, y):
    # Setup proper training, calibration and validation sets
    Train, Valid = split_dataset(TotalTrain, 0.9) #validation set created
    ProperTrain, Calib = split_dataset(Train, 0.8) #calibr set and proper training set created
    
    # Calculate class weights
    train_0 = len([i for i in y[ProperTrain] if i==0])
    train_1 = len([i for i in y[ProperTrain] if i==1])
    class_weights = 1 / torch.FloatTensor([train_0, train_1])
    
    # Define GNN classifier
    if Algor == 'GCN':
        InModel = GCNClassifier(in_feats=n_feats,
                    gcn_hidden_feats=[256, 256],
                    classifier_hidden_feats=256,
                    n_tasks=ClassNo,
                    dropout=0.1)
    elif Algor == 'GAT':
        InModel = GATClassifier(in_feats=n_feats,
                    gat_hidden_feats=[64, 64],
                    num_heads=[8, 8],
                    classifier_hidden_feats=128,
                    n_tasks=ClassNo,
                    dropout=0.1)
    else:
        sys.exit('algorithm type error')
        #print(InModel)
    
    TrainLoader = LoadData(Mol2Graph(X[ProperTrain],atom_featurizer), y[ProperTrain])
    ValidLoader = LoadData(Mol2Graph(X[Valid],atom_featurizer), y[Valid])
    
    Loss_FN = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = RangerLars(InModel.parameters(),lr=0.01,weight_decay=0.005)
    
    GNN_Model = GNN_Train(InModel,TrainLoader,Loss_FN,optimizer,rnd=100,
                          eval_loader=ValidLoader)
    #__, PredProb=GNN_Pred(GNN_Model,ValidLoader,result=True)
    
    datasets = [Valid, Calib, Test]
    types = ["Valid", "Calib", "Test"]
    for ds, i in zip(datasets,types):
        PredLoader = LoadData(Mol2Graph(X[ds],atom_featurizer), y[ds])
        __, PredProb=GNN_Pred(GNN_Model,PredLoader,result=True)
        pred = np.round(PredProb,4).astype('str')
        n = len(ds)
        Result = np.insert(pred, 0, values=[[i]*n, [Fold]*n, ds, y[ds]], axis=1)
        ResultList.append(Result)
    Fold+=1
ColName=['Model','Fold','ID','Class','ProbClass0','ProbClass1']
ResultDF=pd.DataFrame(np.concatenate(ResultList),columns=ColName)


os.chdir(OutPath)
OutName='%s_%s_%s'%(os.path.splitext(os.path.basename(File))[0],Target,Algor)
DF_Writer(OutName,ResultDF)