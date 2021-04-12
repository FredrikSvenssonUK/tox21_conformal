#!/usr/bin/env python

# imports
import os,sys
from bisect import bisect_left
from bisect import bisect_right

import numpy as np
from numpy.core.numeric import asanyarray
import sklearn
import pandas as pd


def search(alist, item):
    'Locate the leftmost value exactly equal to item'
    i = bisect_right(alist, item)
    return i


############## main program ###################
try:
    sys.argv[1]
except IndexError:
    print ("You need to specify and input file with combined calibration and test set scores")
    sys.exit(1)
try:
    sys.argv[2]
except IndexError:
    print ("You need to specify maximum number of models to use (<0 = all models)")
    sys.exit(1)


aa = sys.argv[1] + '_p-values_pred.csv'
f = open(aa,'w')
f.write('Title\tPred\tp-value low class\tp-value high class\tpred_class_0.2\tclass\tloop\n')


df = pd.read_csv(sys.argv[1], sep='\t', header = 0, index_col = None)

dfhigh = df.loc[df['class'] > 0]
dflow = df.loc[df['class'] <= 0]

maxmodel = df['model'].max()
if int(sys.argv[2]) > 0:
    maxmodel = int(sys.argv[2])

print (maxmodel)
for model in range(0, maxmodel+1):
    
    print ('model', model)
    calibrhigh = df.loc[df['class'] > 0]
    calibrhigh = calibrhigh.loc[calibrhigh['model'] == model]
    print(calibrhigh)
    calibrhigh = calibrhigh.loc[calibrhigh['set']  == 'cal']
    calibrhigh = calibrhigh['score_high'] 
    calibrsort1 =  np.sort(calibrhigh, axis=None)
    calibrsort1 = calibrsort1.astype(float)


    calibrlow = df.loc[df['class'] <= 0]
    calibrlow = calibrlow.loc[calibrlow['model'] == model]
    #print(calibrlow)
    calibrlow = calibrlow.loc[calibrlow['set']  == 'cal']
    calibrlow = calibrlow['score_low'] 
    calibrsort0 =  np.sort(calibrlow, axis=None)
    calibrsort0 = calibrsort0.astype(float)
                    

    lencl0 = len(calibrsort0)
    lencl1 = len(calibrsort1)
    #print (lencl0, lencl1)

    test = df.loc[df['model'] == model]
    test = test.loc[test['set']  == 'test']
    testid = test['id']

    testclass = test['class']
    testhigh = asanyarray(test['score_high'])
    testlow = asanyarray(test['score_low'])

    ll = len(testclass)
    for yy in range(0, ll):
        pos0 = search(calibrsort0, testlow[yy])
        pos1 = search(calibrsort1, testhigh[yy])
        lencl00 = lencl0 + 1
        pos0 = float(pos0)/float(lencl00)
        lencl11 = lencl1 + 1
        pos1 = float(pos1)/float(lencl11)

        testid = asanyarray(test['id'])
        testclass = asanyarray(testclass)

        write0 = str(testid[yy]) + '\tNA\t' + str(pos0) + '\t' + str(pos1)  + '\tNA\t' + str(testclass[yy]) + '\t' + str(model) + '\n'
        f.write(write0)

f.close()
print ('Finished')


