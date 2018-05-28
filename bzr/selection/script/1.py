import sys
import numpy as np
from sklearn.model_selection import ShuffleSplit,StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from SleepingThread.ml import *

def fun(data_name,C,kernel,scale,trainer_type):

    def print_qual(qual_vec,fmt="%.2f"):
        print "   ",fmt%np.min(qual_vec),fmt%np.max(qual_vec),fmt%np.average(qual_vec)
        return 

    scaler = StandardScaler()
    data = np.loadtxt("../"+data_name)
    target = np.loadtxt("../values")  

    if trainer_type=="SVC":
        if scale:
            data = scaler.fit_transform(data)
        trainer = SVC(C=C,kernel=kernel,class_weight={1:float(np.sum(target==-1))/float(np.sum(target==1))})
    elif trainer_type=="SVM_L0":
        trainer = SVM_L0(verbose=1,eps=1.0e-3,feature_selection=True)
    else:
        raise Exception("Unknown trainer_type")


    cv = RepeatedStratifiedKFold(n_splits=5,random_state=0,n_repeats=10)
    #cv = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)
    res = cross_validate(trainer,data,target,cv=cv,scoring={'accuracy':'accuracy','wa':weighted_accuracy},return_train_score=True)
    
    print data_name,C,kernel,"%4d"%data.shape[1],scale,trainer_type,":"
    #print_qual(res["train_accuracy"]) 
    #print_qual(res["test_accuracy"])
    print_qual(res["train_wa"])
    print_qual(res["test_wa"])
    
    return

data_name = \
    [
        "cdk.txt",
        "qp4non.txt",
        "qp4nom.txt",
        "qp4wdn.txt",
        "qp4non.txt",
        "md2___.txt",
        "md2d__.txt",
        "md2db_.txt",
        "md2dbr.txt",
        "md3___.txt",
        "md3d__.txt",
        "md3db_.txt",
        "md3dbr.txt",
        "md4___.txt",
        "md4d__.txt",
        "md4db_.txt",
        "md4dbr.txt"
    ]

data_name = data_name[:1]

ExecuteGrid(fun,{'data_name':data_name,'C':[1.0],'kernel':['linear'],'scale':[True],'trainer_type':["SVM_L0"]})

