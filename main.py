import csv
import numpy as np
import random
import math

# features' name (except AQI)
features = ['AQI','SO2','CO','O3','PM10','PM2.5','NO2','NOx','NO']

'''
    list to numpy and can reshape matrix
'''
def list2numpy(data,shape):
    # turn list into nparray
    nparray = np.array(data)
    # reshape 1-Dim to 2-Dim
    nparray = np.reshape(nparray,shape)
    return nparray


'''
  load data from datasets
'''
def load_data(filename):
    data_set = []
    train_set = []
    #test_set = []
    global features
    # open csv file, change the argument of encode
    with open(filename,mode='r',encoding='utf-8',newline='') as csvfile:
        
        rows = csv.DictReader(csvfile)
        print('type-row:',type(rows))
        # append each of elements of rows
        for row in rows:
            for fea in features:
                data_set.append(row[fea])
        # list to numpy and reshape matrix
        npdata = list2numpy(data_set,(-1,len(features)))
        # numpy to list for using the function of list
        datalist = npdata.tolist()
        print('data_set:',len(datalist))

        # random datasets and split datasets into trainset and testset
        for column in datalist:
            train_set.append(column)
        print('train_set:',len(train_set))
        return train_set

'''
    set label data 
'''
def label(datas):
    # classification of labels
    labels = []
    for row in datas:
        if int(row[0])<=50:
            labels.append(1)
        elif int(row[0])>50 and int(row[0])<=100:
            labels.append(2)
        elif int(row[0])>100 and int(row[0])<=150:
            labels.append(3)
        elif int(row[0])>150 and int(row[0])<=200:
            labels.append(4)
        elif int(row[0])>200 and int(row[0])<=300:
            labels.append(5)
        else:
            labels.append(6)
    return labels

'''
    set label AQI data
'''
def labelAQI(datas):
    # classification of AQI
    labels = []
    for row in datas:
        labels.append(row[0])
    return labels

'''
    remove label element from trainset and testset
'''
def removeEle(datas):
    #remove AQI(label) from trainset and testset
    Sdatas = []
    for row in datas:
        row.pop(0)
        Sdatas.append(row)
    return Sdatas

'''
    calculate number of label(1,2,3,4,5,6)
    1-6 is sequence from best to worst
'''
def eachlen_label(datas):
    # calculate the number of each of labels
    nplab = np.zeros(6,int)
    lab = nplab.tolist()
    for row in datas:
        if int(row) is 1:
            lab[0]+=1
        elif int(row) is 2:
            lab[1]+=1
        elif int(row) is 3:
            lab[2]+=1
        elif int(row) is 4:
            lab[3]+=1
        elif int(row) is 5:
            lab[4]+=1
        else:
            lab[5]+=1
    return lab
    
'''
    calculate prior probability, Prob = best~worst/all data
'''
def cal_prior(datas):
    # calculate prob for (each of labels)/(sum of labels)
    lab_prior = []
    for x in range(0,6):
        prior = eachlen_label(datas)[x]/len(datas)
        if prior <= 0:
            prior = 0.00001
        lab_prior.append(prior)
    print('lab_prior:',lab_prior)
    return lab_prior

'''
    split features
'''
def average_split(datas):
    # calculate an average to split features
    global features
    big = [0]*(len(features)-1)
    small = [1000]*(len(features)-1)
    average = [0]*(len(features)-1)
    for x in range(0,(len(features)-1)):
        for row in datas:
            if float(row[x]) > big[x]:
                big[x] = float(row[x])
            if float(row[x]) < small[x]:
                small[x] = float(row[x])
        average[x] = (big[x]+small[x])/2
    print('big:',big)
    print('small:',small)
    print('average:',average)
    return average

'''
    calculate likelihood
'''
def cal_likelihood(datas,labels,average):
    # the number of features
    mark = []
    mark.append([0]*((len(features)-1)*2))
    mark.append([0]*((len(features)-1)*2))
    mark.append([0]*((len(features)-1)*2))
    mark.append([0]*((len(features)-1)*2))
    mark.append([0]*((len(features)-1)*2))
    mark.append([0]*((len(features)-1)*2))
    for x in range(0,len(labels)):
        if labels[x] is 1:
            for y in range(0,len(datas[x])):
                if float(datas[x][y]) < average[y]:
                    mark[0][y*2]+=1
                else:
                    mark[0][y*2+1]+=1
        elif labels[x] is 2:
            for y in range(0,len(datas[x])):
                if float(datas[x][y]) < average[y]:
                    mark[1][y*2]+=1
                else:
                    mark[1][y*2+1]+=1
        elif labels[x] is 3:
            for y in range(0,len(datas[x])):
                if float(datas[x][y]) < average[y]:
                    mark[2][y*2]+=1
                else:
                    mark[2][y*2+1]+=1
        elif labels[x] is 4:
            for y in range(0,len(datas[x])):
                if float(datas[x][y]) < average[y]:
                    mark[3][y*2]+=1
                else:
                    mark[3][y*2+1]+=1
        elif labels[x] is 5:
            for y in range(0,len(datas[x])):
                if float(datas[x][y]) < average[y]:
                    mark[3][y*2]+=1
                else:
                    mark[3][y*2+1]+=1
        elif labels[x] is 6:
            for y in range(0,len(datas[x])):
                if float(datas[x][y]) < average[y]:
                    mark[3][y*2]+=1
                else:
                    mark[3][y*2+1]+=1                
    print('mark:',mark)

    # calculate likelihood for each of marks
    mark_prob = [[0 for i in range(0,len(mark[0]))] for j in range(0,len(mark))]
    
    for x in range(0,len(mark)):
        for y in range(0,len(mark[0])):
            if (mark[x][0]+mark[x][1]) > 0:
                mark_prob[x][y] = mark[x][y]/(mark[x][0]+mark[x][1])
            else:
                mark_prob[x][y] = 0.0
    print('mark_prob:',mark_prob)
    return mark_prob
    

def cal_MAP(like,prior,size=3):
    MAP_res = [0 for x in range(0,6)]
    # calculate log of prior
    for x in range(0,6):
        prior[x] = math.log(prior[x])
    print('log_prior:',prior)

    # calculate MAP
    for x in range(0,size):
        feature = random.randint(x*2,x*2+1)
        for y in range(0,6):
            if prior[y]!=0 and like[y][feature]!=0:
                prior[y] = prior[y]+math.log(like[y][feature])
            else:
                prior[y] = prior[y]+like[y][feature]
    MAP_res = prior
    print('MAP_res:',MAP_res)
    return MAP_res

def classify(MAP):
    nearZero = -1000
    index = -1
    for x in range(0,len(MAP)):
        if MAP[x] > nearZero:
            nearZero = MAP[x]
            index = x

    print('result:',end='')
    if index is 0:
        print('0~50 best AIQ')
    elif index is 1:
        print('51~100 good AIQ')
    elif index is 2:
        print('101~150 so-so AIQ')
    elif index is 3:
        print('151~200 bad AIQ')
    elif index is 4:
        print('201~300 worse AIQ')
    else:
        print('301~ worst AIQ')
    
            
filename = 'AQI_utf8.csv'
Xtrain= load_data(filename)
Ytrain= label(Xtrain)
YAQI_train = labelAQI(Xtrain)

Xtrain = removeEle(Xtrain)

prior = cal_prior(Ytrain)
avg =average_split(Xtrain)

prob_like = cal_likelihood(Xtrain,Ytrain,avg)
MAP = cal_MAP(prob_like,prior)
classify(MAP)




