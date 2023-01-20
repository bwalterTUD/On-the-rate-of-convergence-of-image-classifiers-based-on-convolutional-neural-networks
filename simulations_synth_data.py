# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:24:39 2021

@author: bwalter
"""
import time
import math
from math import floor
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import  Input, Dense, GlobalMaxPool2D, Conv2D, ZeroPadding2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
import random
from PIL import Image, ImageDraw
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry import Point
import statistics
import itertools
from sklearn.model_selection import GridSearchCV,ShuffleSplit
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from itertools import product

# Choose task, training sample size and resolution of synthetic images
task = 1
n_train = 200
res = 32

# Starting time of computation
#t_time = time.time()
now = datetime.now()
start = now.strftime("Date: "+"%D"+" Time:"+"%H:%M:%S")
filename = now.strftime('results_'+'%d.%m_%H.%M'+'_synth_data_task='+str(task)+'_n='+str(n_train)+'.txt')

# Define parameter grids:

# Grid for CNN
grid_2 = {'level': [2,3,4], 'channels': [2,4,8], 'layers': [1,2,3], 'paths': [1,2], 'rnet': [5,10]}

# Grid for standard FNN
grid_0={'neurons':[10,20,50,100,200],'layers':[1,2,3,4,5,6,7,8]}

# Grid for nearest neighbor estimate
kn=[1,2,3]+list(range(4,int(n_train*0.8),4))
grid_r_0={'n_neighbors':kn}

# Grid for random forest
grid_r_1={'n_estimators':[10,50,100,200],'max_leaf_nodes':[8,16,32]}

# Grid for SVM poly
grid_r_2={'kernel':['poly'],'degree':[2,3,4],'gamma':[1e-2,1e-1,1,1e+1],'C':[1e-2,1e-1,1e+0,1e+1]}

# Grid for SVM rbf
grid_r_3={'kernel':['rbf'],'gamma':[1e-2,1e-1,1,1e+1],'C':[1e-2,1e-1,1e+0,1e+1]}

# Define models for non-deep approaches
model0=neighbors.KNeighborsClassifier()
model1=RandomForestClassifier()
model2 = svm.SVC()
model3 = svm.SVC()
models=[model0,model1,model2,model3]
param_grids=[grid_r_0,grid_r_1,grid_r_2,grid_r_3]

# Generator for splitting of the sample
gen = ShuffleSplit(n_splits=1,test_size=0.2,train_size=0.8,random_state=0)

# Scaler for preprocessing data for SVMs
scaler = StandardScaler()

# Define callbacks for training for deep approaches
early_stop =tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.000001, patience=500)
early_stop_2 =tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.000001, patience=2000)
early_stop_1 =tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0005, patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.001)
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
# Defne initializer for trainable weights
initializer = 'glorot_uniform'
initializer2 = initializers.Constant(value=0.5)
initializer3 = initializers.RandomNormal(mean=0.0, stddev=0.1)
tf.keras.backend.clear_session()

# Create layers for CNN architecture

class BoundLayer(Layer):
    def __init__(self, d1, d2, **kwargs):
        self.d1 = d1
        self.d2 = d2
        super(BoundLayer, self).__init__(**kwargs)

    def call(self, x):
        return x[:, :self.d1, :self.d2, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (self.d1,) + (self.d2,) + (input_shape[3],)

class LinOut(Layer):
    def __init__(self, **kwargs):
        super(LinOut, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = [1, 1, input_shape[-1], 1]
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer=initializer3)
        super(LinOut, self).build(input_shape)

    def call(self, x):
        return K.conv2d(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


def convpart(x, input_shape, Par):
    (level, channels, layers, M, paths) = Par
    d1 = math.ceil((input_shape[0] - 2 ** level + 1))
    d2 = math.ceil((input_shape[1] - 2 ** level + 1))
    for r in range(level):
        for j in range(layers):
            x = ZeroPadding2D(padding=((0, M[r] - 1), (0, M[r] - 1)))(x)
            x = Conv2D(channels, (M[r], M[r]), activation='relu', kernel_initializer=initializer, bias_initializer='zeros')(x)
    x = BoundLayer(d1, d2)(x)
    x = LinOut()(x)
    x = GlobalMaxPool2D()(x)
    return x

# Create overall CNN architecture

def CNN_arch(lr=0.01, input_shape=(32, 32, 1), Par=(2, 1, 2, [3, 3], 2, 4)):
    (level, channels, layers, M, paths, rnet) = Par
    input_shape = input_shape
    input = Input(shape=input_shape)
    Cpaths = []
    if paths == 1:
        output = convpart(input, input_shape, Par[:5])
    elif paths > 1:
        Cpaths = []
        for i in range(paths):
            Cpaths.append(convpart(input, input_shape, Par[:5]))
        output = concatenate(Cpaths)
        #output = tf.keras.layers.Maximum()(Cpaths)
    for i in range(layers):
        output = Dense(rnet, activation='relu', kernel_initializer=initializer, bias_initializer='zeros')(output)
    output = Dense(1, kernel_initializer=initializer, bias_initializer=initializer2)(output)
    model = Model(inputs=input, outputs=output)
    optimizer = Adam(learning_rate=lr, amsgrad=True)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=Adam(learning_rate=lr, amsgrad=True), loss='MSE', metrics=['accuracy', lr_metric])
    return model

# Create parameter grid for CNN architecture

def CNN_grid(grid):
    all_comb = []
    level = grid['level']
    channels = grid['channels']
    layers = grid['layers']
    paths = grid['paths']
    rnet = grid['rnet']
    for l in level:
        filter_sizes = [(2 ** np.arange(1,l+1) ).tolist()]
        #filter_sizes = [(2 * np.floor(np.divide(2 ** np.arange(l), 4)) + 3).astype(int).tolist()]
        all_comb += [a for a in product([l], channels, layers, filter_sizes, paths, rnet)]
    return all_comb

# Create standard FNN architecture

def NN_create_model(input_shape=(res*res,),neurons=10,layers=2):
    input_shape=input_shape
    input = Input(shape=input_shape)
    x=Dense(neurons,activation='relu')(input)
    for i in range(layers-1):
        x=Dense(neurons,activation='relu')(x)
    x=Dense(1,kernel_initializer=initializer3, bias_initializer=initializer2)(x)
    model=Model(inputs=input, outputs=x)
    # Compile model
    model.compile(optimizer=Adam(lr=0.0001), loss='MSE',metrics=['accuracy'])
    return model

def grid_search_NN(train_images, train_labels, grid, input_shape=(32, 32, 1) ):
    dim = input_shape[0]*input_shape[0]
    error=[]
    par=[]
    n=train_images.shape[0]
    train_images=train_images.reshape(n,dim)
    neurons=grid['neurons']
    layers=grid['layers']
    # verbosity during training
    ver = 0
    # number of epochs while grid search
    epochs1=1000
    # number of training with final parameters
    epochs2=1000
    for (neurons,layers) in itertools.product(neurons,layers):
        model=NN_create_model(neurons=neurons,layers=layers)
        n_train=floor(n*0.8)
        hist=model.fit(train_images[0:n_train],train_labels[0:n_train],epochs=epochs1,verbose=ver,callbacks=[early_stop_1])
        s=1-model.evaluate(train_images[n_train:],train_labels[n_train:],verbose=0)[1]
        error.append(s)
        par.append((neurons,layers))
        f=open(filename,'a')
        f.write('\n------------------------------------------------------------------------\n')
        f.write(' neurons: '+str(neurons)+' layers: '+str(layers))
        f.write('\n Training loss: '+str(hist.history['loss'][-1]))
        f.write('\n------------------------------------------------------------------------\n\n')
        f.close()
        del model
        K.clear_session()
    error_min=min(error)
    ind=error.index(error_min)
    best_par={'neurons':par[ind][0],'layers':par[ind][1]}
    (neurons,layers)=par[ind]
    model=NN_create_model(neurons=neurons,layers=layers)
    hist=model.fit(train_images,train_labels,epochs=epochs2,verbose= ver,callbacks=[early_stop])
    f=open(filename,'a')
    f.write('Training loss '+str(hist.history['loss'][-1])+'\n')
    f.close()
    return model,error_min,best_par

def grid_search_CNN(train_images, train_labels, grid, input_shape=(32, 32, 1), func_class=0):
    error = []
    par = []
    n=train_images.shape[0]
    # verbosity during training
    ver=0
    # number of epochs per initialization
    epochs1 = 20
    # number of epochs while grid search
    epochs2 = 1000
    # number of training with final parameters
    epochs4 = 1000
    # number of initializations
    init = 25
    weights_list = []
    for Par in grid:
        model = CNN_arch(lr=0.01, input_shape=input_shape, Par=Par)
        weights=model.count_params()
        # avoid overparameterization:
        if weights>n:
            continue
        n_train=floor(n*0.8)
        losses_init=[]
        weights_in=[]
        t_time=time.time()
        for i in range(init):
            model = CNN_arch(lr=0.01, input_shape=input_shape, Par=Par)
            hist=model.fit(train_images[0:n_train],train_labels[0:n_train],epochs=epochs1,verbose=ver,batch_size=32,callbacks=[early_stop_1])
            current_loss = hist.history['loss'][-1]
            losses_init+=[current_loss]
            weights_in+=[model.get_weights()]
            del model
            K.clear_session()
        model = CNN_arch(lr=0.01, input_shape=input_shape, Par=Par)
        model.set_weights(weights_in[losses_init.index(min(losses_init))])
        hist=model.fit(train_images[0:n_train],train_labels[0:n_train],epochs=epochs2,verbose=ver,callbacks=[reduce_lr, early_stop])
        s = 1-model.evaluate(train_images[n_train:],train_labels[n_train:],verbose=ver)[1]
        f = open(filename,'a')
        f.write('\n'+'-'*140+'\n')
        f.write('Parameters: \n')
        f.write('l='+str(Par[0])+', k='+str(Par[1])+', layers='+str(Par[2])+', M='+str(Par[3])+' and paths='+str(Par[4])+'.'+str(Par))
        f.write('\n Int_loss='+str(min(losses_init))+' number of weights: '+str(weights)+' Training loss: '+str(hist.history['loss'][-1])+' error: '+str(s)+"--- %s seconds ---" % round(time.time() - t_time))
        f.write('\n'+'-'*140+'\n\n')
        f.close()
        weights_list.append(model.get_weights())
        error.append(s)
        par.append(Par)
    error_min=min(error)
    ind=error.index(error_min)
    Par=par[ind]
    weights = weights_list[ind]
    model = CNN_arch(lr=0.005, input_shape=input_shape, Par=Par)
    model.set_weights(weights)
    hist=model.fit(train_images, train_labels, epochs=epochs4, verbose=ver, batch_size=32, callbacks=[reduce_lr, early_stop_2])
    f=open(filename,'a')
    f.write('Training loss '+str(hist.history['loss'][-1])+'\n')
    f.close()
    return model, error_min, Par

# Create synthetic image data

# Center of image area
m = (res + 1) / 2 - 1

# Create object "circle"
def create_circle():
    circle = Point(m, m).buffer(1)
    q = math.sqrt(circle.area)
    r = math.sqrt(random.uniform(60, 80))
    obj = shapely.affinity.scale(circle, r / q, r / q)
    return obj

# Create object "polygon"
def create_p(k):
    angles = np.arange(k) * math.pi * 2 * (1 / k)
    p = []
    for angle in angles:
        p = p + [[math.sin(angle), math.cos(angle)]]
    p = np.array(p)
    P = Polygon(p)
    P = shapely.affinity.translate(P, xoff=m - P.centroid.x, yoff=m - P.centroid.y)
    q = math.sqrt(P.area)
    scale = math.sqrt(random.uniform(60, 80))
    P = shapely.affinity.scale(P, yfact=scale / q, xfact=scale / q, origin=(m, m))
    angle = random.uniform(0, 360 / k)
    P = affinity.rotate(P, angle)
    return P

# Randomly position single object
def trans(obj):
    dist = 0
    B = obj.bounds
    x_trans = random.uniform(-B[0] + dist, res - 1 - B[2] - dist)
    y_trans = random.uniform(-B[1] + dist, res - 1 - B[3] - dist)
    obj2 = shapely.affinity.translate(obj, xoff=x_trans, yoff=y_trans)
    return obj2

# Randomly position the last object of a list
def trans_xy(list):
    obj_new = trans(list[-1])
    q = []
    for obj in list[:-1]:
        q += [obj_new.intersection(obj).area / obj.area]
    q = max(q)
    while q > 0.01:
        q = []
        obj_new = trans(list[-1])
        for obj in list[:-1]:
            q += [obj_new.intersection(obj).area / obj.area]
        q = max(q)
    return obj_new

# Create object of task 1
def create_obj_1():
    i = random.uniform(0, 1)
    b = 1 - 0.5 ** (1 / 3)
    if i > b:
        j = random.randint(0, 1)
        if j == 1:
            return create_p(3), 0
        else:
            return create_p(4), 0
    else:
        return create_circle(), 1

# Create object of task 2
def create_obj_2():
    i = random.randint(0, 1)
    if i == 1:
        return create_circle(), i
    else:
        return create_p(3), i

# Create random image and label from task 1
def task_1():
    im = Image.new('L', (res, res), color=0)
    draw = ImageDraw.Draw(im)
    obj1, i = create_obj_1()
    obj2, j = create_obj_1()
    obj3, k = create_obj_1()
    obj1 = trans(obj1)
    obj2 = trans_xy([obj1, obj2])
    obj3 = trans_xy([obj1, obj2, obj3])
    colors = [85, 170, 255]
    color1 = random.choice(colors)
    colors.remove(color1)
    color2 = random.choice(colors)
    colors.remove(color2)
    color3 = random.choice(colors)
    draw.polygon(obj1.exterior.coords, fill=color1)
    draw.polygon(obj2.exterior.coords, fill=color2)
    draw.polygon(obj3.exterior.coords, fill=color3)
    label = int(i + j + k > 0)
    return im, label

# Create random image and label from task 2
def task_2():
    im = Image.new('L', (res, res), color=0)
    draw = ImageDraw.Draw(im)
    obj1, i = create_obj_2()
    obj2, j = create_obj_2()
    obj1 = trans(obj1)
    obj2 = trans_xy([obj1, obj2])
    colors = [128, 255]
    color1 = random.choice(colors)
    colors.remove(color1)
    color2 = random.choice(colors)
    draw.polygon(obj1.exterior.coords, fill=color1)
    draw.polygon(obj2.exterior.coords, fill=color2)
    label = int(i == j)
    return im, label

# List of tasks
tasks = [task_1, task_2]

# create image data sets with sample size N
def create_data(N):
    images = np.empty([N, res, res, 1])
    labels = np.empty([N])
    for i in range(N):
        im, label = tasks[task-1]()
        labels[i] = label
        images[i, :, :, :] = np.array(im).reshape((res, res, 1))
    images = images.astype('float32') / 255
    return images, labels

# Define lists of scores of the six approaches
scores = [[], [], [], [], [], []]

for j in range(25):
    train_images, train_labels = create_data(n_train)
    test_images, test_labels = create_data(10000)
    images_shape = train_images.shape[1:]
    # CNN
    grid = CNN_grid(grid_2)
    start_time = time.time()
    f = open(filename, 'a')
    f.write('\n')
    f.write(str(j + 1) + ".run: Testing class of convolutional neural networks F%f :" % int(1) + '\n')
    f.close()
    model, error_min, best_par = grid_search_CNN(train_images, train_labels, grid, input_shape=images_shape,func_class=0)
    s = 1 - model.evaluate(test_images, test_labels, verbose=0)[1]
    scores[0].append(s)
    f = open(filename, 'a')
    f.write('\n')
    f.write("Best: %s using %s with emp. miscl risk: %s" % (error_min, best_par, s) + '\n')
    f.write('Number of parameters: ' + str(model.count_params()) + '\n')
    f.write("--- %s seconds ---" % round(time.time() - start_time) + '\n')
    f.close()
    # standard FFN
    start_time = time.time()
    f = open(filename, 'a')
    f.write('\n')
    f.write(str(j + 1) + ".run: Testing standard FFN :" + '\n')
    f.close()
    model, error_min, best_par = grid_search_NN(train_images,train_labels, grid_0, input_shape=images_shape)
    test_images=test_images.reshape(test_images.shape[0],images_shape[0]*images_shape[0])
    s = 1 - model.evaluate(test_images, test_labels, verbose=0)[1]
    scores[1].append(s)
    f = open(filename, 'a')
    f.write('\n')
    f.write("Best: %s using %s with emp. miscl risk: %s" % (error_min, best_par, s) + '\n')
    f.write('Number of parameters: ' + str(model.count_params()) + '\n')
    f.write("--- %s seconds ---" % round(time.time() - start_time) + '\n')
    f.close()
    train_images = train_images.reshape(train_images.shape[0], images_shape[0]*images_shape[0])
    # non-deep approaches
    for i in range(4):
        if i > 1:
            train_images = scaler.fit_transform(train_images)
            test_images = scaler.transform(test_images)
        start_time = time.time()
        grid = GridSearchCV(estimator=models[i], param_grid=param_grids[i], n_jobs=1, cv=gen)
        grid_result = grid.fit(train_images, train_labels)
        f = open(filename, 'a')
        f.write("Best: %f using %s" % (1 - grid_result.best_score_, grid_result.best_params_) + '\n')
        s = 1 - grid_result.score(test_images, test_labels)
        scores[2+i].append(s)
        f.write("--- %s seconds ---" % round(time.time() - start_time) + '\n')
        f.close()
    f = open(filename, 'a')
    f.write('\n' + 'Results: ' + str(scores) + '\n')
    f.write('\n' + 'Median approach 1 (CNN): ' + str(statistics.median(scores[0])) + ' IQR:' + str(np.subtract(*np.percentile(scores[0], [75, 25]))) + '\n')
    f.write('\n' + 'Median approach 2 (stand. FNN): ' + str(statistics.median(scores[1])) + ' IQR:' + str(np.subtract(*np.percentile(scores[1], [75, 25]))) + '\n')
    f.write('\n' + 'Median approach 3 (nearest N.): ' + str(statistics.median(scores[2])) + ' IQR:' + str(np.subtract(*np.percentile(scores[2], [75, 25]))) + '\n')
    f.write('\n' + 'Median approach 4 (rand forest): ' + str(statistics.median(scores[3])) + ' IQR:' + str(np.subtract(*np.percentile(scores[3], [75, 25]))) + '\n')
    f.write('\n' + 'Median approach 5 (SVM poly): ' + str(statistics.median(scores[4])) + ' IQR:' + str(np.subtract(*np.percentile(scores[4], [75, 25]))) + '\n')
    f.write('\n' + 'Median approach 6 (SVM rbf): ' + str(statistics.median(scores[5])) + ' IQR:' + str(np.subtract(*np.percentile(scores[5], [75, 25]))) + '\n')
    f.close()
f=open(filename, 'a')
f.write("--------------------------------End-------------------------------\n\n\n")
f.close()
print("Results:")
print('\n' + 'Median approach 1: ' + str(statistics.median(scores[0])) + ' IQR:' + str(np.subtract(*np.percentile(scores[0], [75, 25]))) + '\n')
print('\n' + 'Median approach 2: ' + str(statistics.median(scores[1])) + ' IQR:' + str(np.subtract(*np.percentile(scores[1], [75, 25]))) + '\n')
print('\n' + 'Median approach 3: ' + str(statistics.median(scores[2])) + ' IQR:' + str(np.subtract(*np.percentile(scores[2], [75, 25]))) + '\n')
print('\n' + 'Median approach 4: ' + str(statistics.median(scores[3])) + ' IQR:' + str(np.subtract(*np.percentile(scores[3], [75, 25]))) + '\n')
print('\n' + 'Median approach 5: ' + str(statistics.median(scores[4])) + ' IQR:' + str(np.subtract(*np.percentile(scores[4], [75, 25]))) + '\n')
print('\n' + 'Median approach 6: ' + str(statistics.median(scores[5])) + ' IQR:' + str(np.subtract(*np.percentile(scores[4], [75, 25]))) + '\n')
