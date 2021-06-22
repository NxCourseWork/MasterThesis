#!/usr/bin/env python 3

# -*- coding: utf-8 -*-
"""

@Author: Narmada Ambigapathy

@Email: narmada.ambika@gmail.com

@Github:

"""

#########################################################
#---------------------START-OF-CODE---------------------#
#########################################################


# Importing required packages
import numpy as np
from collections import Counter
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import alpha_dropout, softmax
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold,StratifiedKFold
from torch.autograd import Variable


#########################################################
#------------------EXAMPLE MAIN FUNCTION----------------#
#########################################################


def main():
    """
    Main function loading some example Dataset and ...
    """
    # Set the parent directory, as well as the directory the datasets are storen inside
    parent_dir = os.getcwd()
    data_dir = os.path.join(parent_dir, "Data")

    # Definig the path of some example dataset
    data_path = os.path.join(data_dir, "Irene.xls")

    # Loading the example dataset
    Irene_collected_Data = pd.read_excel(data_path)

    n_collected_Data = len(Irene_collected_Data)

    [all_chosen_colors, all_given_color_options] = collected_data_to_vector(Irene_collected_Data)

    counts = histogram_of_chosen_colors(Irene_collected_Data)

    all_chosen_colors_newly_encoded = new_encoding(
        all_given_color_options,
        all_chosen_colors
    )[0]

    learning_rate = 0.0001

    dataset_x = torch.tensor(all_given_color_options).float()
    dataset_y = torch.tensor(
        all_chosen_colors_newly_encoded,
        dtype=torch.float64
    )
    k_folds=5

    K_fold_Cross_Validation_model_evaluation(dataset_x, dataset_y, k_folds,learning_rate)


#########################################################
#---------------END OF EXAMPLE MAIN FUNCTION------------#
#########################################################



#########################################################
#---------------OUR PROPOSED MODEL----------------------#
#########################################################


def efficient_dot_product_attention_with_fuzzy_logic(input):
    """The Parameters of attention mechanism are Query,Q and key-value pairs(K, V).
    As a first step,the model performs the dot-Product of key and value is
    performed which is defined  as attention score. By applying fuzzy
    similarity logic for the attention score, the model exhibits the most
    relevant feature of the target. The weighted sum of elementwise
    multiplication is calculated with this output and the value matrix.

    Args:
        input (torch.tensor): Output of the last layer of our network

    Returns:
        torch.tensor: Output of our Attention Network
    """
    # w_key, w_query and w_value are generic for our problem and were
    # determined heuristically by trying to get good results
    w_key = torch.tensor([[ 0.8438,  0.0458,  1.2261],
        [ 1.4970,  0.7737,  1.9437],
        [ 1.1215, -0.8844,  0.3796]])
    w_query = torch.tensor([[ 0.6880],
        [ 0.0036],
        [-0.2792]])
    w_value = torch.tensor([[-2.1909, -0.3406,  0.7994],
        [-0.5766,  0.9782, -0.3785],
        [ 2.5631,  0.2146, -0.6863]])

    s = torch.empty(3, 3)
    keys = input @ w_key
    querys = input @ w_query
    values = input @ w_value
    attn_scores = keys.T @ values
    for i in range(len(attn_scores)):
        if  i == len(attn_scores) -1:
            _scores = fuzzy_similarity( attn_scores[i], attn_scores[0])
        if  i < len(attn_scores) -1:
            _scores = fuzzy_similarity( attn_scores[i], attn_scores[i+1])
        s[i] = torch.Tensor(_scores)
    #attn_scores_softmax = softmax(s, dim=-1)
    attn_scores_softmax = softmax(s, dim=-1)

    weighted_values = querys[:,None] * attn_scores_softmax.T[:,:,None]
    outputs = weighted_values.sum(dim=0)
    #print(outputs)
    return outputs



#########################################################
#---------------THE NEURAL NETWORK----------------------#
#########################################################



class Net(nn.Module):
    """Neural network function (Net) that consists of three fully
    connected layer with Relu activation function to introduce the
    non-linearity and Stochastic gradient function  for back propagation.
    the output of the model is fed into proposed dot-product attention layer.
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return efficient_dot_product_attention_with_fuzzy_logic(x)



#########################################################
#---------------vALIDATION OF OUR MODEL-----------------#
#########################################################



# K-fold Cross Validation model evaluation -->x
def K_fold_Cross_Validation_model_evaluation(dataset_x, dataset_y, k_folds, learning_rate):
    """K-fold Cross Validation model evaluation of our proposed model

    Args:
        dataset_x (List): The given color options
        dataset_y (List): The chosen colors
        k_folds (Integer): Number of dataset foldings
        learning_rate (Float): The learning rate
    """
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')
    epochs = 100

    # K-fold Cross Validation model evaluation -->x
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_x)):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        print(train_ids.shape)
        print(test_ids.shape)

        # Define data loaders for training and testing data in this fold
        trainloader_x = torch.utils.data.DataLoader(
                        dataset_x,  sampler=train_subsampler)
        trainloader_y = torch.utils.data.DataLoader(
                        dataset_y, sampler=train_subsampler)
        testloader_x = torch.utils.data.DataLoader(
                        dataset_x, sampler=test_subsampler)
        testloader_y = torch.utils.data.DataLoader(
                        dataset_y, sampler=test_subsampler)

        #initiate optimizer and network
        net = Net()
        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)#weight_decay = 0.001)#, momentum=0.5)
        # create a loss function
        criterion = nn.MSELoss()
        plot_loss_train=[]
        plot_loss_test=[]

        for epoch in range(epochs):
            print(epoch)
            a = torch.ones(800,3,1)
            net_out_np = []
            net_out_loss =[]
            current_loss = 0.0

            #training
            for input_, output_ in zip(trainloader_x, trainloader_y):
                data = input_.reshape(3,3)
                target = output_.reshape(3,1)
                optimizer.zero_grad()
                net_out = net(data)
                loss = criterion((net_out).float(), target.float())
                loss.backward()
                optimizer.step()
                a = net_out.detach().numpy()
                s = np.asarray(a)
                net_out_np.append(s)
                net_out_loss.append(loss)
            plot_loss_train.append(np.sum(net_out_loss)/800)
            print(plot_loss_train)

            net.eval()
            b = torch.ones(200,3,1)
            net_out_np_test = []
            net_out_loss_test =[]

            #validation
            for input_, output_ in zip(testloader_x, testloader_y):
                data = input_.reshape(3,3)
                target = output_.reshape(3,1)
                optimizer.zero_grad()
                net_out = net(data)
                loss = criterion((net_out).float(), target.float())
                loss.backward()
                optimizer.step()
                b = net_out.detach().numpy()
                s = np.asarray(b)
                net_out_np_test.append(s)
                net_out_loss_test.append(loss)
            plot_loss_test.append(np.sum(net_out_loss_test)/200)
            print(plot_loss_test)

        #testing
        #Saving & Loading Model for Inference
        model = Net()
        PATH = "/Users/Nataka/Desktop/Master_Thesis/Nakamura_sensai/Model_Parameters/June18_Attention_accuracy_Final-Graph_crossvalidation.pth"
        torch.save(net.state_dict(), PATH)
        model.load_state_dict(torch.load(PATH))
        model.eval()

        c = torch.ones(200,3,1)
        net_out_np_test = []
        net_out_loss_test =[]
        correct, total = 0, 0
        with torch.no_grad():
            for input_, output_ in zip(testloader_x, testloader_y):
                data = input_.reshape(3,3)
                target = output_.reshape(3,1)
                net_out = net(data)
                loss = criterion((net_out).float(), target.float())
                c = net_out.detach().numpy()
                s = np.asarray(c)
                net_out_np_test.append(s)
                net_out_loss_test.append(loss)
            plot_loss_test.append(np.sum(net_out_loss_test)/200)
            print(plot_loss_test)





#########################################################
#---------------------USED FUNCTIONS--------------------#
#########################################################


def encode_to_rgb(Color_Name):
    """A function to get the RGB values of the colors defined in Tkinter

    Args:
        Color_Name (string): Colorname in Tkinter

    Returns:
        Array: Containig RGB values of Color_Name
    """
    # Defining the path to some .xls file containg the RGB values
    # for the Tkinter colors
    color_dataset_path = os.path.join(data_dir, "Handmade_color_dataset.xls")

    # Loading the RGB values of the Tkinter colors
    rgb_encode = pd.read_excel(color_dataset_path)

    color_listed = rgb_encode['color']
    n_rgbencode = len(rgb_encode)

    tmp = 0
    for i in range(n_rgbencode):
        if color_listed[i] == Color_Name:
            tmp = rgb_encode.iloc[i][1:4]
    return tmp

#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#


def collected_data_to_vector(CollectedData):
    """Converts the several given color options and user-chosen colors into some vectors 'choices2vector'
    and 'chosen_color2vector'.

    Args:
        CollectedData (pandas.DataFrame object): A DataFrame consisting of four columns: The first three
            containing the several given color options (three per round) and the user-chosen one. All given
            as RGB values


    Returns:
        [array, array]: The two vectors 'choices2vector' and 'chosen_color2vector' containing given options
            the users choice.
    """
    n_collectedData = len(CollectedData)
    choices2vector = []
    chosen_color2vector = []
    for i in range(n_collectedData):
        color_option_1 = [encode_to_rgb(CollectedData['color_1'][i]).to_numpy( dtype=object)/255]
        color_option_2 = [encode_to_rgb(CollectedData['color_2'][i]).to_numpy( dtype=object)/255 ]
        color_option_3 = [ encode_to_rgb(CollectedData['color_3'][i]).to_numpy( dtype=object)/255  ]
        chosen_color = encode_to_rgb(CollectedData['chosen_color'][i]).to_numpy( dtype=object)/255
        color_options_tuple = np.append(np.append(color_option_1,color_option_1, axis=0),color_option_1, axis = 0 )
        chosen_color2vector.append(chosen_color.astype(float))
        choices2vector.append(color_options_tuple.astype(float))
    return [chosen_color2vector, choices2vector]


#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#


def histogram_of_chosen_colors(CollectedData):
    """Counts the occurrences of the several subject chosen colors and return these as a dictionary, whose keys are the colors
    and values are the number of occurrences.

    Args:
        CollectedData (pandas.DataFrame object): The collected Dataset

    Returns:
        [Dictionary]: Dictionary whose keys are strings giving the several chosen colors and whose values are their occurences during the experiment
    """
    colors_chosen = []
    n_collectedData = len(CollectedData)
    for i in range(n_collectedData):
        d = CollectedData['chosen_color'][i]
        colors_chosen.append(d)
    return Counter(colors_chosen)


#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#


def fuzzy_similarity(color_a,color_b):
    """Compares the RGB values of the two given colors 'color_a' and 'color_b'
    by calculating the quotient of the R, G and B values and returns it. This is done such that the
    the qoutient is allways smaller than 1. If one of the values is 0, the quotient is set to 0.

    Args:
        color_a (Tuple): RGB-tuple of the first color
        color_b (Tuple): RGB-tuple of the second color

    Returns:
        Tuple: Tuple consisting of the qoutients of the RGB-values
    """

    if len(color_a) == 1:
        num = min(color_a,color_b)
        den = max(color_a,color_b)
        return num/den
    if len(color_a) >1:
        array = []
        for i in range(len(color_a)):
            num = min(color_a[i],color_b[i])
            den = max(color_a[i],color_b[i])
            if den !=0:
                result = num/den
            if den ==0:
                result = 0.0
            array.append(result)
        return array


#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#



def first_feature_elimination(color):
    sum_ = np.sum(color, axis=0)
    return  np.where(sum_ == max(sum_))



#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#


def other_colors(color_options,chosen_color):
    """Function that returns the remaining two not chosen colors of the three given options

    Args:
        color_options (Array): Array containing RGB-tuples of the three color options
        chosen_color (Tuple): Tuple conataining the RGB values of the chose color of that three options

    Returns:
        Array: Array consisiting of the RGB-tuples of the two not chosen colors
    """
    other2colors = []
    for i in range(len(color_options)):
        if np.array_equal(color_options[i],chosen_color) != True:
            other2colors.append(color_options[i])
    return other2colors


#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#

def new_encoding(X,Y):
    """Whenever the R, G or B value of two or more of the given color options
    are the same, the fuzzy similarity will return 1, thereby forcing our proposed
    model to put attention on this feature. In particular, when all three options
    have, e.g. the same R value equal to 1, this seems to be misleading. Therefore
    we decided to encode our RGB features in a new fashion which takes such cases
    into account. We thereby also considered the problem, that whenever one (or more)
    of the R, G or B values of one (ore more) of the options is zero (in these
    cases the fuzzy similarity will return 0, which also turns out to be misleading).
    The new encoding returns a one for the "truly preferred" color and zero otherwise
    and is done as follows:

    Options to chose: color_1, color_2, color_3
    Chosen color: color_3

    Other_colors function find other two colors out of chosen color from
    the options which results color_1 and color_2

    Start: Eliminating unimportant feature based on the chosen color

    Set1 : {Chosen color (color_3) and color_1}
    Set2 : { Chosen color (color_3) and color_2}
    Apply fuzzy similarity logic for Set1 and Set2:
    Best case scenario: It will result one of the feature that should not to be considered
    Worst case scenario: It might result in all zeros or ones and ends up not finding which
    feature has to be eliminated. This case new encoding technique find max and min
    for combinations for set1 and set2 to end up eliminating the first feature.
    once the unimportant feature is eliminated, other features are fed into new encoding
    function until ending up with one important feature.


    Args:
        X (List): The RGB-Values of the three optional colors of one experiment run
        Y (List): The RGB-Values of the participants chosen color

    Returns:
        [List]: A three-element-list, whose entries are all 0 except one, which is one.
            This entry determines which R, G or B value is the one "preferred"
            among the given options.
    """
    new_output = []
    new_output_string = []
    for k in range(len(y)):
        target_x = x[k]
        target_y = y[k]
        p2=[]
        resulto = 0
        other2colors = other_colors(target_x, target_y)
        result_1 = fuzzy_similarity(other2colors[0], other2colors[1])
        result_1_index = max(result_1)


        first_feature_index = np.asarray(np.where(result_1 == result_1_index))
        first_feature_index_1 = first_feature_index

        if first_feature_index_1.shape[1] == 0:
            first_feature_index = np.asarray(result_1[0])
        if first_feature_index_1.shape[1] > 1:
            print(target_x)
            print(first_feature_index)
            first_feature_index = np.asarray(first_feature_elimination(target_x))
            if first_feature_index.shape[1] >1:
                first_feature_index =np.random.choice([first_feature_index[0][0],first_feature_index[0][1]],1)
        #print("    X :  "+str(k))
        #print(first_feature_index)

        step_21 = np.nan_to_num(fuzzy_similarity(y[k], other2colors[0]))
        second_1_feature = min(step_21)
        second_1_feature_index = np.asarray(np.where(step_21 == second_1_feature))

        #print((second_1_feature_index))
        second_1_feature_index_shape = second_1_feature_index.shape[1]

        step_22 = np.nan_to_num(fuzzy_similarity(y[k], other2colors[1]))
        second_2_feature = min(step_22)
        second_2_feature_index = np.asarray(np.where(step_22 == second_2_feature))

        #print((second_2_feature_index))
        second_2_feature_index_shape = second_2_feature_index.shape[1]

        #If second_1_feature_index and second_2_feature_index has same shape
        if   second_1_feature_index_shape ==1 and  second_2_feature_index_shape == 1  :
            if second_2_feature_index == second_1_feature_index:
                if second_2_feature_index!= first_feature_index:
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    listo.remove(second_2_feature_index)
                    resulto = listo
                    if len(listo) == 2:
                        listo = np.random.choice([listo[0],listo[1]],1)
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
                if second_2_feature_index == first_feature_index: #Rarecase
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    resulto = listo
                    if len(listo) == 2:
                        listo = np.random.choice([listo[0],listo[1]],1)
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
            if second_2_feature_index != second_1_feature_index:
                if second_2_feature_index!= first_feature_index and second_1_feature_index!= first_feature_index:
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    listo.remove(np.random.choice([second_2_feature_index[0][0],second_1_feature_index[0][0]],1)) #random
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
                if second_2_feature_index == first_feature_index or second_1_feature_index == first_feature_index: #Rarecase
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    if second_2_feature_index[0][0] in listo:
                        listo.remove(second_2_feature_index[0][0])
                    if second_1_feature_index[0][0] in listo:
                        listo.remove(second_1_feature_index[0][0])
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
        if   second_1_feature_index_shape ==2 and  second_2_feature_index_shape == 2  :
            #print("huh_2")
            if second_2_feature_index.all() == second_1_feature_index.all():
                if second_2_feature_index.all() != first_feature_index:
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    listo.remove(second_2_feature_index.all())
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
                if second_2_feature_index.all() == first_feature_index: #Rarecase
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    if second_2_feature_index[0][0] in listo:
                        listo.remove(second_2_feature_index[0][0])
                    if second_2_feature_index[0][1] in listo:
                        listo.remove(second_2_feature_index[0][1])
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
            if second_2_feature_index.all() != second_1_feature_index.all():
                if second_2_feature_index.all()!= first_feature_index and second_1_feature_index.all()!= first_feature_index:
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    listo.remove(np.random.choice([second_2_feature_index[0][0],second_1_feature_index[0][0]],1)) #random
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
                if second_2_feature_index.all() == first_feature_index or second_1_feature_index.all() == first_feature_index: #Rarecase
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    if second_2_feature_index[0][0] in listo:
                        listo.remove(second_2_feature_index[0][0])
                    if second_1_feature_index[0][0] in listo:
                        listo.remove(second_1_feature_index[0][0])
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto      :"+str(resulto))
        if  second_1_feature_index_shape ==3 and  second_2_feature_index_shape == 3  : #Rare case: Remove the row from dataset
            concat_ = np.unique(np.concatenate((second_1_feature_index, second_2_feature_index), axis = 1))
            if len(concat_) ==3:
                #print(concat_[0])
                print(target_y)
                check = np.where(target_y ==max(target_y))
                resulto = check[0]

        #If second_1_feature_index and second_2_feature_index, both has different shape
        if second_1_feature_index_shape ==1  and second_2_feature_index_shape==2:
            concat_ = np.unique(np.concatenate((second_1_feature_index, second_2_feature_index), axis = 1))
            if len(concat_) == 2:
                if concat_.all() != first_feature_index:
                        #print("Pappu")
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    for i in second_2_feature_index:
                        for j in i:
                            if j== second_1_feature_index and j in listo:
                                listo.remove(j)
                    resulto = listo
                    if len(listo) == 2:
                        listo = np.random.choice([listo[0],listo[1]],1)
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                        #print("resulto000     :"+str(resulto))
                if concat_.all() == first_feature_index: #Rarecase

                        #print("Pappu _2")
                        listo = [0,1,2]
                        listo.remove(first_feature_index)
                        if concat_[0] in listo:
                            listo.remove(concat_[0])
                        if concat_[1] in listo:
                            listo.remove(concat_[1])
                        if len(listo) == 0:
                            listo = np.random.choice([concat_[0],concat_[1]],1)
                        resulto = listo
                        print(listo)
                        if target_y[listo] ==0:
                            check = np.where(target_y ==max(target_y))
                            resulto = check[0]
                        #print("resulto000      :"+str(resulto))
            if len(concat_) ==3:
                listo = [0,1,2]
                #print(concat_[0])
                listo.remove(first_feature_index)
                if second_1_feature_index!= first_feature_index:
                    listo.remove(second_1_feature_index)
                if second_1_feature_index== first_feature_index:
                    listo.remove(second_2_feature_index.any())

                resulto = listo
                if target_y[listo] ==0:
                    check = np.where(target_y ==max(target_y))
                    resulto = check[0]
                #print("resulto000      :"+str(resulto))

        if second_1_feature_index_shape ==2  and second_2_feature_index_shape==1:
            concat_ = np.unique(np.concatenate((second_1_feature_index, second_2_feature_index), axis = 1))
            #print(concat_)
            if concat_.all() != first_feature_index:
                    #print("Pappu")
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    for i in second_1_feature_index:
                        for j in i:
                            if j== second_2_feature_index and j in listo:
                                listo.remove(j)
                    resulto = listo
                    if len(listo) == 2:
                        listo = np.random.choice([listo[0],listo[1]],1)
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto111     :"+str(resulto))
            if concat_.all() == first_feature_index: #Rarecase
                    #print("Pappu _2")
                    listo = [0,1,2]
                    listo.remove(first_feature_index)
                    if concat_[0] in listo:
                        listo.remove(concat_[0])
                    if concat_[1] in listo:
                        listo.remove(concat_[1])
                    if len(listo) == 0:
                        listo = np.random.choice([concat_[0],concat_[1]],1)
                    resulto = listo
                    if len(listo) == 2:
                        listo = np.random.choice([listo[0],listo[1]],1)
                    resulto = listo
                    if target_y[listo] ==0:
                        check = np.where(target_y ==max(target_y))
                        resulto = check[0]
                    #print("resulto111     :"+str(resulto))
        if second_1_feature_index_shape ==3  and second_2_feature_index_shape==1:

            concat_ = np.unique(np.concatenate((second_1_feature_index, second_2_feature_index), axis = 1))
            if len(concat_) ==3:
                #print(concat_[0])
                listo = [0,1,2]
                listo.remove(first_feature_index)
                for i in second_1_feature_index:
                    for j in i:
                        if j== second_2_feature_index and j in listo:
                            listo.remove(j)
                if len(listo) == 2:
                    listo = np.random.choice([listo[0],listo[1]],1)
                resulto = listo
                if target_y[listo] ==0:
                    check = np.where(target_y ==max(target_y))
                    resulto = check[0]
                #print("resulto000      :"+str(resulto))
        if second_1_feature_index_shape ==1  and second_2_feature_index_shape==3:
            concat_ = np.unique(np.concatenate((second_1_feature_index, second_2_feature_index), axis = 1))
            if len(concat_) ==3:
                listo = [0,1,2]
                #print(concat_[0])
                listo.remove(first_feature_index)
                if concat_[0] in listo:
                    listo.remove(concat_[0])
                if concat_[1] in listo:
                    listo.remove(concat_[1])
                if len(listo) == 0:
                    listo = np.random.choice([concat_[0],concat_[1]],1)
                resulto = listo
                if target_y[listo] ==0:
                    check = np.where(target_y ==max(target_y))
                    resulto = check[0]
                #print("resulto000      :"+str(resulto))
        if second_1_feature_index_shape ==2  and second_2_feature_index_shape==3:
            concat_ = np.unique(np.concatenate((second_1_feature_index, second_2_feature_index), axis = 1))
            if len(concat_) ==3:
                listo = [0,1,2]
                #print(concat_[0])
                listo.remove(first_feature_index)
                if concat_[0] in listo:
                    listo.remove(concat_[0])
                if concat_[1] in listo:
                    listo.remove(concat_[1])
                if len(listo) == 0:
                    listo = np.random.choice([concat_[0],concat_[1]],1)
                resulto = listo
                if target_y[listo] ==0:
                    check = np.where(target_y ==max(target_y))
                    resulto = check[0]
                #print("resulto000      :"+str(resulto))
        if second_1_feature_index_shape ==3  and second_2_feature_index_shape==2:
            concat_ = np.unique(np.concatenate((second_1_feature_index, second_2_feature_index), axis = 1))
            if len(concat_) ==3:
                listo = [0,1,2]
                listo.remove(first_feature_index)
                for i in second_1_feature_index:
                    for j in i:
                        if j == second_2_feature_index.any() and j in listo:
                            listo.remove(j)
                if len(listo) == 2:
                    listo = np.random.choice([listo[0],listo[1]],1)
                resulto = listo
                if target_y[listo] ==0:
                    check = np.where(target_y ==max(target_y))
                    resulto = check[0]

                #print("resulto000      :"+str(resulto))

        o = target_y
        o = o.reshape(3,1)
        p1 = o.reshape(3,1)
        if resulto[0] == 0:
            o[1]=p1[1] = 0
            o[2]=p1[2] = 0
            p=p1
            p2 = np.append(o,['Red'])
        if resulto[0] == 1:
            o[0]=p1[0] = 0
            o[2] =p1[2]= 0
            p=p1
            p2 = np.append(o,['Green'])
        if resulto[0] == 2:
            o[0]  =p1[0]= 0
            o[1]  =p1[1]= 0
            p=p1
            p2 = np.append(o,['Blue'])
        new_output.append(p1)
        new_output_string.append(p2)
    return new_output, new_output_string

#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#


def Ranking_table(y):
    table = []
    list_ = list(dict.fromkeys(y))
    for i in range(len(list_)):
        a = y.count(list_[i])
        table.append(np.concatenate(([list_[i]], [a]),axis = 0))
    return table


#---------------------------------------------------------#
#                     New Function                        #
#---------------------------------------------------------#

def preference_table(h, preference):
    store_color = []
    for i in range(len(h)):
        input_ = h[i]
        if input_[3] ==preference:
            if preference == 'Red':
                color = Irene_collectedData['chosen_color'][i]
                store_color.append([Irene_collectedData['chosen_color'][i], input_[0]])
            if preference == 'Green':
                store_color.append([Irene_collectedData['chosen_color'][i], input_[1]])
            if preference == 'Blue':
                store_color.append([Irene_collectedData['chosen_color'][i], input_[2]])
    return store_color

priority_list = preference_table(new_y[1], 'Blue')


if __name__ == "__main__":
    main()
