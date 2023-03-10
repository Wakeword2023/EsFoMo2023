#Libraies needed
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#Custom files needed
import models


#From the following: https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
def plotWeights(modelName, model, layerNumber):
    #extracting the model features at the particular layer number
    weightTensor = None
    if((modelName == 'resnet18') or (modelName == 'resnet34') or (modelName == 'resnet50') or (modelName == 'resnet101') or (modelName == 'resnet152')):
        weightTensor = model.conv1.weight.data
    elif((modelName == 'vgg19_bn')):
        weightTensor = model.features[layerNumber].weight.data
    elif((modelName == 'shufflenet_v2_x0_5') or (modelName == 'shufflenet_v2_x1_0') or (modelName == 'shufflenet_v2_x1_5') or (modelName == 'shufflenet_v2_x2_0') or (modelName == 'custom_shufflenet')):
        weightTensor = model.conv1[0].weight.data
    elif((modelName == 'efficientnet_b0') or (modelName == 'efficientnet_b1') or (modelName == 'efficientnet_b2') or (modelName == 'efficientnet_b3') or (modelName == 'efficientnet_b4') or (modelName == 'efficientnet_b5') or (modelName == 'efficientnet_b6') or (modelName == 'efficientnet_b7') or (modelName == 'efficientnet_custom')):
        weightTensor = model._conv_stem.weight.data
    elif((modelName == 'densenet_bc_100_12') or (modelName == 'densenet_bc_250_24') or (modelName == 'densenet_bc_190_40')):
        weightTensor = model.conv1.weight.data
    elif(modelName == 'mobilenet_v2'):
        weightTensor = model.features[0][0].weight.data
    else:
        print("ERROR: Model Type Not Avaliable!!!")
        exit()

    #Plot the filters
    plot_filters_single_channel(weightTensor)
    return


#From the following: https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
def plot_filters_single_channel(t):
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 8
    nrows = 1 + nplots//ncols

    #Convert to numpy image
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    t = t.numpy()

    #Get the average at each CNN position
    average = np.average(t, axis=0)[0]

    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            #Set z and set to 0 if less than the average at this position
            z = t[i, j]
            z[z < average] = 0

            #Add the subplot
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(z)
            #Normalize z
            #npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            #npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))

            #Add to the graph
            ax1.imshow(npimg, cmap='plasma')
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()
    return


#Main function
if __name__ == '__main__':
    #Parse the command line arguements
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of NN')
    parser.add_argument("--model-path", type=str, required='True', help='path to the saved model')
    args = parser.parse_args()

    #Initialize the model, load it, and send to the CPU
    model = models.create_model(model_name=args.model, num_classes=2, in_channels=1)
    
    model = torch.load(args.model_path)
    model = model.to('cpu')

    #visualize weights for the model - first conv layer
    plotWeights(modelName=args.model, model=model, layerNumber=0)
