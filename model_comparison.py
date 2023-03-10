#Libraries needed
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


#Load in the numpy data at 'path' and append to the list
def load_data(list, path):
    data = np.load(path)
    list.append(data)
    return list


#Plot the data corresponding to each model
def plot(models, data, trainingData, yAxis, title):
    #Check to ensure that we have data corresponding to each model
    if(len(models) != len(data)):
        print("ERROR: Should have data corresponding to each model!")
        exit()

    #Create a numpy array to store the epoch values for plotting
    epochs = np.arange(data[0].shape[-1], dtype=int)

    #Loop through all the models and plot the corresponding data
    for i in range(len(models)):
        #Compute the mean of train/validation data across runs
        mean = np.mean(data[i], axis=0)

        #Get the color (8 models max)
        color = None
        if(i%8==0): color = 'blue'
        elif(i%8==1): color = 'green'
        elif(i%8==2): color = 'red'
        elif(i%8==3): color = 'cyan'
        elif(i%8==4): color = 'magenta'
        elif(i%8==5): color = 'yellow'
        elif(i%8==6): color = 'black'
        else: color = 'orange'

        #Plot the data
        if(trainingData == True):
            plt.plot(epochs, mean, color=color, linestyle='dashed', label=models[i])
        else:
            plt.plot(epochs, mean, color=color, label=models[i])

    #Create the axis and show the plot
    plt.xlabel("Epoch Number")
    plt.ylabel(yAxis)
    plt.title(title)
    plt.legend()
    plt.show()


#Main function
if __name__ == '__main__':
    #Parse the command line arguements
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-folder", type=str, default='checkpoints/marvin', help='path to the folder containing the trained models')
    parser.add_argument('--models', type=str, required=True, nargs='+', help='the models to produce the graphics over')
    args = parser.parse_args()

    #Create the path for where to load the training results from, get the models list
    main_path = args.model_folder
    models = args.models

    #Create lists to store the training/validation loss, accuracy,
    #precision, recall, and f1 scores
    trainLoss = []
    trainAccuracy = []
    trainPrecision = []
    trainRecall = []
    trainF1 = []
    validLoss = []
    validAccuracy = []
    validPrecision = []
    validRecall = []
    validF1 = []

    #Loop through all the models and get each one's stats
    for model in models:
        #Create the path to load the model data from
        path = os.path.join(main_path, model)
        #Load the data for each model and append to the corresponding list
        trainLoss = load_data(trainLoss, os.path.join(path, "trainLoss.npy"))
        trainAccuracy = load_data(trainAccuracy, os.path.join(path, "trainAccuracy.npy"))
        trainPrecision = load_data(trainPrecision, os.path.join(path, "trainPrecision.npy"))
        trainRecall = load_data(trainRecall, os.path.join(path, "trainRecall.npy"))
        trainF1 = load_data(trainF1, os.path.join(path, "trainF1.npy"))
        validLoss = load_data(validLoss, os.path.join(path, "validLoss.npy"))
        validAccuracy = load_data(validAccuracy, os.path.join(path, "validAccuracy.npy"))
        validPrecision = load_data(validPrecision, os.path.join(path, "validPrecision.npy"))
        validRecall = load_data(validRecall, os.path.join(path, "validRecall.npy"))
        validF1 = load_data(validF1, os.path.join(path, "validF1.npy"))

    #Plot the training results
    plot(models, trainLoss, True, "Cross Entropy Loss", "Training Loss vs Epoch")
    plot(models, trainAccuracy, True, "Accuracy", "Training Accuracy vs Epoch")
    plot(models, trainPrecision, True, "Precision", "Training Precision vs Epoch")
    plot(models, trainRecall, True, "Recall", "Training Recall vs Epoch")
    plot(models, trainF1, True, "F1 Score", "Training F1 Score vs Epoch")

    #Plot the validation results
    plot(models, validLoss, False, "Cross Entropy Loss", "Validation Loss vs Epoch")
    plot(models, validAccuracy, False, "Accuracy", "Validation Accuracy vs Epoch")
    plot(models, validPrecision, False, "Precision", "Validation Precision vs Epoch")
    plot(models, validRecall, False, "Recall", "Validation Recall vs Epoch")
    plot(models, validF1, False, "F1 Score", "Validation F1 Score vs Epoch")
