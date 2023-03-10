
#Libraries needed
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from skimage.transform import rescale


"""Class to store a training or validation dataset, derived from the PyTorch Dataset/DataLoader classes"""
class AudioDataset(Dataset):
    """Mel Spectrogram dataset"""
    def __init__(self, root_dir, keyword):
        #Save the root directory and keyword name
        self.root_dir = root_dir
        self.keyword = keyword

        #Get the names of files of the true keywords and create their labels = 1
        keywordFileNames = os.listdir(self.root_dir+"/"+self.keyword)
        keywordFileNames = [self.root_dir+"/"+self.keyword+"/"+filename for filename in keywordFileNames]
        keywordLabels = torch.ones(len(keywordFileNames), dtype=torch.long)

        #Get the names of files of non keywords and create their labels = 0
        nonKeywordFileNames = os.listdir(self.root_dir+"/non_keyword")
        nonKeywordFileNames = [self.root_dir+"/non_keyword/"+filename for filename in nonKeywordFileNames]
        nonKeywordLabels = torch.zeros(len(nonKeywordFileNames), dtype=torch.long)

        #Concatenate the keyword files with the non keyword files as well as their labels
        #Saves these as class variables for fast loading
        self.filenames = keywordFileNames + nonKeywordFileNames
        self.labels = torch.cat((keywordLabels, nonKeywordLabels), dim=0)

    #Get the number of files/samples in the dataset
    def __len__(self):
        return len(self.filenames)

    #Get an item/sample from the dataset
    def __getitem__(self, idx):
        #Load the audio at idx position and return the audio, label, and filename loaded
        audio = torch.load(self.filenames[idx])
        return audio, self.labels[idx], self.filenames[idx]


"""Function to train the model"""
def train(model, criterion, optimizer, dataLoader, GPU, gradAcc):
    #Set the model to training mode and create variables to store the
    #loss, total batches, true positives, false positives, true negatives,
    #false negatives, and number of samples throughout training
    model.train()
    trainLoss = 0
    totalBatches = 0
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    sampleCount = 0
    sampleTotal = len(dataLoader.dataset)
    counter = 0

    #For gradient accumulation: batchSize * accumulator = batch size w/o accumulator
    #Set this manually. GradientAccumulation will only run for models defined in train.py
    accumulator = 8

    for batchNumber, (batchAudio, batchLabels, batchFilenames) in enumerate(dataLoader):
        #If the GPU is avaliable send the data to it
        if (GPU != None):
            batchAudio = batchAudio.to(GPU)
            batchLabels = batchLabels.to(GPU)
            
        #Since we are dealing with CNNs that expect RGB input add a
        #singular dimension on top of the Mel Spectrogram
        batchAudio = torch.unsqueeze(batchAudio, 1)
        #Send the data through the network and compute the output and loss
        output = model(batchAudio)
        print(output.size(), batchLabels.size())
        loss = criterion(output, batchLabels)
        #Compute the stats
        tPositive, fPositive, tNegative, fNegative = compute_stats(output, batchLabels)
        truePositives += tPositive
        falsePositives += fPositive
        trueNegatives += tNegative
        falseNegatives += fNegative
        trainLoss += loss.item()
        
        #Compute gradients and update weights
        #With gradient accumulation:
        if (gradAcc):
            if (batchNumber == 0):
                optimizer.zero_grad()

            (loss / accumulator).backward()
            nextBatch = batchNumber + 1
            
            if ((nextBatch % accumulator == 0) or (nextBatch == len(dataLoader))):
                optimizer.step()
                optimizer.zero_grad()
                
        #Without gradient accumulation:
        else:
            # clear gradients for next train
            optimizer.zero_grad()
                
            loss.backward()
            optimizer.step()
            
        #Add the counts to total batches and sample count
        totalBatches += 1
        sampleCount += batchLabels.size(0)
        
    #Return the trained model
    #Compute and return the epoch average loss, epoch accuracy,
    #epoch precision, epoch recall, and f1 score
    accuracy = (truePositives + trueNegatives) / sampleCount
    precision = 0
    if(truePositives + falsePositives != 0):
        precision = truePositives / (truePositives + falsePositives)
    recall = 0
    if(truePositives + falseNegatives != 0):
        recall = truePositives / (truePositives + falseNegatives)
    f1 = 0
    if(precision + recall != 0):
        f1 = 2 * (precision * recall) / (precision + recall)
    return model, trainLoss/totalBatches, accuracy, precision, recall, f1


"""Function to evaluate the model"""
def evaluate(model, criterion, dataLoader, GPU, batchLimit, printInfo):
    #Set the model to eval mode and create variables to store the
    #loss, total batches, true positives, false positives, true negatives,
    #false negatives, number of samples throughout training, and inference time
    model.eval()
    evalLoss = 0
    totalBatches = 0
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    sampleCount = 0
    totalTime = 0
    for batchNumber, (batchAudio, batchLabels, batchFilenames) in enumerate(dataLoader):
        if ((batchLimit == None) or (batchNumber < batchLimit)):
            #If the GPU is avaliable send the data to it
            if (GPU != None):
                batchAudio = batchAudio.to(GPU)
                batchLabels = batchLabels.to(GPU)
                
            #Since we are dealing with CNNs that expect RGB input add a
            #singular dimension on top of the Mel Spectrogram
            batchAudio = torch.unsqueeze(batchAudio, 1)
            #Send the data through the network and compute the output, loss, and inference time
            startTime = time.time()
            output = model(batchAudio)
            #print(output)
            totalTime += time.time() - startTime
            loss = criterion(output, batchLabels)
            #Compute the stats
            tPositive, fPositive, tNegative, fNegative = compute_stats(output, batchLabels)
            truePositives += tPositive
            falsePositives += fPositive
            trueNegatives += tNegative
            falseNegatives += fNegative
            evalLoss += loss.item()
            #Add the counts to total batches and sample count
            totalBatches += 1
            sampleCount += batchLabels.size(0)
    
    if (sampleCount != 0):
        #Compute and return the epoch average loss, epoch accuracy,
        #epoch precision, epoch recall, and epoch f1 score
        accuracy = (truePositives + trueNegatives) / sampleCount
    else:
        accuracy = 0
    
    if (printInfo):
        print("True Positives: " + str(truePositives))
        print("False Positives: " + str(falsePositives))
        print("True Negatives: " + str(trueNegatives))
        print("False Negatives: " + str(falseNegatives))
        print(sampleCount)
        
    precision = 0
    if(truePositives + falsePositives != 0):
        precision = truePositives / (truePositives + falsePositives)
        
    recall = 0
    if(truePositives + falseNegatives != 0):
        recall = truePositives / (truePositives + falseNegatives)

    f1 = 0
    if(precision + recall != 0):
        f1 = 2 * (precision * recall) / (precision + recall)
    return evalLoss/totalBatches, accuracy, precision, recall, f1, totalTime/totalBatches


"""Function to compute the stats from the model predictions"""
def compute_stats(predictions, targets):
    #Compute the number of true-positives, false-positives,
    #true-negatives, and false-negatives
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    #Get the predictions from our model
    predictions = torch.argmax(predictions, dim=-1)
    for i in range(predictions.size(0)):
        #Prediction and target match
        #print("Predictions: " + str(predictions[i]) + " Targets: " + str(targets[i]))
        if(predictions[i] == targets[i]):
            if(predictions[i] == 1):
                truePositive += 1
            else:
                trueNegative += 1
        #Prediction and target don't match
        else:
            if(predictions[i] == 1):
                falsePositive += 1
            else:
                falseNegative += 1
    return truePositive, falsePositive, trueNegative, falseNegative


"""Function to plot the results of training based on the stats computed"""
def plot(trainValues, validValues, graphType):
    #Compute the mean and standard deviation of train/validation data across runs
    trainMean = np.mean(trainValues, axis=0)
    trainStd = np.std(trainValues, axis=0)
    validMean = np.mean(validValues, axis=0)
    validStd = np.std(validValues, axis=0)

    #Create a numpy array to store the epoch values for plotting
    epochs = np.arange(trainValues.shape[-1], dtype=int)

    #Plot the results
    plt.errorbar(x=epochs, y=trainMean, yerr=trainStd, color='black', label='Training')
    plt.errorbar(x=epochs, y=validMean, yerr=validStd, color='red', label='Validation')
    plt.xlabel("Epoch Number")
    #Create the proper y-axis label based on the graph type being created
    if(graphType == 'loss'):
        plt.ylabel("Cross Entropy Loss")
    elif(graphType == 'accuracy'):
        plt.ylabel("Accuracy")
    elif(graphType == 'precision'):
        plt.ylabel("Precision Value")
    elif(graphType == 'recall'):
        plt.ylabel("Recall Value")
    elif(graphType == 'f1'):
        plt.ylabel("F1 Score")
    else:
        print("ERROR: Improper graph type input!")
        exit()
    plt.legend()
    plt.show()

################################################################################################################
################################################################################################################
"""For Multi-Objective Training"""


"""Class to store a training or validation dataset, derived from the PyTorch Dataset/DataLoader classes"""
class AudioDatasetMultiObjective(Dataset):
    """Mel Spectrogram dataset"""
    def __init__(self, root_dir, keyword):
        #Save the root directory, keyword, and words for multi-objective training
        self.root_dir = root_dir
        self.keyword = keyword
        self.words = os.listdir(self.root_dir)

        #Variables to store a list of all files in the training dataset,
        #keyword labels, and multi-objective training word labels
        self.filenames = []
        self.word_labels = None
        self.keyword_labels = None

        #Loop through all folders in the directory and get the
        #filenames and labels for each file
        for index, word in enumerate(self.words):
            #Get the file names
            wordFileNames = os.listdir(self.root_dir+"/"+word)
            wordFileNames = [self.root_dir+"/"+word+"/"+filename for filename in wordFileNames]
            #Create the multi-objective word labels
            word_labels = torch.zeros(len(wordFileNames), dtype=torch.long)
            word_labels[:] = index
            #Create the keyword labels
            keyword_labels = torch.zeros(len(wordFileNames), dtype=torch.float)
            if(word == self.keyword):
                keyword_labels[:] = 1
            #Add the filenames, multi-objective word labels, and keyword labels to
            #the proper class variables
            self.filenames += wordFileNames
            if(index == 0):
                self.word_labels = word_labels
                self.keyword_labels = keyword_labels
            else:
                self.word_labels = torch.cat((self.word_labels, word_labels), dim=0)
                self.keyword_labels = torch.cat((self.keyword_labels, keyword_labels), dim=0)

    #Get the number of files/samples in the dataset
    def __len__(self):
        return len(self.filenames)

    #Get an item/sample from the dataset
    def __getitem__(self, idx):
        #Load the audio at idx position and return the audio, label, and filename loaded
        audio = torch.load(self.filenames[idx])
        return audio, self.word_labels[idx], self.keyword_labels[idx], self.filenames[idx]


"""Function to train the model in multi-objective training case"""
def trainMultiObjective(model, criterion, optimizer, dataLoader, GPU):
    #Set the model to training mode and create variables to store the
    #loss, total batches, true positives, false positives, true negatives,
    #false negatives, and number of samples throughout training
    model.train()
    trainLoss = 0
    totalBatches = 0
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    sampleCount = 0
    #Create variables to store the total loss and label correct predictions
    trainLossTotal = 0
    trainCorrectPredictions = 0
    for batchNumber, (batchAudio, batchWordLabels, batchKeywordLabels, batchFilenames) in enumerate(dataLoader):
        #If the GPU is avaliable send the data to it
        if (GPU != None):
            batchAudio = batchAudio.to(GPU)
            batchWordLabels = batchWordLabels.to(GPU)
            batchKeywordLabels = batchKeywordLabels.to(GPU)
        #Since we are dealing with CNNs that expect RGB input add a
        #singular dimension on top of the Mel Spectrogram
        batchAudio = torch.unsqueeze(batchAudio, 1)
        #Send the data through the network and compute the output
        #Pass the wakeword detector through the sigmoid as it needs to be a probability
        output = model(batchAudio)
        output[:,0] = torch.sigmoid(output[:,0])
        #Compute the wakeword and class label prediction losses
        labelLoss = criterion(output[:,1:], batchWordLabels)
        wakewordLoss = neg_log_likelihood(output[:,0], batchKeywordLabels)
        loss = labelLoss + wakewordLoss
        #Compute the stats
        tPositive, fPositive, tNegative, fNegative = compute_stats2(output[:,0], batchKeywordLabels)
        truePositives += tPositive
        falsePositives += fPositive
        trueNegatives += tNegative
        falseNegatives += fNegative
        trainLoss += wakewordLoss.item()
        trainLossTotal += loss.item()
        # clear gradients for next train
        optimizer.zero_grad()
        #Compute gradients and update weights
        loss.backward()
        optimizer.step()
        #Add the counts to total batches and sample count
        totalBatches += 1
        sampleCount += batchKeywordLabels.size(0)
    #Return the trained model
    #Compute and return the epoch average loss, epoch accuracy,
    #epoch precision, epoch recall, and f1 score
    accuracy = (truePositives + trueNegatives) / sampleCount
    precision = 0
    if(truePositives + falsePositives != 0):
        precision = truePositives / (truePositives + falsePositives)
    recall = 0
    if(truePositives + falseNegatives != 0):
        recall = truePositives / (truePositives + falseNegatives)
    f1 = 0
    if(precision + recall != 0):
        f1 = 2 * (precision * recall) / (precision + recall)
    return model, trainLoss/totalBatches, accuracy, precision, recall, f1, trainLossTotal/totalBatches, trainCorrectPredictions/sampleCount


"""Function to train the model in multi-objective training case
def evaluateMultiObjective(model, criterion, optimizer, dataLoader, GPU):
    #Set the model to training mode and create variables to store the
    #loss, total batches, true positives, false positives, true negatives,
    #false negatives, and number of samples throughout training
    model.train()
    trainLoss = 0
    totalBatches = 0
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    sampleCount = 0
    #Create variables to store the total loss and label correct predictions
    trainLossTotal = 0
    trainCorrectPredictions = 0
    for batchNumber, (batchAudio, batchWordLabels, batchKeywordLabels, batchFilenames) in enumerate(dataLoader):
        #If the GPU is avaliable send the data to it
        if torch.cuda.is_available():
            batchAudio = batchAudio.to(GPU)
            batchWordLabels = batchWordLabels.to(GPU)
            batchKeywordLabels = batchKeywordLabels.to(GPU)
        #Since we are dealing with CNNs that expect RGB input add a
        #singular dimension on top of the Mel Spectrogram
        batchAudio = torch.unsqueeze(batchAudio, 1)
        #Send the data through the network and compute the output
        #Pass the wakeword detector through the sigmoid as it needs to be a probability
        output = model(batchAudio)
        output[:,0] = torch.sigmoid(output[:,0])
        #Compute the wakeword and class label prediction losses
        labelLoss = criterion(output[:,1:], batchWordLabels)
        wakewordLoss = neg_log_likelihood(output[:,0], batchKeywordLabels)
        loss = labelLoss + wakewordLoss
        #Compute the stats
        tPositive, fPositive, tNegative, fNegative = compute_stats2(output[:,0], batchKeywordLabels)
        truePositives += tPositive
        falsePositives += fPositive
        trueNegatives += tNegative
        falseNegatives += fNegative
        trainLoss += wakewordLoss.item()
        trainLossTotal += loss.item()
        # clear gradients for next train
        optimizer.zero_grad()
        #Compute gradients and update weights
        loss.backward()
        optimizer.step()
        #Add the counts to total batches and sample count
        totalBatches += 1
        sampleCount += batchKeywordLabels.size(0)
    #Return the trained model
    #Compute and return the epoch average loss, epoch accuracy,
    #epoch precision, epoch recall, and f1 score
    accuracy = (truePositives + trueNegatives) / sampleCount
    precision = 0
    if(truePositives + falsePositives != 0):
        precision = truePositives / (truePositives + falsePositives)
    recall = 0
    if(truePositives + falseNegatives != 0):
        recall = truePositives / (truePositives + falseNegatives)
    f1 = 0
    if(precision + recall != 0):
        f1 = 2 * (precision * recall) / (precision + recall)
    return model, trainLoss/totalBatches, accuracy, precision, recall, f1, trainLossTotal/totalBatches, trainCorrectPredictions/sampleCount"""


"""Compute the loss for the Wakeword only using negative log likelihood"""
def neg_log_likelihood(predictions, targets):
    return -torch.mean(targets*torch.log(predictions), dim=-1)


"""Function to compute the stats from the model predictions"""
def compute_stats2(predictions, targets):
    #Compute the number of true-positives, false-positives,
    #true-negatives, and false-negatives
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    threshold = 0.5
    #Get the predictions from our model
    predictions = predictions > threshold
    targets = targets > threshold
    for i in range(predictions.size(0)):
        #Prediction and target match
        if(predictions[i] == targets[i]):
            if(predictions[i] == 1):
                truePositive += 1
            else:
                trueNegative += 1
        #Prediction and target don't match
        else:
            if(predictions[i] == 1):
                falsePositive += 1
            else:
                falseNegative += 1
    return truePositive, falsePositive, trueNegative, falseNegative
