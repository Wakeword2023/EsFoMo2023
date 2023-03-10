#Libraies needed
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

#Custom files needed
import models
from utils import *


#Main function
if __name__ == '__main__':
    #Parse the command line arguements
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-dataset", type=str, default='MFCC_dataset/train', help='path of pre-computed Mel spectrogram train dataset')
    parser.add_argument("--valid-dataset", type=str, default='MFCC_dataset/valid', help='path of pre-computed Mel spectrogram validation dataset')
    parser.add_argument("--keyword", type=str, default='marvin', help='Name of the wakeword being used (must be the same as used in pre_process.py)')
    parser.add_argument("--batch-size", type=int, default=64, help='Size of a batch for training')
    parser.add_argument("--epochs", type=int, default=75, help='Number of epochs to train for')
    parser.add_argument("--runs", type=int, default=5, help='Number of total training runs to perform')
    parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
    parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
    parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate for optimization')
    parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.5, help='learning rate is multiplied by the gamma to decrease it')
    parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of NN')
    args = parser.parse_args()

    #Create the path for where to save the model, checkpoint, and training results
    path = 'checkpoints/'+args.keyword+'/'+args.model
    if not os.path.exists(path):
        os.makedirs(path)

    #Create the training dataset as well as its dataloader
    trainWakeword = AudioDataset(args.train_dataset, args.keyword)
    trainDataloader = DataLoader(dataset=trainWakeword, batch_size=args.batch_size, shuffle=True, num_workers=args.dataload_workers_nums)

    #Create the validation dataset as well as its dataloader
    validWakeword = AudioDataset(args.valid_dataset, args.keyword)
    validDataloader = DataLoader(dataset=validWakeword, batch_size=args.batch_size, shuffle=False, num_workers=args.dataload_workers_nums)

    #Get the GPU if avaliable for fast training
    GPU = None
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        GPU = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS GPU")
        GPU = torch.device("mps")

    #Numpy array to store the train / validation loss, accuracy,
    #precision, recall, and f1 scores across all runs of training
    #for all epochs
    shape = (args.runs, args.epochs)
    trainLossValues = np.zeros(shape)
    trainAccuracyValues = np.zeros(shape)
    trainPrecisionValues = np.zeros(shape)
    trainRecallValues = np.zeros(shape)
    trainF1Values = np.zeros(shape)
    validLossValues = np.zeros(shape)
    validAccuracyValues = np.zeros(shape)
    validPrecisionValues = np.zeros(shape)
    validRecallValues = np.zeros(shape)
    validF1Values = np.zeros(shape)

    #Variable used to store the best validation f1 score so we know when to save the model
    bestF1 = 0
    #Train the model over all runs
    for i in range(args.runs):
        print("########## Run:", i, "##########")
        #Create the model used for training and its criterion
        model = models.create_model(model_name=args.model, num_classes=2, in_channels=1)
        criterion = torch.nn.CrossEntropyLoss()

        #Send the model and criterion to the GPU
        if (GPU != None):
            model = model.to(GPU)
            criterion = criterion.to(GPU)

        #Create the optimizer for training our model
        optimizer = None
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        #Create the learning rate scheduler
        lr_scheduler = None
        if args.lr_scheduler == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs/10, gamma=args.lr_scheduler_gamma, last_epoch=-1)

        #Train the model over all epochs
        for j in range(args.epochs):
            #Reduce the learning rate according to StepLR
            if args.lr_scheduler == 'step':
                lr_scheduler.step()

            #If the model is large, use gradient accumulation to avoid memory error
            #Manually add models to the conditional below if you want to use gradient accumulation with them.
            #!You may need to change the accumulator for evaluate() in utils.py according to instructions there
            gradAcc = False
            if ((args.model).count("densenet") != 0):
                gradAcc = True
                
            #Perform training on the train set and evaluation of the validation set
            model, trainLoss, trainAccuracy, trainPrecision, trainRecall, trainF1 = train(model, criterion, optimizer, trainDataloader, GPU, gradAcc)
            validLoss, validAccuracy, validPrecision, validRecall, validF1, _ = evaluate(model, criterion, validDataloader, GPU)

            #Save the train/validation loss, accuracy, precision, recall, and f1 score per epoch per run
            trainLossValues[i,j] = trainLoss
            trainAccuracyValues[i,j] = trainAccuracy
            trainPrecisionValues[i,j] = trainPrecision
            trainRecallValues[i,j] = trainRecall
            trainF1Values[i,j] = trainF1
            validLossValues[i,j] = validLoss
            validAccuracyValues[i,j] = validAccuracy
            validPrecisionValues[i,j] = validPrecision
            validRecallValues[i,j] = validRecall
            validF1Values[i,j] = validF1

            #Reduce the learning rate according to ReduceLROnPlateau
            if args.lr_scheduler == 'plateau':
                lr_scheduler.step(metrics=validLoss)

            #Check for best validation F1 score so far seen, if so save the model
            if(validF1 > bestF1):
                bestF1 = validF1
                #Checkpoint for model saving
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'loss': validLoss,
                    'accuracy': validAccuracy,
                    'precision': validPrecision,
                    'recall': validRecall,
                    'f1': validF1,
                    'optimizer' : optimizer.state_dict(),
                }
                #Save the checkpoint and the model
                torch.save(checkpoint, (path+'/best-f1-checkpoint-%s.pth') % args.keyword)
                torch.save(model.state_dict(), (path+'/best-f1-model-%s.pth') % args.keyword)

            #Print the stats resulting from training and performing validation
            print("Epoch:", j, "--- Train Loss:", round(trainLoss,4), "--- Train Accuracy:", round(trainAccuracy,4), "--- Train Precision:", round(trainPrecision,4), "--- Train Recall:", round(trainRecall,4), "--- Train F1:", round(trainF1,4))
            print("Epoch:", j, "--- Valid Loss:", round(validLoss,4), "--- Valid Accuracy:", round(validAccuracy,4), "--- Valid Precision:", round(validPrecision,4), "--- Valid Recall:", round(validRecall,4), "--- Valid F1:", round(validF1,4))
            print()

    #Plot the train/validation loss, accuracy, precision, recall, and f1 score
    #versus the epoch using the error bar method for plotting to equalize across all runs
    plot(trainLossValues, validLossValues, 'loss')
    plot(trainAccuracyValues*100, validAccuracyValues*100, 'accuracy')
    plot(trainPrecisionValues, validPrecisionValues, 'precision')
    plot(trainRecallValues, validRecallValues, 'recall')
    plot(trainF1Values, validF1Values, 'f1')

    #Save the results from training/validation to file
    np.save(path+"/trainLoss", trainLossValues)
    np.save(path+"/trainAccuracy", trainAccuracyValues)
    np.save(path+"/trainPrecision", trainPrecisionValues)
    np.save(path+"/trainRecall", trainRecallValues)
    np.save(path+"/trainF1", trainF1Values)
    np.save(path+"/validLoss", validLossValues)
    np.save(path+"/validAccuracy", validAccuracyValues)
    np.save(path+"/validPrecision", validPrecisionValues)
    np.save(path+"/validRecall", validRecallValues)
    np.save(path+"/validF1", validF1Values)
