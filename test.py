
#Libraies needed
import argparse
import torch
from torch.utils.data import DataLoader
import copy
from torch.quantization import convert, prepare
from torch.quantization import get_default_qconfig as get_default_qconfig_eager

#Custom files needed
import models
from utils import *


#Main function
if __name__ == '__main__':
    #Parse the command line arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test-dataset", type=str, default='MFCC_dataset/test', help='path of pre-computed Mel spectrogram test dataset')
    parser.add_argument("--keyword", type=str, default='marvin', help='Name of the wakeword being used (must be the same as used in pre_process.py)')
    parser.add_argument("--batch-size", type=int, default=1, help='Size of a batch for training')
    parser.add_argument("--dataload-workers-nums", type=int, default=1, help='number of workers for dataloader')
    parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of NN')
    parser.add_argument("--model-path", type=str, required='True', help='path to the saved model')
    parser.add_argument("--checkpoint-path", type=str, required='True', help='path to the saved checkpoint')
    args = parser.parse_args()

    #Create the test dataset as well as its dataloader
    testWakeword = AudioDataset(args.test_dataset, args.keyword)
    testDataloader = DataLoader(dataset=testWakeword, batch_size=args.batch_size, shuffle=False, num_workers=args.dataload_workers_nums)

    #Get the GPU if avaliable for fast testing
    GPU = None
    device = None
    if torch.cuda.is_available():
        print("Using GPU")
        GPU = torch.device("cuda")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS GPU")
        GPU = torch.device("mps")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    def getModel(filename):
        stateDict = torch.load(filename, map_location=device)
        model = models.create_model(model_name=args.model, num_classes=2, in_channels=1)
        return stateDict, model

    #Initialize the model and criterion, load the model and checkpoint
    stateDict, rawModel = getModel(args.model_path)
    #If the model is stored as a stateDict, load that. Otherwise load the checkpoint stateDict
    try:
       rawModel.load_state_dict(stateDict)
    except:
        exceptCheckpoint = torch.load(args.checkpoint_path, map_location=device)
        stateDict = exceptCheckpoint['state_dict']
        rawModel.load_state_dict(stateDict)
        
    m = copy.deepcopy(rawModel)
    m.eval()
    
    #For mobilenet only
    if (args.model == "mobilenet_v2"):
        m.fuse_model()
    
    def quantize(rawModel):
        testWakeword = AudioDataset("MFCC_dataset/test", "sheila")
        testDataloader = DataLoader(dataset=testWakeword, batch_size=1, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()

        #fbgemm for x86, qnnpack for arm
        torch.backends.quantized.engine = 'qnnpack'

        rawModel.qconfig = get_default_qconfig_eager("qnnpack")
        
        def calibrate(model_to_calibrate, data_loader):
            model_to_calibrate.eval()
            with torch.no_grad():
                for audioFile, target in data_loader:
                    model_to_calibrate(audioFile)

        prepared_model = prepare(
            rawModel,
            inplace = True
        )

        batchLimit = 10
        printInfo = False
        
        #Add calibration
        print("Calibrating for quantization")
        evaluate(prepared_model, criterion, testDataloader, None, batchLimit, printInfo)

        model = convert(prepared_model)
        
        return model
    
    #model = quantize(m)

    criterion = torch.nn.CrossEntropyLoss()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    #Variable to store the total number of parameters in our model
    parameter_count = 0
    #Loop through all layers/blocks getting the number of parameters in each
    for parameter in model.parameters():
        #Variable to store the number of parameters per layer, get the layer-wise parameters
        layer_parameter_count = 0
        layer_parameters = parameter.size()
        #Calculate the total number of parameters in layer/block based on the layer-wise parameters
        for i in layer_parameters:
            if(layer_parameter_count == 0):
                layer_parameter_count += i
            else:
                layer_parameter_count *= i
        parameter_count += layer_parameter_count


    #Print the best/saved model stats from validation
    print("Model results (best model):")
    print("Valid Loss:", round(checkpoint['loss'],4), "--- Valid Accuracy:", round(checkpoint['accuracy'],4), "--- Valid Precision:", round(checkpoint['precision'],4),
          "--- Valid Recall:", round(checkpoint['recall'],4), "--- Valid F1:", round(checkpoint['f1'],4))

    #Send the model and criterion to the device
    model = model.to(device)
    criterion = criterion.to(device)

    batchLimit = None
    printInfo = True
    
    #Perform testing
    testLoss, testAccuracy, testPrecision, testRecall, testF1, inferenceTime = evaluate(model, criterion, testDataloader, GPU, batchLimit, printInfo)

    #Print the stats resulting from testing
    print(" Test Loss:", round(testLoss,4), "---  Test Accuracy:", round(testAccuracy,4), "---  Test Precision:", round(testPrecision,4), "---  Test Recall:", round(testRecall,4), "---  Test F1:", round(testF1,4))
    print("Total Model Parameters:", parameter_count)
    print("Average Inference Time:", inferenceTime)
