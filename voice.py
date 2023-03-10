#-*- coding: utf-8 -*-
"""
Referenced
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
for quantization

"""
import os
import argparse
import pyaudio
from queue import Queue
from threading import Thread
import sys
import time
import numpy as np
import numpy as np
import site
import torch
import torchvision
import librosa
import time
import models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import *
from transforms import *
from utils import *

from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig as get_default_qconfig_fx
from torch.quantization import convert, prepare
from torch.quantization import get_default_qconfig as get_default_qconfig_eager


#logging related 
import pandas as pd
from pthflops import count_ops
from thop import profile

#debugging
import pdb
import copy


#Select device
device = None
if torch.cuda.is_available():
    print("Using CUDA GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS GPU")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")

#Load model and statedict
def getModel(filename):
    stateDict = torch.load(filename, map_location=device)
    pathComponents = filename.split("/")
    modelName = pathComponents[-2]
    print(modelName)
    keyword = ((pathComponents[-1].split("-"))[-1].split("."))[-2]
    model = models.create_model(model_name=modelName, num_classes=2, in_channels=1)
    return stateDict, model, modelName

#Manually define the path to the model
filename = "checkpoints/keyword/modelName/best-f1-model-keyword.pth"
stateDict, rawModel, modelName, keyword = getModel(filename)

#If the model was saved as a stateDict, load it. Otherwise, load its checkpoint file
try:
    pathComponents = filename.split("/")
    pathComponents[3] = pathComponents[3].replace("model", "checkpoint")
    checkpoint = torch.load("/".join(pathComponents), map_location=device)
    rawModel.load_state_dict(stateDict)
except:
    stateDict = checkpoint['state_dict']
    rawModel.load_state_dict(stateDict)
    
#It is necessary to send the model to the device when using MPS
if (device == "cuda" or device == "mps"):
    rawModel.to(device)



rawModel.eval()

# Uncomment for quantization
#m = copy.deepcopy(rawModel)
#m.eval()

#For mobilenet only. Uncomment for quantization
#if (modelName == "mobilenet_v2"):
#    m.fuse_model();


def quantize(rawModel):
    testWakeword = AudioDataset("MFCC_dataset/test", keyword)
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
    
    #Optional calibration
    #print("Calibrating for quantization")
    #calibrate(prepared_model, testDataloader)

    model = convert(prepared_model)
    
    return model

#Uncomment next line + comment out following for quantization
#model = quantize(m)
model = rawModel


###Hooks###

#Content to be logged to a file. Should contain 2-tuples.
log_list = []


###Hook using DFS - uncomment if you want to use it###
#Global lists to be accessed only from hooks
#startTimes = []
#hasHooks = [False]
#index = [-1]


#def getStartTime(m, i):
#    """
#    Function for forward pre-hook which logs the time before running a layer
#
#    Argument:
#    m -- layer of a model
#    i -- input to the layer
#    """
#    #Update the location of most recent start time
#    index[0] += 1
#
#    startTimes.append((m, time.time()))
#
#
#def getEndTime(m, i, o):
#    """
#    Function for forward hook which logs the time after a layer has been run

#    Argument:
#    m -- layer of a model
#    i -- input to the layer
#    o -- output of the layer
#    """
#    currentTime = time.time()
#    assert(m == startTimes[index[0]][0])
#
#    #Add the layer's name and inference time to log_list
#    log_list.append((m, currentTime - startTimes[index[0]][1]))
#
#
##Depth-first search through a model's graph, adding forward pre-hooks
## and forward hooks to 'leaf node layers'
#def insertHookHelper(layer):
#    """
#    Function which does a depth-first search through a model's graph, adding forward
#    pre-hooks and forward hooks to 'leaf node layers'
#
#    Argument:
#    layer -- layer of a model
#    
#    """
#    #print(layer)
#    #'leaf node layers' are tuples of (layer name, layer)
#    if (isinstance(layer, type((1,2)))):
#        layer = layer[1]
#
#    children = list(layer.named_children())
#
#    if (children == []):
#        #Add hooks
#        layer.register_forward_pre_hook(getStartTime)
#        layer.register_forward_hook(getEndTime)
#    else:
#        for child in children:
#            insertHookHelper(child)
#
#
#def insertHook(m, i):
#    """
#    Function for a model's forward pre-hook which adds timing hooks to a model's
#    'leaf node layers'
#
#    Argument:
#    m -- a model
#    i -- the model's input
#
#    """
#    if (not hasHooks[0]):
#        insertHookHelper(m)
#        hasHooks[0] = True
#

#Adds forward pre-hook for timing. Comment the below line out to
# disable timing.
#model.register_forward_pre_hook(insertHook)


###Hook using model-level timers###

def getTime(m, i, o):
    """
    Function for forward hook which gets a model's runtime information for the most recent
    forward pass and logs it

    Argument:
    m -- layer of a model
    i -- input to the layer
    o -- output of the layer
    """
    model_name = m.name
    for j in range(0, len(m.names)):
        log_list.append([m.names[j],m.times[j]])

#Adds forward pre-hook for timing. As timing is implemented manually in this version of the code,
# uncommenting will not disable timing.
model.register_forward_hook(getTime)



###Audio Processing Setup###

chunk_duration = 1 #Each read length in seconds from mic.
fs = 16000 #Sampling rate to process audio
micIndex = 1 #Audio device index for PyAudio, change if channel errors
mic_sample_rate = (pyaudio.PyAudio().get_device_info_by_index(micIndex))['defaultSampleRate'] #Default mic sample rate
chunk_samples = int(fs * chunk_duration) #Each read length in number of samples

#Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 1
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

#Starting point in audio data queue for oldest chunk whose data you want to pass to the model
chunkFrames = int(feed_duration * fs - chunk_duration * fs) 

#Queue to communicate between the audio callback and main thread
q = Queue()

run = True

#Audio input volume threshold for audio to be sent to the model
silence_threshold = 350

#Run the demo for a timeout seconds
timeout = time.time() + 20  # 0.1 minutes from now

#Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')


def detect_triggerword_spectrum(inpt, timeTotal):
    """
    Function to predict the location of the trigger word.

    Argument:
    inpt -- spectrum of shape (freqs, Tx)
    i.e. (Number of frequencies, The number time steps)
    timeTotal -- total time spent running the model during demo

    Returns:
    predictions -- flattened numpy array to shape (number of output time steps)
    timeTotal -- total time spent running the model during demo
    """
    #The spectogram outputs, and we want (Tx, freqs) to input into the model
    
    #Initialize a torch tensor matrix with the input data
    inpt = inpt.detach().clone()
    
    #Add two dimensions to the array - becomes nested two more layers
    inpt = inpt[np.newaxis,np.newaxis,:,:]

    #Divides the number of rank 3 values by number of rank 2 values in
    # the tensor array
    bsize = int(inpt.shape[3]/inpt.shape[2])
    
    #Cap the number of values in dimension 4 to bsize*32
    inpt = inpt[:,:,:,:bsize*32]

    #Rearrange the array
    inpt = inpt.reshape((int(bsize),1,32,32))

    #Get start for calculating the model's overall inference time
    startTime = time.time()

    #Run input through model
    out = model(inpt.to(device))

    #Calculate the model's inference time, and add it to timeTotal
    # which will be used to calculate average inference time
    inferenceTime = time.time() - startTime
    timeTotal += inferenceTime

    #Uncomment to get macs and parameters using thop
    #macs, params = profile(model, inputs=(inpt.to(device), ))
    #Uncomment to print out flops (approximated by doubling macs)
    #print("Flops: " + str(macs * 2))

    #Get model's prediction for whether the wakeword has been detected
    # 0 - negative, 1 - positive
    prediction = np.argmax(out.cpu().detach().numpy(),axis=1)
    return prediction[0], timeTotal
    

def get_audio_input_stream(callback):
    """
    Function opening a pyaudio audio input stream

    Argument:
    callback -- the callback function to be used

    Returns:
    stream -- a pyaudio audioinput stream
    """
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=micIndex,
        stream_callback=callback)
    return stream


def callback(in_data, frame_count, time_info, status):
    """
    Callback function for the audio stream. Communicates between 'publisher' and 'subscriber'
        to process audio input in manageable chunks

    Argument:
    in_data -- audio data buffer
    frame_count -- sampling rate
    time_info -- TimeInfo struct with timing information for the buffers, in seconds
    status -- callback status flags 

    Returns:
    (in_data, pyaudio.paContinue) -- a pyaudio audioinput stream
    """
    global run, timeout, data, silence_threshold
    #Close the stream if timeout has been reached
    if time.time() > timeout:
        run = False

    #Convert the data buffer 'in_data' to array form
    data0 = np.frombuffer(in_data, dtype=np.int16)
    #If the chunk is determined to be silence, write - and keep going
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    else:
        #Sound detected, indicate this
        sys.stdout.write('.')

    #Add this chunk's sound data to overall sound data
    data = np.append(data,data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)


###Stream###

#Start the stream
stream = get_audio_input_stream(callback)
stream.start_stream()

#Define variables for averaging model run time across chunks
chunkCount = 0
timeTotal = 0

#Stream audio processing
try:
    print("START")
    while run:
        #Initialize the queue
        data = q.get()

        chunkCount += 1

        #Process the chunk's input data
        voice_feature_transform = Compose([ToMelSpectrogramRT(n_mels=32, sample_rate=fs), ToTensorRT()])
        voice_transform = Compose([LoadAudioRT(sample_rate=fs), FixAudioLengthRT(), voice_feature_transform])

        #Process the last chunk's input data
        spectrum = voice_transform(data[chunkFrames:])

        #Get predictions from the model
        preds, timeTotal = detect_triggerword_spectrum(spectrum, timeTotal)
        
        if preds > 0:
            print('1')
        else:
            print('0')

except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False


print("Average model inference time: " + str(timeTotal/chunkCount))

stream.stop_stream()
stream.close()


###Logging###

#Uncomment this section to turn off layer time logging

#By default, logs will be saved with the filename format
# modelName_runID_Times.csv. This block creates/references a file to
# store the current runID, ensuring nothing is overwritten
try:
    runIDFile = open("IDFile.txt", "r")
    runID = int(runIDFile.read()) + 1
    runIDFile.close()
    runIDFile = open("IDFile.txt", "w")
    runIDFile.write(str(runID))
    runIDFile.close()
except:
    runID = 1
    runIDFile = open("IDFile.txt", "w")
    runIDFile.write(str(runID))
    runIDFile.close()

#Write log into file 
df = pd.DataFrame(log_list, columns =['Name', 'Time'])
df.to_csv(modelName + "_" + str(runID) + "_" + "Times.csv")

print("RunID: " + str(runID))
print("Done")

