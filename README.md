# Experiment Demos
https://www.youtube.com/watch?v=mPDiCb8Z-2k

# PyTorch Wake-Word Detection with Model Distillation and Live Demo
## Project Overview

## Installing Dependencies
To create a conda virtual environment for the project:
* $conda create -n WakeWord pip

To activate the environment:
* $conda activate Wakeword

The following commands install the proper dependencies:
* $conda install pytorch=0.4.1 cuda90 -c pytorch (assumes CUDA 9.0) (https://pytorch.org/)
* $pip install torchvision (https://pytorch.org/docs/stable/torchvision/index.html) (https://pypi.org/project/torchvision/)
* $pip install librosa (https://librosa.github.io/librosa/install.html)
* $pip install matplotlib (https://pypi.org/project/matplotlib/)
* $pip install pip install scikit-image (https://scikit-image.org/)

## To Download the Google Speech Commands Dataset:
Run the following:
* $chmod 777 download_speech_commands_dataset.sh
* $./download_speech_commands_dataset.sh

About the Google Speech Commands Dataset: (https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)

## To Analyze Various WAV File Transformations:
Run the following command:
* $python analyze_transformations_wav.py --filename wav_audio_file --folder results_folder --background-noise wav_noise_file

This command reads in the original wav audio file and performs various transformations on it both saving the result as another wav file and computing the Mel Spectogram and saving it:
* Saves the original
* Multiplies the Amplitude by a factor of 0.5
* Changes the Speed and Pitch by a factor between 0.833 and 1.25
* Stretches the audio by a scale between -0.2 and 0.2
* Time-shifts the audio between -0.2 seconds and 0.2 seconds
* Saves the background noise
* Adds background noise to the original audio at some percentage between 0.00 and 0.45

The results of applying these transformations can be seen in the **wav_transforms_results** folder. These results were computed using a recording of the word **Marvin** as well as generic **White Noise**. For transformation values see **Notes.txt**.

## To Analyze Various STFT Transformations:
Run the following command:
* $python analyze_transformations_stft.py --filename wav_audio_file --folder results_folder --background-noise wav_noise_file

This command reads in the original wav audio file, computes the STFT, and performs various transformations on it both saving the result as another wav file and computing the Mel Spectogram and saving it:
* Saves the original
* Stretches the STFT by a scale between -0.2 and 0.2
* Time-shifts the STFT between -8 and 8 frames
* Saves the background noise
* Adds background noise to the original audio at some percentage between 0.00 and 0.45

The results of applying these transformations can be seen in the **stft_transforms_results** folder. These results were computed using a recording of the word **Marvin** as well as generic **White Noise**. For transformation values see **Notes.txt**

## To Pre-Process the Data and Create Wakeword Training, Validation and Test Sets
To speed up training we pre-process the dataset by generating **Mel Spectrograms** based on a number of different transformations applied to both the original audio signal as well as the Short Time Fourier Transform (STFT) of it. This also allows us to go ahead and create **Keyword** and **Non-Keyword** datasets from the original Google Speech Commands data. Here you can choose a specific word from Google Speech Commands to be the wakeword to train on; all other samples and silence are then grouped into the Non-Wakeword bucket. To run this command, pre-process the data, and create Mel Spectrograms to train, run validation, and test over use the following:
* $python pre_process.py

Note that this command comes with a number of command-line arguements that can make pre-processing more customizable:
* train-dataset: Path to the training dataset (default: datasets/speech_commands/train)
* valid-dataset: Path to the validation dataset (default: datasets/speech_commands/valid)
* test-dataset: Path to the test dataset (default: datasets/speech_commands/test)
* background-noise: Path to the background noise data (default: datasets/speech_commands/train/_background_noise_)
* n-mels: Number of Mel filters when taking the STFT (default: 32)
* keyword: Word to use as the Wakeword from Google Speech Commands (default: 'marvin')
* copies: Number of copies to make of each true Wakeword sample from the training set with different random transformations applied to each (default: 5)
* MFCC-folder: Name of the folder to save the new Mel spectrogram Wakeword / Non-Wakeword training and validation sets to (default: MFCC_dataset)
* silence-percentage: Percentage of Non-Wakeword training/validation examples to be silence (default: 0.1 = 10%)

Note that the pre-computed Mel spectrogram Wakeword / Non-Wakeword training, validation, and test sets will be saved to the folder specified by the **MFCC-folder** arguement.

## To Train a Model and Perform Validation across the Pre-Computed Training/Validation Sets
To train a model on the pre-computed dataset run the following command:
* $python train.py

Note that this command comes with the following command-line arguements to make training more customizable:
* train-dataset: Path to the pre-computed Mel spectrogram train dataset (default: MFCC_dataset/train)
* valid-dataset: Path to the pre-computed Mel spectrogram validation dataset (default: MFCC_dataset/valid)
* keyword: Word to use as the Wakeword from Google Speech Commands (default: 'marvin')
* batch-size: Size of a batch for training/validation (default: 64)
* epochs: Number of epochs to train for (default: 75)
* runs: Number of total training runs to perform (default: 5)
* dataload-workers-nums: Number of workers for dataloader (default: 4)
* weight-decay: Weight decay value of model parameters (default: 1e-2)
* optim: Model optimizer to use (default: 'sgd')
* learning-rate: Learning rate for optimization (default: 1e-4)
* lr-scheduler: Method to adjust learning rate (default: 'plateau')
* lr-scheduler-patience: Number of epochs with no improvement after which learning rate will be reduced for plateau scheduler (default: 5)
* lr-scheduler-gamma: Learning rate multiplier for reduction (default: 0.5)
* model: Neural network model architecture (default: 'vgg19_bn')

In regard to the **optim** command the following options are avaliable:
* 'sgd': Stochastic gradient descent optimizer
* 'adam': Adam optimizer

 For the **lr-scheduler** command there are two options:
 * 'plateau': Learning rate is decreased on plateaus
 * 'step': Learning rate is decreased every so many epochs

The following CNN model architectures are avaliable for training. Note that the MnasNet and SqueezeNet variants fail to perform well on this data:
* vgg19_bn (https://arxiv.org/pdf/1409.1556.pdf)
* resnet18 (https://arxiv.org/abs/1512.03385)
* resnet34 (See Resnet Paper Above)
* resnet50 (See Resnet Paper Above)
* resnet101 (See Resnet Paper Above)
* resnet152 (See Resnet Paper Above)
* densenet_bc_100_12 (https://arxiv.org/abs/1608.06993)
* densenet_bc_250_24 (See DenseNet Paper Above)
* densenet_bc_190_40 (See DenseNet Paper Above)
* mobilenet_v2 (https://arxiv.org/abs/1704.04861)
* squeezenet1_0 (https://arxiv.org/abs/1602.07360)
* squeezenet1_1 (See SqueezeNet Paper Above)
* mnasnet0_5 (https://arxiv.org/abs/1807.11626)
* mnasnet0_75 (See MnasNet Paper Above)
* mnasnet1_0 (See MnasNet Paper Above Above)
* mnasnet1_3 (See MnasNet Paper Above Above)
* shufflenet_v2_x0_5 (https://arxiv.org/abs/1807.1116 Above4)
* shufflenet_v2_x1_0 (See ShuffleNet Paper Above Above)
* shufflenet_v2_x1_5 (See ShuffleNet Paper Above)
* shufflenet_v2_x2_0 (See ShuffleNet Paper Above Above)
* custom_shufflenet (See ShuffleNet Paper Above Above)
* efficientnet_b0 (https://arxiv.org/abs/1905.11946 Above)
* efficientnet_b1 (See EfficientNet Paper Above)
* efficientnet_b2 (See EfficientNet Paper Above)
* efficientnet_b3 (See EfficientNet Paper Above Above)
* efficientnet_b4 (See EfficientNet Paper Above)
* efficientnet_b5 (See EfficientNet Paper Above Above)
* efficientnet_b6 (See EfficientNet Paper Abov Above Abovee)
* efficientnet_b7 (See EfficientNet Paper Above)
* efficientnet_custom (See EfficientNet Paper Above)

## To Test a Trained Model across the Pre-Computed Test Sets
To test a trained model on the pre-computed dataset run the following command:
* $python test.py --model model_name --model-path path_to_saved_model --checkpoint-path path_to_checkpoint

Note that this command comes with the following command-line arguements to make training more customizable:
* test-dataset: Path to the pre-computed Mel spectrogram test dataset (default: MFCC_dataset/test)
* keyword: Word to use as the Wakeword from Google Speech Commands (default: 'marvin')
* batch-size: Size of a batch for training/validation (default: 1)
* dataload-workers-nums: Number of workers for dataloader (default: 1)
* model: Neural network model architecture (default: 'vgg19_bn')
* model-path: Path to the saved model (required) (default: None)
* checkpoint-path: Path to the saved checkpoint' (required) (default: None)

This command tests the model and reports results on the validation set from the checkpoint as well as results from testing. This also returns the total number of parameters in the model and the average inference time per sample. The following model statistics are reported in regard to both the validation and test sets:
* Cross-Entropy Loss
* Accuracy
* Precision
* Recall
* F1 Score

## To Visualize the First Layer Learned Filters of a Model
To visualize the first layer learned filters of a model run the following command:
* $python visualize_filter.py --model model_name --model-path path_to_saved_model

This command will compute the mean across all filters at each location. For each filter if the filter index value is less than the mean then we set this filter index to 0 for visualization so that a clearer picture can be made over what filters correspond to what portions of the audio recognition. All filters are then visualized using Matplotlib.

## To Compare the Results from Various Models
To compare the training / validation results from various models run the following command:
* $python model_comparison.py --model-folder path_to_model_folder --models model_name_1 model_name_2 model_name_3 ...

Running this command will visualize the following from performing training / validation of the models listed:
* Training Loss vs Epoch  
* Training Accuracy vs Epoch  
* Training Precision vs Epoch  
* Training Recall vs Epoch  
* Training F1 vs Epoch  
* Validation Loss vs Epoch  
* Validation Accuracy vs Epoch  
* Validation Precision vs Epoch  
* Validation Recall vs Epoch  
* Validation F1 vs Epoch   
=======
