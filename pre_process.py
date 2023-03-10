"""
Purpose: This code can be used to pre-process the Google Speech Commands
         dataset and create a Wakeword detection dataset. This code works
         by creating a training, validation, and test set with data that
         falls into the keyword and non-keyword groups. The keyword is a
         command line arguement that can be used to choose any word within
         Google Speech Commands as the wakeword. Non-keywords are then
         created from the remaining classes as well as a number of silence
         examples. To create a proper training set a number of different
         transformations are applied to improve model generalization.
         Also since the dataset sizes are limited the copy parameter allows
         the provided keyword samples to each be copied this number of
         times with different transformations applied to each.
"""

#Libraries needed
import os
import argparse
import torch
import torchvision
from torchvision.transforms import *

#Custom files needed
from transforms import *


class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise.
    This class stores a numpy array of 1 second long
    utterances of background noise using the data contained
    within the folder input to this class
    """

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        #Get all the audio files in the folder and read in the data of each appending to samples
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        #Convert our list of background noise audio samples to a numpy array of 1 second long
        #utterances of background noise
        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c

        #Store the background noise, important classes, sample rate, transformation composition,
        #and path to the background noise folder in class variables
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = folder

    def __len__(self):
        """Returns the number of different 1 second long background
        noise utterances contained in this class
        """
        return len(self.samples)

    def __getitem__(self, index):
        """This returns a Python dictionary of the background noise utterance
        contained at the index location in self.samples and also performs the
        various transformations on it contained in self.transforms
        """
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}
        if self.transform is not None:
            data = self.transform(data)
        return data


"""Function to build the Mel Spectrogram data set by applying tranformations to the data
   and converting audio signals to their Mel Spectrogram representations
"""
def build_MFCC(data_type, dataset, transforms, keyword, copies, silence_percentage, out_folder):
    #Create the out folder if it does not already exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    #Create the correct path based on if we are dealing with training, validation, or test data
    path = None
    if(data_type == "train"):
        print("Pre-computing training data")
        path = os.path.join(out_folder, "train")
    elif(data_type == "valid"):
        print("Pre-computing validation data")
        path = os.path.join(out_folder, "valid")
    else:
        print("Pre-computing test data")
        path = os.path.join(out_folder, "test")

    #Create the training/validation/test folder within out folder based on what we are doing
    if not os.path.exists(path):
        os.makedirs(path)

    #Create the path to the keyword examples and create the folder if it doesn't already exist
    final_path = os.path.join(path, keyword)
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    #Loop through all examples of the keyword copies time, perform some transformations,
    #and save the Mel Spectrogram results to a folder
    total_keyword_files = 0
    for i in range(copies):
        for folder in os.listdir(dataset):
            if(folder == keyword):
                data_path = os.path.join(dataset, folder)
                for file in os.listdir(data_path):
                    data = {"path": data_path+"/"+file}
                    data = transforms(data)
                    torch.save(data['input'], os.path.join(final_path, file+"_"+str(i)+".ten"))
                    total_keyword_files += 1

    #Print the total number of keyword files saved
    print("Number of files saved to the keyword folder:", total_keyword_files)

    #Create the path to the non-keyword examples and create the folder if it doesn't already exist
    final_path = os.path.join(path, "non_keyword")
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    #Create silence examples at the silence percentage for the non keyword dataset
    silence_count = int(total_keyword_files * silence_percentage)
    total_non_keyword_files = 0
    while(total_non_keyword_files < silence_count):
        data = {"path": ''}
        data = transforms(data)
        torch.save(data['input'], os.path.join(final_path, "silence_"+str(total_non_keyword_files)+".ten"))
        total_non_keyword_files += 1

    #Print the number of silence non keyword files saved
    print("Number of silence files saved to the non keyword folder:", total_non_keyword_files)

    #Get the same number of non keyword transformed audio signals as was saved for the keyword itself
    while(total_non_keyword_files < total_keyword_files):
        #Get a random folder from the directory
        folder = random.choice(os.listdir(dataset))
        #Check that the folder is not our keyword folder, starts with a _ (what the background noise folder begins with),
        #or is some random file and not a folder
        if((folder != keyword) and (folder[0] != "_") and (folder != "LICENSE") and (folder != "README.md") and (folder != "testing_list.txt") and (folder != "validation_list.txt")):
            data_path = os.path.join(dataset, folder)
            #Get a random speech file from the folder
            file = random.choice(os.listdir(data_path))
            data = {"path": data_path+"/"+file}
            data = transforms(data)
            torch.save(data['input'], os.path.join(final_path, folder+"_"+file+".ten"))
            total_non_keyword_files += 1

    #Print the total number of non keyword files saved
    print("Number of files saved to the non keyword folder:", total_non_keyword_files)
    print()
    return


if __name__ == '__main__':
    #Parse the command line arguements
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
    parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
    parser.add_argument("--test-dataset", type=str, default='datasets/speech_commands/test', help='path of test dataset')
    parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
    parser.add_argument("--n-mels", type=int, default=32, help='Number of Mel filters when taking STFT')
    parser.add_argument("--keyword", type=str, default='marvin', help='Name of the wakeword being used')
    parser.add_argument("--copies", type=int, default=5, help='Number of copies to make of each true wakeword with different transformations applied')
    parser.add_argument("--MFCC-folder", type=str, default='MFCC_dataset', help='Name of the folder to store the new MFCC data set to')
    parser.add_argument("--silence-percentage", type=float, default=0.1, help='Percentage of non-keyword training/validation examples to be silence')
    args = parser.parse_args()

    #Create the transforms needed for the training files and load in the background noise data
    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(args.background_noise, data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=args.n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_transform = Compose([LoadAudio(), data_aug_transform, add_bg_noise, train_feature_transform])

    #Create the transforms needed for the validation files
    valid_feature_transform = Compose([ToMelSpectrogram(n_mels=args.n_mels), ToTensor('mel_spectrogram', 'input')])
    valid_transform = Compose([LoadAudio(), FixAudioLength(), valid_feature_transform])

    #Build the training data set
    build_MFCC("train", args.train_dataset, train_transform, args.keyword, args.copies, args.silence_percentage, args.MFCC_folder)

    #Build the validation data set
    build_MFCC("valid", args.valid_dataset, valid_transform, args.keyword, 1, args.silence_percentage, args.MFCC_folder)

    #Build the test data set
    build_MFCC("test", args.test_dataset, valid_transform, args.keyword, 1, args.silence_percentage, args.MFCC_folder)
