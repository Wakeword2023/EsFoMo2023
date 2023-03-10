"""
Purpose: This code reads in wav file, computes the STFT,
         and performs a number of transformations while
         saving the results
"""

#Libraries needed
import argparse
import matplotlib.pyplot as plt
import librosa.display
import os
import copy
import soundfile

#Local files needed
from transforms import *


"""Function perform the inverse STFT and save the audio and mel spectrogram after some transformation"""
def save(data, folder, filename, Mel_Spec):
    audio = librosa.istft(data['stft'], hop_length=data['hop_length'])
    save_wav(audio, data['sample_rate'], args.folder, filename+"_audio"+".wav")
    audio_mfcc = Mel_Spec(data)
    save_mel_spec(audio_mfcc, args.folder, filename+"_mel")
    return


"""Function to display and save the mel spectrogram from stft audio data"""
def save_mel_spec(data, folder, plot_name):
    #Create the plot and plot it
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data['mel_spectrogram'], x_axis='time', y_axis='mel', sr=data['sample_rate'], fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    #Create the directory if it does not exist already
    if not os.path.exists(folder):
        os.makedirs(folder)

    #Save the figure
    path = os.path.join(folder, plot_name)
    plt.savefig(fname=path)
    plt.show()
    return


"""Function to save the transformed wav audio file"""
def save_wav(data, sample_rate, folder, filename):
    #Create the directory if it does not exist already
    if not os.path.exists(folder):
        os.makedirs(folder)

    #Save the audio
    path = os.path.join(folder, filename)
    soundfile.write(path, data, sample_rate)
    return


"""Function to stretch the audio along the STFT"""
def stretch_on_stft(audio, FixSTFT, folder, Mel_Spec):
    Stretcher = StretchAudioOnSTFT(max_scale=0.2, prob=1.0)
    audio = FixSTFT(Stretcher(audio))
    save(audio, folder, "stretch", Mel_Spec)
    return


"""Function to timeshift the audio in STFT"""
def timeshift_on_stft(audio, FixSTFT, folder, Mel_Spec):
    Timeshift = TimeshiftAudioOnSTFT(max_shift=8, prob=1.0)
    audio = FixSTFT(Timeshift(audio))
    save(audio, folder, "timeshift", Mel_Spec)
    return


"""Function to add background noise to the audio in the STFT"""
def add_background_noise_on_stft(audio, folder, Mel_Spec, background_noise):
    BackgroundNoise = AddBackgroundNoiseOnSTFT(bg_dataset=[background_noise], max_percentage=0.45, prob=1.0)
    audio = BackgroundNoise(audio)
    save(audio, folder, "background_noise", Mel_Spec)
    return


#Main function
if __name__ == "__main__":
    #Get the name of the file we want to perform the transformations on, the background noise file,
    #folder to write results to, and number of mels to compute for
    parser = argparse.ArgumentParser(description='Process command line arguements')
    parser.add_argument('--filename', required=True, help="Name of the file to perform the transformations on")
    parser.add_argument('--mels', default=40, help="The number of mels to compute the MFCC on")
    parser.add_argument('--folder', required=True, help="Name of the folder to save the results to")
    parser.add_argument("--background-noise", type=str, required=True, help='Name of the file with background noise')
    args = parser.parse_args()

    #Classes to load the audio, fix the length, compute STFT,
    #convert to the Mel spectrogram, and fix the STFT dimension
    Loader = LoadAudio()
    FixLength = FixAudioLength()
    STFT = ToSTFT()
    Mel_Spec = ToMelSpectrogramFromSTFT(n_mels=args.mels)
    FixSTFT = FixSTFTDimension()

    #Load the audio sample, compute the STFT, and use the inverse STFT to re-derive the original
    #audio while computing the Mel spectrogram
    data = {"path": args.filename}
    audio = STFT(FixLength(Loader(data)))
    save(copy.copy(audio), args.folder, "original", Mel_Spec)

    #Iterate through the various transformations by applying them
    #and saving their results both as wav and mel spectrogram plots
    stretch_on_stft(copy.copy(audio), FixSTFT, args.folder, Mel_Spec)
    timeshift_on_stft(copy.copy(audio), FixSTFT, args.folder, Mel_Spec)

    #Load the background noise sample, compute the STFT, and use the inverse STFT
    #to re-derive the audio while computing the Mel spectrogram
    noise_data = {"path": args.background_noise}
    noise = STFT(FixLength(Loader(noise_data)))
    save(copy.copy(noise), args.folder, "noise_original", Mel_Spec)

    #Add background noise to the STFT
    add_background_noise_on_stft(copy.copy(audio), args.folder, Mel_Spec, copy.copy(noise))
