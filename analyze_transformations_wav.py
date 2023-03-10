"""
Purpose: This code reads in wav file and performs a number
         of transformations while saving the results
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


"""Function to save the audio and mel spectrogram after some transformation"""
def save(audio, folder, filename, Mel_Spec):
    save_wav(audio, args.folder, filename+"_audio"+".wav")
    audio_mfcc = Mel_Spec(audio)
    save_mel_spec(audio_mfcc, args.folder, filename+"_mel")
    return


"""Function to display and save the mel spectrogram from wav audio data"""
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
def save_wav(data, folder, filename):
    #Create the directory if it does not exist already
    if not os.path.exists(folder):
        os.makedirs(folder)

    #Save the audio
    path = os.path.join(folder, filename)
    #Edited
    soundfile.write(path, data['samples'], data['sample_rate'])
    return


"""Function to multiply the amplitude of audio by some factor"""
def change_amplitude(audio, folder, Mel_Spec):
    AmplitudeModifier = ChangeAmplitude(amplitude_range=(0.5, 0.5), prob=1.0)
    audio = AmplitudeModifier(audio)
    save(audio, folder, "ampliture_mod", Mel_Spec)
    return


"""Function to speed up the audio by some factor"""
def change_speed_pitch(audio, folder, Mel_Spec):
    SpeedPitchModifier = ChangeSpeedAndPitchAudio(max_scale=0.2, prob=1.0)
    audio = SpeedPitchModifier(audio)
    save(audio, folder, "speed_pitch_mod", Mel_Spec)
    return


"""Function to stretch the audio by some factor"""
def stretch_audio(audio, folder, Mel_Spec):
    Stretcher = StretchAudio(max_scale=0.2, prob=1.0)
    audio = Stretcher(audio)
    save(audio, folder, "stretch", Mel_Spec)
    return


"""Function to timeshift the audio by some factor"""
def timeshift(audio, folder, Mel_Spec):
    Timeshift = TimeshiftAudio(max_shift_seconds=0.2, prob=1.0)
    audio = Timeshift(audio)
    save(audio, folder, "timeshift", Mel_Spec)
    return


"""Function to add background noise to the audio at some percentage"""
def add_background_noise(audio, folder, Mel_Spec, background_noise):
    BackgroundNoise = AddBackgroundNoise(bg_dataset=[background_noise], max_percentage=0.45, prob=1.0)
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

    #Classes to load the audio, fix the length, and convert to Mel spectrogram
    Loader = LoadAudio()
    FixLength = FixAudioLength()
    Mel_Spec = ToMelSpectrogram(n_mels=args.mels)

    #Load the audio sample, save it, and compute the original mel spectrogram
    data = {"path": args.filename}
    audio = FixLength(Loader(data))
    save(copy.copy(audio), args.folder, "original", Mel_Spec)

    #Iterate through the various transformations by applying them
    #and saving their results both as wav and mel spectrogram plots
    change_amplitude(copy.copy(audio), args.folder, Mel_Spec)
    change_speed_pitch(copy.copy(audio), args.folder, Mel_Spec)
    stretch_audio(copy.copy(audio), args.folder, Mel_Spec)
    timeshift(copy.copy(audio), args.folder, Mel_Spec)

    #Load the background noise sample, save it, and compute the mel spectrogram
    noise_data = {"path": args.background_noise}
    noise = FixLength(Loader(noise_data))
    save(copy.copy(noise), args.folder, "noise_original", Mel_Spec)

    #Add background noise to the sample
    add_background_noise(copy.copy(audio), args.folder, Mel_Spec, copy.copy(noise))
