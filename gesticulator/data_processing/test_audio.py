from tools import *
audio_filename = "/home/taras/Documents/Datasets/SpeechToMotion/Irish/test_audio/NaturalTalking_21.wav" #result21.wav"
input_vectors = calculate_mfcc(audio_filename)
print(input_vectors)