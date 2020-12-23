import os
import csv
from argparse import ArgumentParser

if __name__ == '__main__':
    #  Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--folder_downloaded', '-f_down', required=True,
                                   help="Path where the downloaded GENEA dataset is stored. Should end with `GENEA_Challenge_2020_data_release`.")
    parser.add_argument('--folder_renamed', '-f_renamed',  default="./raw_data/",
                                   help="Path where renamed dataset will be stored")

    params = parser.parse_args()

    folder_processed = params.folder_renamed

    print("Renaming and moving files ... ")

    # TRAIN
    # Go though the indices from the GENEA training set and rename files for those in the original dataset
    train_folder = params.folder_downloaded + "/Training_data/"
    with open('Data_renaming_indices.csv') as csvfile:
         csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
         headers = next(csv_reader, None)
         for row in csv_reader:
             x = row[0].split(",")
             genea_id = x[0]
             orig_id = x[1]

             print("Recording_" + str(genea_id).zfill(3) + " -> " + "NaturalTalking_" + str(orig_id).zfill(3))

             # rename audio files
             genea_file_name = "Recording_"+str(genea_id).zfill(3)+".wav"
             original_file_name = "NaturalTalking_"+str(orig_id).zfill(3)+".wav"
             audio_genea_folder = train_folder + "Audio/"
             audio_original_folder = folder_processed + "/Audio/"

             os.rename(os.path.join(audio_genea_folder, genea_file_name), os.path.join(audio_original_folder, original_file_name))

             # rename motion files
             genea_file_name = "Recording_"+str(genea_id).zfill(3)+".bvh"
             original_file_name = "NaturalTalking_"+str(orig_id).zfill(3)+".bvh"
             audio_genea_folder = train_folder + "/Motion/"
             audio_original_folder = folder_processed + "/Motion/"

             os.rename(os.path.join(audio_genea_folder, genea_file_name), os.path.join(audio_original_folder, original_file_name))

             # rename text files
             genea_file_name = "Recording_"+str(genea_id).zfill(3)+".json"
             original_file_name = "NaturalTalking_"+str(orig_id).zfill(3)+".json"
             audio_genea_folder = train_folder + "/Transcripts/"
             audio_original_folder = folder_processed + "/Transcripts/"

             os.rename(os.path.join(audio_genea_folder, genea_file_name), os.path.join(audio_original_folder, original_file_name))

    # TEST
    # Go though the indices from the GENEA test set and rename files for those in the original dataset
    test_folder =  params.folder_downloaded  + "/Test_data/"
    with open('Data_renaming_indices.csv') as csvfile:
         csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
         headers = next(csv_reader, None)
         for row in csv_reader:
             x = row[0].split(",")
             genea_id = x[2]
             orig_id = x[3]

             if genea_id == "":
                 continue

             print("TestSeq"+str(genea_id).zfill(3) + " -> " + "NaturalTalking_" + str(orig_id).zfill(3))

             # rename audio files
             genea_file_name = "TestSeq"+str(genea_id).zfill(3)+".wav"
             original_file_name = "NaturalTalking_"+str(orig_id).zfill(3)+".wav"
             audio_genea_folder = test_folder + "Audio/"
             audio_original_folder = folder_processed + "/Audio/"

             os.rename(os.path.join(audio_genea_folder, genea_file_name), os.path.join(audio_original_folder, original_file_name))

             # rename motion files
             genea_file_name = "TestSeq"+str(genea_id).zfill(3)+".bvh"
             original_file_name = "NaturalTalking_"+str(orig_id).zfill(3)+".bvh"
             audio_genea_folder = test_folder + "/Motion/"
             audio_original_folder = folder_processed + "/Motion/"

             os.rename(os.path.join(audio_genea_folder, genea_file_name), os.path.join(audio_original_folder, original_file_name))

             # rename text files
             genea_file_name = "TestSeq"+str(genea_id).zfill(3)+".json"
             original_file_name = "NaturalTalking_"+str(orig_id).zfill(3)+".json"
             audio_genea_folder = test_folder + "/Transcripts/"
             audio_original_folder = folder_processed + "/Transcripts/"

             os.rename(os.path.join(audio_genea_folder, genea_file_name), os.path.join(audio_original_folder, original_file_name))

    print("Done!")