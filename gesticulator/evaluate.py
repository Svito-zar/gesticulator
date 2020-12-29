from os import makedirs
from os.path import isfile, join, isdir
from gesticulator.model.model import GesticulatorModel
from config.model_config import construct_model_config_parser
from pytorch_lightning import Trainer

def main(test_params):
    model = GesticulatorModel.load_from_checkpoint(
        test_params.model_file, inference_mode=True)

    model.eval()
    
    create_save_dirs(model)

    model.generate_evaluation_videos(
        semantic = test_params.use_semantic_input,
        random = test_params.use_random_input)

def create_save_dirs(model):
    save_dir = join(model.hparams.generated_gestures_dir, "evaluation")
    for output_type in ["raw_gestures", "3d_coordinates", "videos"]:
        output_dir = join(save_dir, output_type)
        if not isdir(output_dir):
            makedirs(output_dir)

def add_test_script_arguments(parser):
    parser.add_argument('--use_semantic_input', '-semantic', action="store_true",
                        help="If set, test the model with the semantic input segments")

    parser.add_argument('--use_random_input', '-random', action="store_true",
                        help="If set, test the model with the random input segments")
    
    parser.add_argument('--model_file', '-model', help="Path to the model checkpoint to be loaded")

    return parser

if __name__ == "__main__":
    """
    Usage: 
        python evaluate.py (--model_file MODEL_FILE | --run_name RUN_NAME ) [--use_semantic_input] [--use_random_input]
    """
    # Model parameters are added here
    parser = construct_model_config_parser()
    parser = add_test_script_arguments(parser)
    args = parser.parse_args()
    
    # If the model file is not provided, look for the trained checkpoint 
    # in the results/RUN_NAME directory
    if not args.model_file:
        last_ckpt = join(args.result_dir, args.run_name, "trained_model.ckpt")
        
        if isfile(last_ckpt):
            args.model_file = last_ckpt
        else:
            print("ERROR: Please provide the saved model checkpoint with:")
            print("         python test.py --model_file PATH_TO_CHECKPOINT\n")
            print("       Alternatively, you can provide the run_name argument you set during training:")
            print("         python test.py --run_name RUN_NAME")
            exit(-1)

    if not args.use_semantic_input and not args.use_random_input:
        print("ERROR: Please choose your evaluation type by providing the -random or -semantic flags!")    

    main(args)
