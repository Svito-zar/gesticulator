from os.path import isfile, join
from gesticulator.model.model import GesticulatorModel
from config.model_config import construct_model_config_parser
from pytorch_lightning import Trainer

def main(test_params):
    model = load_model(test_params)
    trainer = Trainer.from_argparse_args(test_params, logger=False)

    trainer.test(model)

def load_model(test_params):
    """This function enables the test datasets that were selected by the user"""
    model = GesticulatorModel.load_from_checkpoint(test_params.model_file, inference_mode=True)
    
    # Make sure that at least one of the two test datasets are enabled
    if not test_params.test_semantic and not test_params.test_random:
        print("ERROR: Please provide at least one of the following two flags:")
        print("       python test.py --test_semantic (to use the semantic test input segments)")
        print("       python test.py --test_random (to use the random test input segments)")
        exit(-1)

    model.hparams.generate_semantic_test_predictions = test_params.test_semantic
    model.hparams.generate_random_test_predictions = test_params.test_random
    
    return model

def add_test_script_arguments(parser):
    parser.add_argument('--test_semantic', '-semantic', action="store_true",
                        help="If set, test the model with the semantic input segments")

    parser.add_argument('--test_random', '-random', action="store_true",
                        help="If set, test the model with the random input segments")
    
    parser.add_argument('--model_file', '-model', help="Path to the model checkpoint to be loaded")

    return parser

if __name__ == "__main__":
    """
    Usage: 
        python test.py (--model_file MODEL_FILE | --run_name RUN_NAME ) [--test_semantic] [--test_random]
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
    
    main(args)