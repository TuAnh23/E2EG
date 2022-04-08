"""
Used in hyperparams sweep with wandb.
Collect the fixed-parsed-arguments to the training scripts (which do not need sweeping over), save them to a json file
to use later in the YAML file.
"""
from pecos.xmc.xtransformer.train import parse_arguments
import json


if __name__ == "__main__":
    parser = parse_arguments()

    # Additional argument to specify where to store to json file.
    # This argument should not be included in the json file
    parser.add_argument(
        "--save_json_path",
        type=str,
        metavar="PATH",
        help="Path to save the json file."
    )

    # Additional argument to specify the arguments being swept over
    # These arguments should not be included in the json file
    parser.add_argument(
        "--swept_args",
        type=str,
        help="Arguments being swept over (passes in as a string, seperated by |). "
             "Will be excluded in the json file."
    )

    args, unknown = parser.parse_known_args()
    args_dict = vars(args)

    save_path = args_dict.pop("save_json_path")
    excluded_args = args_dict.pop("swept_args").split("|")

    for arg in excluded_args:
        args_dict.pop(arg)

    with open(save_path, "w") as write_file:
        json.dump(args_dict, write_file, indent=2)
