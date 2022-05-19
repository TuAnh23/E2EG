#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
import argparse
import logging
import os

from pecos.utils import cli, logging_util, smat_util

logging_util.setup_logging_config(level=3)

from pecos.utils.featurization.text.preprocess import Preprocessor

from .model import XTransformer, XTransformerMultiTask

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    def str_to_bool(value: str):
        if value.lower() == 'yes' or value.lower() == 'true':
            return True
        elif value.lower() == 'no' or value.lower() == 'false':
            return False
        else:
            raise ValueError

    """Parse encoding arguments"""
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "-t",
        "--text-path",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the instance text file.",
    )
    parser.add_argument(
        "-m",
        "--model-folder",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to load x-transformer model.",
    )
    parser.add_argument(
        "-o",
        "--save-emb-path",
        type=str,
        required=True,
        metavar="PATH",
        help="The path where the embeddings will be written.",
    )
    parser.add_argument(
        "--saved-text-pt",
        default=None,
        metavar="PATH",
        help="dir to save/load tokenized text tensor",
    )
    parser.add_argument(
        "--memmap",
        type=str_to_bool,
        default=False,
        help="Whether to use memmap (i.e., lazily load text data tensor from disk) rather than having everything in RAM",
    )
    parser.add_argument(
        "--mtask",
        type=str_to_bool,
        default=False,
        help="Whether the model used to encode the text was a multi-task model",
    )
    # ======= Other parameters ========
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        metavar="INT",
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--use-gpu",
        type=cli.str2bool,
        metavar="[true/false]",
        default=True,
        help="if true, use CUDA if available. Default true",
    )
    parser.add_argument(
        "--batch-gen-workers",
        type=int,
        metavar="INT",
        default=4,
        help="number of CPUs to use for batch generation",
    )
    parser.add_argument(
        "--max-pred-chunk",
        default=10**7,
        metavar="INT",
        type=int,
        help="Max number of instances to predict on at once, set to avoid OOM. Set to None to predict on all instances at once. Default 10^7",
    )
    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=1,
        metavar="INT",
        help=f"the verbose level, {', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}, default 1",
    )
    return parser


def do_encode(args):
    """Generate text embeddings with XTransformer and save to file.

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """
    if os.path.isdir(args.save_emb_path):
        args.save_emb_path = os.path.join(args.save_emb_path, "embeddings.npy")

    # load instance feature and text
    X_text = Preprocessor.load_data_from_file(args.text_path, label_text_path=None, text_pos=0)[
        "corpus"
    ]

    if args.mtask:
        xtf = XTransformerMultiTask.load(args.model_folder)
        X_emb = xtf.encode(
            X_text,
            batch_size=args.batch_size,
            batch_gen_workers=args.batch_gen_workers,
            use_gpu=args.use_gpu,
            max_pred_chunk=args.max_pred_chunk,
            memmap=args.memmap,
            saved_text_dir=args.saved_text_pt,
        )
    else:
        xtf = XTransformer.load(args.model_folder)
        X_emb = xtf.encode(
            X_text,
            batch_size=args.batch_size,
            batch_gen_workers=args.batch_gen_workers,
            use_gpu=args.use_gpu,
            max_pred_chunk=args.max_pred_chunk,
            memmap=args.memmap,
            saved_text_dir=args.saved_text_pt,
        )

    smat_util.save_matrix(args.save_emb_path, X_emb)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_encode(args)
