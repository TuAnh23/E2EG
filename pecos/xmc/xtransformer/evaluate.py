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
import numpy as np

from pecos.utils import logging_util, smat_util
from ogb.nodeproppred import Evaluator

LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """Parse predicting arguments"""
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--y-class-true",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to the true class target.",
    )
    parser.add_argument(
        "--y-class-pred",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to the pred class target.",
    )
    parser.add_argument(
        "--verbose-level",
        type=int,
        choices=logging_util.log_levels.keys(),
        default=1,
        metavar="INT",
        help=f"the verbose level, "
             f"{', '.join([str(k) + ' for ' + logging.getLevelName(v) for k, v in logging_util.log_levels.items()])}, "
             f"default 1",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv"
    )
    return parser


def do_evaluate(args):
    """Evaluate predictions.

    Args:
        args (argparse.Namespace): Command line arguments parsed by `parser.parse_args()`
    """
    evaluator = Evaluator(name=args.dataset)

    y_class_true = smat_util.load_matrix(args.y_class_true, dtype=np.float32)
    y_class_pred = smat_util.load_matrix(args.y_class_pred, dtype=np.float32).argmax(axis=1)

    result_dict = evaluator.eval({'y_true': np.expand_dims(y_class_true, axis=1),
                                  'y_pred': np.expand_dims(y_class_pred, axis=1)})

    print(f"\tMulti-class accuracy: {result_dict['acc'] * 100}\n")


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    logging_util.setup_logging_config(level=args.verbose_level)
    do_evaluate(args)
