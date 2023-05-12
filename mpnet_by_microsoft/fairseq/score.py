#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BLEU scoring of generated translations against reference translations.
"""


from fairseq_cli.score import cli_main


if __name__ == "__main__":
    cli_main()
