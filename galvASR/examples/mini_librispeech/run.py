#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import click
import os
import subprocess
import sys

from galvASR.python import kaldi_environment

kaldi_environment.setup_environment()

@click.command()
@click.option('--stage', default=0)
@click.option('--raw-data-dir', type=str)
@click.option('--kaldi-data-dir', type=str)
@click.option('--mini-librispeech-kaldi-dir',
              default=os.path.join(os.environ['KALDI_ROOT'], 'egs/mini_librispeech/s5/'))
def run(stage, raw_data_dir, kaldi_data_dir, mini_librispeech_kaldi_dir):
    # If only this script could inherit --mini-librispeech-kaldi-dir
    # automatically.
    def call_local(script, args):
        subprocess.check_call(
            [os.path.join(mini_librispeech_kaldi_dir, 'local', script)] + args)

    if stage <= 0:
        for part in ('dev-clean-2', 'train-clean-5'):
            call_local('download_and_untar.sh',
                       [raw_data_dir, 'www.openslr.org/resources/31', part])
    if stage <= 1:
        call_local('download_lm.sh', ['www.openslr.org/resources/11',
                                      'data/local/lm'])

    if stage <= 2:
        for part in ('dev-clean-2', 'train-clean-5'):
            call_local('data_prep.sh',
                       [os.path.join(raw_data_dir, 'LibriSpeech', part),
                        os.path.join(kaldi_data_dir, part.replace('-', '_'))])

    

@click.command()
@click.argument(u'--data-dir')
def download(data_dir):
    pass


if __name__ == u'__main__':
    run()
