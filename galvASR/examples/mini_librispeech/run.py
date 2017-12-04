#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import click
import os
import subprocess
import tempfile

from galvASR.python import kaldi_environment

kaldi_environment.setup_environment()


@click.command()
@click.option('--stage', default=0)
@click.option('--raw-data-dir', type=str)
@click.option('--kaldi-data-dir', type=str)
@click.option('--mini-librispeech-kaldi-dir',
              default=os.path.join(os.environ['KALDI_ROOT'], 'egs/mini_librispeech/s5/'))
@click.option('--exp-dir', default='exp')
@click.option('--oov-symbol', default="<UNK>")
def run(stage, raw_data_dir, kaldi_data_dir, mini_librispeech_kaldi_dir,
        exp_dir, oov_symbol):
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

        call_local('prepare_dict.sh',
                   ['--stage', '3', os.path.join(kaldi_data_dir, 'local', 'lm'),
                    os.path.join(kaldi_data_dir, 'local', 'lm'),
                    os.path.join(kaldi_data_dir, 'local', 'dict_nosp')])
        subprocess.check_call(['utils/prepare_lang.sh',
                               os.path.join(kaldi_data_dir, 'local', 'dict_nosp'),
                               oov_symbol,
                               os.path.join(kaldi_data_dir, 'local', 'lang_tmp_nosp'),
                               os.path.join(kaldi_data_dir, 'lang_nosp')])
        call_local('format_lms.sh',
                   ['--src-dir', os.path.join(kaldi_data_dir, 'lang_nosp'),
                    os.path.join(kaldi_data_dir, 'local', 'lm')])
        subprocess.check_call(['utils/build_const_arpa_lm.sh',
                               'data/local/lm/lm_tglarge.arpa.gz',
                               'data/lang_nosp', 'data/lang_nosp_test_tglarge'])

    if stage <= 3:
        with tempfile.NamedTemporaryFile('w+', suffix='conf') as mfcc_config:
            mfcc_config.write("""\
--use-energy=false   # use average of log energy, not energy.
--num-mel-bins=40    # similar to Google's setup.
--num-ceps=40        # there is no dimensionality reduction.
--low-freq=20        # low cutoff frequency for mel bins... this is high-bandwidth data, so
                     # there might be some information at the low end.
--high-freq=-400     # high cutoff frequently, relative to Nyquist of 8000 (=7600)""")
            for part in ('dev_clean_2', 'train_clean_5'):
                subprocess.check_call(['steps/make_mfcc.sh', '--cmd', os.environ['train_cmd'],
                                       '--mfcc-config', mfcc_config.name,
                                       os.path.join(kaldi_data_dir, part),
                                       os.path.join(exp_dir, 'make_mfcc', part),
                                       os.path.join(kaldi_data_dir, 'mfcc')])
                subprocess.check_call(['steps/compute_cmvn_stats.sh',
                                       os.path.join(kaldi_data_dir, part),
                                       os.path.join(exp_dir, 'make_mfcc', part),
                                       os.path.join(kaldi_data_dir, 'mfcc')])

    if stage <= 4:
        pass


if __name__ == u'__main__':
    with kaldi_environment.kaldi_load_utils_and_steps():
        run()
