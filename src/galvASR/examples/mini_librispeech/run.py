#!/usr/bin/env python3

import os
import subprocess

from absl import app
from absl import flags

from galvASR.python import kaldi_environment

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage', 0, None)
flags.DEFINE_integer('max_process_parallelism', 1, None)
flags.DEFINE_string('raw_data_dir', '/export/data/mini_librispeech', None)
flags.DEFINE_string('kaldi_data_dir', 'data', None)
flags.DEFINE_string('work_dir', 'exp', None)
flags.DEFINE_string('oov_symbol', "<UNK>", None)


def main(argv):
  nj = str(FLAGS.max_process_parallelism)
  mini_librispeech_kaldi_dir = os.path.join(os.environ['KALDI_ROOT'],
                                            'egs/mini_librispeech/s5/')

  def call_local(script, args):
    with kaldi_environment.load_local(mini_librispeech_kaldi_dir):
      execv_args = [os.path.join('local', script)] + args
      print("Execv args: {}".format(execv_args))
      subprocess.check_call(execv_args)

  if FLAGS.stage <= 0:
    os.makedirs(FLAGS.raw_data_dir, exist_ok=True)
    for part in ('dev-clean-2', 'train-clean-5'):
      call_local('download_and_untar.sh',
                 [FLAGS.raw_data_dir, 'www.openslr.org/resources/31', part])
  if FLAGS.stage <= 1:
    call_local('download_lm.sh',
               ['www.openslr.org/resources/11', 'data/local/lm'])

  if FLAGS.stage <= 2:
    for part in ('dev-clean-2', 'train-clean-5'):
      call_local('data_prep.sh', [
        os.path.join(FLAGS.raw_data_dir, 'LibriSpeech', part),
        os.path.join(FLAGS.kaldi_data_dir, part.replace('-', '_'))
      ])

    call_local('prepare_dict.sh', [
      '--stage', '3',
      os.path.join(FLAGS.kaldi_data_dir, 'local', 'lm'),
      os.path.join(FLAGS.kaldi_data_dir, 'local', 'lm'),
      os.path.join(FLAGS.kaldi_data_dir, 'local', 'dict_nosp')
    ])
    subprocess.check_call([
      'utils/prepare_lang.sh',
      os.path.join(FLAGS.kaldi_data_dir, 'local', 'dict_nosp'),
      FLAGS.oov_symbol,
      os.path.join(FLAGS.kaldi_data_dir, 'local', 'lang_tmp_nosp'),
      os.path.join(FLAGS.kaldi_data_dir, 'lang_nosp')
    ])
    call_local('format_lms.sh', [
      '--src-dir',
      os.path.join(FLAGS.kaldi_data_dir, 'lang_nosp'),
      os.path.join(FLAGS.kaldi_data_dir, 'local', 'lm')
    ])
    subprocess.check_call([
      'utils/build_const_arpa_lm.sh', 'data/local/lm/lm_tglarge.arpa.gz',
      'data/lang_nosp', 'data/lang_nosp_test_tglarge'
    ])

  if FLAGS.stage <= 3:
    with kaldi_environment.load_local(mini_librispeech_kaldi_dir):
      for part in ('dev_clean_2', 'train_clean_5'):
        subprocess.check_call([
          'steps/make_mfcc.sh', '--cmd', os.environ['train_cmd'], '--nj', nj,
          os.path.join(FLAGS.kaldi_data_dir, part),
          os.path.join(FLAGS.work_dir, 'make_mfcc', part),
          os.path.join(FLAGS.kaldi_data_dir, 'mfcc')
        ])
        subprocess.check_call([
          'steps/compute_cmvn_stats.sh',
          os.path.join(FLAGS.kaldi_data_dir, part),
          os.path.join(FLAGS.work_dir, 'make_mfcc', part),
          os.path.join(FLAGS.kaldi_data_dir, 'mfcc')
        ])
        subprocess.check_call([
          'utils/subset_data_dir.sh', '--shortest', 'data/train_clean_5', '500',
          'data/train_500short'
        ])

  # TODO: use work_dir everywhere here!
  if FLAGS.stage <= 4:
    subprocess.check_call([
      'steps/train_mono.sh', '--boost-silence', '1.25', '--nj', nj, '--cmd',
      os.environ['train_cmd'], 'data/train_500short', 'data/lang_nosp',
      'exp/mono'
    ])

    subprocess.check_call([
      'steps/align_si.sh', '--boost-silence', '1.25', '--nj', nj, '--cmd',
      os.environ['train_cmd'], 'data/train_clean_5', 'data/lang_nosp',
      'exp/mono', 'exp/mono_ali_train_clean_5'
    ])

  if FLAGS.stage <= 5:
    subprocess.check_call([
      'steps/train_deltas.sh', '--boost-silence', '1.25', '--cmd',
      os.environ['train_cmd'], '2000', '10000', 'data/train_clean_5',
      'data/lang_nosp', 'exp/mono_ali_train_clean_5', 'exp/tri1'
    ])

    subprocess.check_call([
      'steps/align_si.sh', '--nj', nj, '--cmd', os.environ['train_cmd'],
      'data/train_clean_5', 'data/lang_nosp', 'exp/tri1',
      'exp/tri1_ali_train_clean_5'
    ])

  if FLAGS.stage <= 6:
    subprocess.check_call([
      'steps/train_lda_mllt.sh', '--cmd', os.environ['train_cmd'],
      '--splice-opts', '--left-context=3 --right-context=3', '2500', '15000',
      'data/train_clean_5', 'data/lang_nosp', 'exp/tri1_ali_train_clean_5',
      'exp/tri2'
    ])

    subprocess.check_call([
      'steps/align_si.sh', '--nj', nj, '--cmd', os.environ['train_cmd'],
      '--use-graphs', 'true', 'data/train_clean_5', 'data/lang_nosp',
      'exp/tri2', 'exp/tri2_ali_train_clean_5'
    ])

  if FLAGS.stage <= 7:
    subprocess.check_call([
      'steps/train_sat.sh', '--cmd', os.environ['train_cmd'], '2500', '15000',
      'data/train_clean_5', 'data/lang_nosp', 'exp/tri2_ali_train_clean_5',
      'exp/tri3'
    ])

  if FLAGS.stage <= 8:
    subprocess.check_call([
      'steps/get_prons.sh', '--cmd', os.environ['train_cmd'],
      'data/train_clean_5', 'data/lang_nosp', 'exp/tri3'
    ])

    subprocess.check_call([
      'utils/dict_dir_add_pronprobs.sh', '--max-normalize', 'true',
      'data/local/dict_nosp', 'exp/tri3/pron_counts_nowb.txt',
      'exp/tri3/sil_counts_nowb.txt', 'exp/tri3/pron_bigram_counts_nowb.txt',
      'data/local/dict'
    ])

    subprocess.check_call([
      'utils/prepare_lang.sh', 'data/local/dict', FLAGS.oov_symbol,
      'data/local/lang_tmp', 'data/lang'
    ])

    call_local('format_lms.sh', ['--src-dir', 'data/lang', 'data/local/lm'])

    subprocess.check_call([
      'utils/build_const_arpa_lm.sh', 'data/local/lm/lm_tglarge.arpa.gz',
      'data/lang', 'data/lang_test_tglarge'
    ])

    subprocess.check_call([
      'steps/align_fmllr.sh', '--nj', nj, '--cmd', os.environ['train_cmd'],
      'data/train_clean_5', 'data/lang', 'exp/tri3',
      'exp/tri3_ali_train_clean_5'
    ])

  if FLAGS.stage <= 9:
    subprocess.check_call([
      'utils/mkgraph.sh', 'data/lang_test_tgsmall', 'exp/tri3',
      'exp/tri3/graph_tgsmall'
    ])


  if FLAGS.stage <= 10:
    pass

  if FLAGS.stage <= 10:
    call_local('chain/run_tdnn.sh', ['--stage', '0', '--gmm', 'tri3'])


if __name__ == '__main__':
  kaldi_environment.setup_environment()
  with kaldi_environment.load_utils_and_steps():
    app.run(main)
