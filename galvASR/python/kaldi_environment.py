from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

log = logging.getLogger('galvASR')

GALVASR_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def setup_environment():
    if os.environ.get('KALDI_ROOT') is None:
        KALDI_ROOT = os.path.join(GALVASR_ROOT, 'third_party', 'kaldi', 'kaldi')
        log.info('KALDI_ROOT environment variable not already set. Defaulting '
                 'to submodule path: %s', KALDI_ROOT)
        os.environ['KALDI_ROOT'] = KALDI_ROOT
    if os.environ.get('OPENFST_ROOT') is None:
        OPENFST_ROOT = os.path.join(GALVASR_ROOT, 'third_party', 'kaldi', 'kaldi')
        log.info('OPENFST_ROOT environment variable not already set. Defaulting '
                 'to symlink path (to Kaldi\'s install): %s', OPENFST_ROOT)
        os.environ['OPENFST_ROOT'] = OPENFST_ROOT
    if not os.path.exists(os.path.join(os.environ['KALDI_ROOT'], 'egs')):
        raise ValueError('KALDI_ROOT is not pointing to a Kaldi repo!')
    for env_variable in ('KALDI_ROOT', 'OPENFST_ROOT'):
        if not os.path.isabs(os.environ[env_variable]):
            raise ValueError('{0} is not an absolute path. Instead, {1}'.
                             format(env_variable, os.environ[env_variable]))

    # This is a modified copy-pasta of
    # $KALDI_ROOT/tools/config/common_path.sh
    os.environ['PATH'] = """\
${KALDI_ROOT}/src/bin:\
${KALDI_ROOT}/src/chainbin:\
${KALDI_ROOT}/src/featbin:\
${KALDI_ROOT}/src/fgmmbin:\
${KALDI_ROOT}/src/fstbin:\
${KALDI_ROOT}/src/gmmbin:\
${KALDI_ROOT}/src/ivectorbin:\
${KALDI_ROOT}/src/kwsbin:\
${KALDI_ROOT}/src/latbin:\
${KALDI_ROOT}/src/lmbin:\
${KALDI_ROOT}/src/nnet2bin:\
${KALDI_ROOT}/src/nnet3bin:\
${KALDI_ROOT}/src/nnetbin:\
${KALDI_ROOT}/src/online2bin:\
${KALDI_ROOT}/src/onlinebin:\
${KALDI_ROOT}/src/sgmm2bin:\
${KALDI_ROOT}/src/sgmmbin:\
${KALDI_ROOT}/src/tfrnnlmbin:\
${OPENFST_ROOT}/bin:\
${PATH}""".format(**os.environ)

    # Get easy access to everything utils/ and steps/ (note these are
    # symlinks in all the Kaldi recipes), but originally defined in WSJ.
    os.environ['PATH'] = """\
${KALDI_ROOT}/egs/wsj/s5/:\
${PATH}
""".format(**os.environ)

    # Search for "LC_ALL" in http://kaldi-asr.org/doc/data_prep.html
    # for a discussion of the importance of this environment variable.
    os.environ['LC_ALL'] = 'C'
