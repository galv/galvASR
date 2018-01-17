from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from contextlib import contextmanager
import logging
from multiprocessing import cpu_count
import os
import re

from galvASR.python import _constants

log = logging.getLogger('galvASR')


def _parse_tools_env_sh_if_exists(env_sh_file_name):
  """
  ${KALDI_ROOT}/tools/env.sh has contents that we can't generally predict

  Mutates os.environ
  """
  export_regex = r'^export ([a-zA-Z0-9_]+)=([a-zA-Z0-9_/{}]*)$'
  if not os.path.exists(env_sh_file_name):
    return
  with open(env_sh_file_name) as env_sh_fh:
    for line in env_sh_fh:
      if line.strip() == '':
        continue
      match = re.search(export_regex)
      if match:
        env_var = match.group(1)
        expression = match.group(2).replace('$', '')
        os.environ[env_var] = expression.format(**os.environ)
      else:
        logging.warn('Unknown line in tools/env.sh: {0}'.format(line))


def setup_environment():
  os.environ['KALDI_ROOT'] = _constants.KALDI_ROOT
  os.environ['OPENFST_ROOT'] = _constants.OPENFST_ROOT
  if not os.path.exists(os.path.join(os.environ['KALDI_ROOT'], 'egs')):
    raise ValueError('KALDI_ROOT is not pointing to a Kaldi repo!')
  for env_variable in ('KALDI_ROOT', 'OPENFST_ROOT'):
    if not os.path.isabs(os.environ[env_variable]):
      raise ValueError('{0} is not an absolute path. Instead, {1}'.
                       format(env_variable, os.environ[env_variable]))

  env_sh = os.path.join(os.environ['KALDI_ROOT'], 'tools/env.sh')
  _parse_tools_env_sh_if_exists(env_sh)

  # This is a modified copy-pasta of
  # $KALDI_ROOT/tools/config/common_path.sh
  os.environ['PATH'] = """\
{KALDI_ROOT}/src/bin:\
{KALDI_ROOT}/src/chainbin:\
{KALDI_ROOT}/src/featbin:\
{KALDI_ROOT}/src/fgmmbin:\
{KALDI_ROOT}/src/fstbin:\
{KALDI_ROOT}/src/gmmbin:\
{KALDI_ROOT}/src/ivectorbin:\
{KALDI_ROOT}/src/kwsbin:\
{KALDI_ROOT}/src/latbin:\
{KALDI_ROOT}/src/lmbin:\
{KALDI_ROOT}/src/nnet2bin:\
{KALDI_ROOT}/src/nnet3bin:\
{KALDI_ROOT}/src/nnetbin:\
{KALDI_ROOT}/src/online2bin:\
{KALDI_ROOT}/src/onlinebin:\
{KALDI_ROOT}/src/sgmm2bin:\
{KALDI_ROOT}/src/sgmmbin:\
{KALDI_ROOT}/src/tfrnnlmbin:\
{OPENFST_ROOT}/bin:\
{PATH}""".format(**os.environ)

  # Add utils/ to path directly for historical reasons.
  # https://github.com/kaldi-asr/kaldi/issues/2058
  # Prefer to make calls with the utils/ prefix, though.
  os.environ['PATH'] = "{KALDI_ROOT}/egs/wsj/s5/utils/:{PATH}".format(**os.environ)

  # Search for "LC_ALL" in http://kaldi-asr.org/doc/data_prep.html
  # for a discussion of the importance of this environment variable.
  os.environ['LC_ALL'] = 'C'

  # Normally the contents of cmd.sh. May want to make these
  # configurable at some point.
  #  len(os.sched_getaffinity(0)) is python 3.3+ only
  max_cpus = cpu_count()
  max_jobs_run = max_cpus - 1 if max_cpus > 1 else max_cpus
  os.environ['train_cmd'] = 'run.pl --max-jobs-run {0}'.format(max_jobs_run)
  os.environ['decode_cmd'] = 'run.pl --max-jobs-run {0}'.format(max_jobs_run)


@contextmanager
def load_utils_and_steps():
  save_cwd = os.getcwd()
  steps = os.path.join(save_cwd, 'steps')
  utils = os.path.join(save_cwd, 'utils')
  path_sh = os.path.join(save_cwd, 'path.sh')
  cmd_sh = os.path.join(save_cwd, 'cmd.sh')

  # While this should catch exceptions, and clean up the symlinks,
  # there is a small chance that python will be killed (e.g., kill
  # -9) without the exit portion of this function running, so don't
  # error out if the symlink already exists.
  if not os.path.islink(steps):
    os.symlink('{KALDI_ROOT}/egs/wsj/s5/steps'.format(**os.environ), steps)
  if not os.path.islink(utils):
    os.symlink('{KALDI_ROOT}/egs/wsj/s5/utils'.format(**os.environ), utils)
  with open(path_sh, 'w+'), open(cmd_sh, 'w+'):
    # Some legacy kaldi scripts insist upon loading path.sh in the
    # current workind directory, though this is wholly unnecessary
    # since it's a precondition to calling those scripts that the
    # environment is set up correctly. setup_environment()
    # alreadys sets up the environment variables as path.sh would,
    # so we just create an empty file. Same goes for cmd.sh
    pass
    try:
      yield
    finally:
      if os.getcwd() != save_cwd:
        log.warning('You changed your working directory. Kaldi\'s steps and '
                    'utils symlinks and path.sh and cmd.sh will still be '
                    ' cleaned up, but this is generally discouraged.')
      os.remove(steps)
      os.remove(utils)
      os.remove(path_sh)
      os.remove(cmd_sh)


@contextmanager
def load_local(recipe_dir):
  local_to_load = os.path.join(recipe_dir, 'local')
  conf_to_load = os.path.join(recipe_dir, 'conf')
  save_cwd = os.getcwd()
  local_link = os.path.join(save_cwd, 'local')
  conf_link = os.path.join(save_cwd, 'conf')
  os.symlink(local_to_load, local_link)
  os.symlink(conf_to_load, conf_link)
  try:
    yield
  finally:
    os.remove(local_link)
    os.remove(conf_link)
