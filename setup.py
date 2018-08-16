from setuptools import find_packages
from skbuild import setup

import os
import re
import shlex
import subprocess
import sys

TOP_DIR = os.path.realpath(os.path.dirname(__file__))

################################################################################
# Version
################################################################################

details = subprocess.check_output(shlex.split('git describe --tags --dirty'),
                                  cwd=TOP_DIR).decode('ascii').strip()
details = details.rstrip()
is_dirty = '-dirty' in details
details = details.rstrip('-dirty')
try:
  version_tag, commits_since_tag, current_hash = details.split('-')
  current_hash = current_hash.lstrip('g')
except ValueError:
  # Happens only when the latest tag points to the current commit
  assert len(details.split('-')) == 1
  version_tag = details
  commits_since_tag = 0
  current_hash = subprocess.check_output(shlex.split('git rev-parse --short HEAD')).decode('ascii').strip()

assert re.match(r'^\d[.]\d[.]\d$', version_tag), 'Malformed version tag'
current_hash = current_hash + '.dirty' if is_dirty else current_hash

# This variable is specifically designed to meet the PEP440 standard.
version = "{0}.dev{1}+{2}".format(version_tag, commits_since_tag, current_hash)

# Require pytest-runner only when running tests
pytest_runner = (['pytest-runner>=2.0,<3dev']
                 if any(arg in sys.argv for arg in ('pytest', 'test'))
                 else [])

setup_requires = pytest_runner

setup(
  name='galvASR',
  version=version,
  description='Daniel Galvez\'s Automatic Speech Recognition Library',
  packages=find_packages('src', include=['galvASR', 'galvASR.*']),
  package_dir={'': 'src'},
  cmake_args=['-DWITH_EXTERNAL_KALDI=third_party/kaldi/kaldi', '-DUSE_TENSORFLOW=YES'],
  tests_require=['pytest'],
  setup_requires=setup_requires
)
