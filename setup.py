# This file was modified from Caffe2's
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

import os
import re
import shlex
import subprocess
import sys


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'galvASR')

install_requires = []
setup_requires = []
tests_require = []

################################################################################
# Pre Check
################################################################################

assert find_executable('cmake'), 'Could not find "cmake" executable!'
assert find_executable('make'), 'Could not find "make" executable!'

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

################################################################################
# Customized commands
################################################################################


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        raise RuntimeError('develop mode is not supported!')


class build_ext(setuptools.command.build_ext.build_ext):
    """
    Compiles everything when `python setup.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable. E.g. to build without cuda support run:
        `CMAKE_ARGS=-DUSE_CUDA=Off python setup.py build`

    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """
    user_options = [
        ('jobs=', 'j', 'Specifies the number of jobs to use with make')
    ]

    def initialize_options(self):
        setuptools.command.build_ext.build_ext.initialize_options(self)
        self.jobs = None

    def finalize_options(self):
        setuptools.command.build_ext.build_ext.finalize_options(self)
        # Check for the -j argument to make with a specific number of cpus
        try:
            self.jobs = int(self.jobs)
        except:
            self.jobs = None

    def _build_with_cmake(self):
        # build_temp resolves to something like: build/temp.linux-x86_64-3.5
        # build_lib resolves to something like: build/lib.linux-x86_64-3.5
        build_temp = os.path.realpath(self.build_temp)
        build_lib = os.path.realpath(self.build_lib)

        if 'CMAKE_INSTALL_DIR' not in os.environ:
            cmake_install_dir = os.path.join(build_temp, 'cmake_install')

            py_exe = sys.executable

            if 'CMAKE_ARGS' in os.environ:
                cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                # prevent crossfire with downstream scripts
                del os.environ['CMAKE_ARGS']
            else:
                cmake_args = []
            log.info('CMAKE_ARGS: {}'.format(cmake_args))

            # Need to change directories first...
            build_dir = os.path.realpath(os.path.join(TOP_DIR, 'setup-py-build'))
            os.makedirs(build_dir, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(build_dir)
            self.compiler.spawn([
              'cmake', TOP_DIR,
              '-DCMAKE_INSTALL_PREFIX:PATH={}'.format(cmake_install_dir),
              '-DPYTHON_EXECUTABLE:FILEPATH={}'.format(py_exe),
              '-DPYTHONINTERP_FOUND=YES',
              '-DUSE_TENSORFLOW=YES',
              '-DUSE_PIP_INSTALLED_TENSORFLOW=YES'
              '-DUSE_CAFFE2=NO',
            ] + cmake_args)
            os.chdir(cwd)
            
            self.compiler.spawn([
                'make', '-j',
                '-C', build_dir,
                'install'
            ])
        else:
            # if `CMAKE_INSTALL_DIR` is specified in the environment, assume
            # cmake has been run and skip the build step.
            cmake_install_dir = os.environ['CMAKE_INSTALL_DIR']

        # CMake will install the python package to a directory that mirrors the
        # standard site-packages name. This will vary slightly depending on the
        # OS and python version.  (e.g. `lib/python3.5/site-packages`)
        python_site_packages = sysconfig.get_python_lib(prefix='')
        src = os.path.join(cmake_install_dir, python_site_packages, 'galvASR')
        self.copy_tree(src, os.path.join(build_lib, 'galvASR'))

    def get_outputs(self):
        return [os.path.join(self.build_lib, 'galvASR')]

    def build_extensions(self):
        assert len(self.extensions) == 1
        self._build_with_cmake()


cmdclass = {
    'build_py': build_py,
    'develop': develop,
    'build_ext': build_ext,
}

################################################################################
# Extensions
################################################################################

ext_modules = [setuptools.Extension('galvASR-ext', [])]

################################################################################
# Packages
################################################################################

packages = []

install_requires.extend(['future',
                         'setuptools',
                         'absl-py',
                         ])
#                         'tf-nightly-gpu==1.6.0.dev20180204', ])

################################################################################
# Test
################################################################################

setup_requires.append('pytest-runner')
tests_require.append('pytest-cov')

################################################################################
# Final
################################################################################

setuptools.setup(
    name='galvASR',
    version=version,
    description='Daniel Galvez\'s Automatic Speech Recognition Library',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    author='Daniel Galvez',
    author_email='dt.galvez@gmail.com',
)
