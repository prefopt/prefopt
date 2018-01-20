#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shlex
import subprocess
import sys

from distutils.cmd import Command
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, 'src')
PREFOPT_DIR = os.path.join(SRC_DIR, 'prefopt')
TEST_DIR = os.path.join(BASE_DIR, 'tests')

ABOUT = {}
with open(os.path.join(SRC_DIR, 'prefopt', '__about__.py')) as f:
    exec(f.read(), ABOUT)


PACKAGES = find_packages(where='src')

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]

INSTALL_REQUIRES = [
    'edward',
    'numpy',
    'scipy',
    'scipydirect',
    'tensorflow',
]
TESTS_REQUIRE = [
    'coverage',
    'pytest',
]
LINT_REQUIRES = [
    'pylint',
    'pycodestyle',
    'pydocstyle',
]
EXTRAS_REQUIRE = {
    'lint': LINT_REQUIRES,
    'test': TESTS_REQUIRE,
}
DEPENDENCY_LINKS = [
    'https://github.com/sigopt/evalset/tarball/master#egg=evalset',
]


class LintCommand(Command):
    """Custom command to run linter."""

    description = 'run linter'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        targets = [
            PREFOPT_DIR,
            TEST_DIR,
            'setup.py'
        ]
        linters = [
            'pylint',
            'pycodestyle',
            'pydocstyle',
        ]

        return_values = []
        for linter in linters:
            errno = subprocess.call(
                [linter] + targets,
                stderr=subprocess.STDOUT,
            )
            return_values.append(errno)

        sys.exit(any(return_values))


class PyTestCommand(TestCommand):

    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


CMDCLASS = {
    'lint': LintCommand,
    'test': PyTestCommand,
}


setup(
    name=ABOUT['__title__'],
    version=ABOUT['__version__'],

    description=ABOUT['__summary__'],
    license=ABOUT['__license__'],
    url=ABOUT['__uri__'],

    author=ABOUT['__author__'],
    author_email=ABOUT['__email__'],

    classifiers=CLASSIFIERS,

    package_dir={"": "src"},
    packages=PACKAGES,

    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRAS_REQUIRE,

    dependency_links=DEPENDENCY_LINKS,
    cmdclass=CMDCLASS,
)
