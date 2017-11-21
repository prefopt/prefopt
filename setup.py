#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

from distutils.cmd import Command
from setuptools import setup, find_packages


class BenchmarkCommand(Command):
    """Custom command to run optimization benchmarks."""

    description = 'run prefopt on sigopt.evalset evaluation benchmark.'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # NOTE `setup` needs to execute before `benchmarks` can be imported
        import benchmarks.sigopt_evalset
        benchmarks.sigopt_evalset.run()


class LintCommand(Command):
    """Custom command to run flake8 linter."""

    description = 'run linter.'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmd = [
            'flake8',
            'benchmarks',
            'examples',
            'prefopt',
            'setup.py',
            'tests',
        ]
        subprocess.call(
            cmd,
            stderr=subprocess.STDOUT,
        )


setup(
    # basic setup
    name='PrefOpt',
    version='0.0.1',
    packages=find_packages(exclude=[
        'examples',
        'docs',
        'tests',
    ]),

    # dependencies
    install_requires=[
        'edward>=1.3.3',
        'evalset>=1.2.1',
        'flake8>=3.4.1',
        'numpy>=1.12.1',
        'scipydirect>=1.3',
    ],

    # dependency links
    dependency_links=[
        'https://github.com/sigopt/evalset/tarball/master#egg=evalset-1.2.1',
    ],

    # testing
    test_suite='tests',

    # additional commands
    cmdclass={
        'benchmark': BenchmarkCommand,
        'lint': LintCommand,
    },

    # PyPI metadata
    author='Ian Dewancker, Jakob Bauer',
    author_email='prefopt-dev@googlegroups.com',
    url='https://github.com/prefopt/prefopt',
    license='MIT',

    classifiers=[
        # how mature is this project? common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        # supported Python versions
        'Programming Language :: Python :: 2.7',
    ]
)
