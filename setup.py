#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from setuptools import find_packages, setup


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
    'tests': TESTS_REQUIRE,
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
)
