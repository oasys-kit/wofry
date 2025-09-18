#! /usr/bin/env python3

from importlib.machinery import SourceFileLoader
import os
import subprocess

try:
    from setuptools import find_packages, setup
except AttributeError:
    from setuptools import find_packages, setup

NAME = 'wofry'

VERSION = '1.0.33'
ISRELEASED = True

DESCRIPTION = 'WOFRY (Wave Optics FRamework in pYthon) kernel library'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Manuel Sanchez del Rio, Luca Rebuffi'
AUTHOR_EMAIL = 'lrebuffi@anl.gov'
URL = 'https://github.com/oasys-kit/wofry'
DOWNLOAD_URL = 'https://github.com/oasys-kit/wofry'
MAINTAINER = 'Luca Rebuffi'
MAINTAINER_EMAIL = 'lrebuffi@anl.gov'
LICENSE = 'GPLv3'

KEYWORDS = [
    'dictionary',
    'glossary',
    'synchrotron'
    'simulation',
]

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Environment :: Plugins',
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: '
    'GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
]

INSTALL_REQUIRES = (
    'syned>=1.0.26',
    'h5py',
    'srxraylib>=1.0.46'
)

SETUP_REQUIRES = (
    'setuptools',
)

PACKAGES = find_packages(exclude=('*.tests', '*.tests.*', 'tests.*', 'tests'))
PACKAGE_DATA = {}

def setup_package():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        # extra setuptools args
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        setup_requires=SETUP_REQUIRES,
    )

if __name__ == '__main__':
    setup_package()
