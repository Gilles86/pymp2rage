from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 2
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache License, 2.0",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "pymp2rage: a Python library to process (ME-)MP2RAGE MRI data"
# Long description will go up on the pypi page
long_description = """

PyMP2RAGE
========
Pymp2rage is a lightweight Python library to process MRI data acquired with the MP2RAGE
sequence.

"""

NAME = "pymp2rage"
MAINTAINER = "Gilles de Hollander"
MAINTAINER_EMAIL = "gilles.de.hollander@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/Gilles86/pymp2rage"
DOWNLOAD_URL = ""
LICENSE = "Apache License, 2.0"
AUTHOR = "Gilles de Hollander"
AUTHOR_EMAIL = "gilles.de.hollander@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'pymp2rage': []}
REQUIRES = ["numpy", "scipy"]
