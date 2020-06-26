=========
PyMP2RAGE
=========

.. image:: https://github.com/Gilles86/pymp2rage/blob/master/examples/figures/qmri.png

This package is a lightweight python implementation of the algorithms described in Marques et al. (2010). They can be used to compute a unified T1-weighted, as well as a quantiative T1 map out of the two phase- and magnitude-images of a MP2RAGE-sequences.

Installation
============
To install clone this repository and install it with git:

    pip install git+https://github.com/Gilles86/pymp2rage

Usage
=====
For some examples of its usage see: 
 * `Bias-field correction, T1 estimation and masking <notebooks/MP2RAGE%20and%20T1%20fitting.ipynb>`_
 * `Automatic processing of data organized according to BIDS-format <notebooks/Load%20and%20save%20to%20BIDs%20dataset.ipynb>`_
 * `Correction for B1 transmit field inhomogenieties using a B1 fieldmap <notebooks/B1%20correction.ipynb>`_
 * `Estimate a T2* map for multi-echo MP2RAGE data (ME-MP2RAGE) <notebooks/MPM%20with%20MEMP2RAGE.ipynb>`_

Readthedocs
===========
See `pymp2rage.readthedocs.io <http://pymp2rage.readthedocs.io/>`_ for a manual compiled using Sphinx.


References
==========
* Marques, J. P., Kober, T., Krueger, G., van der Zwaag, W., van de Moortele, P.-F., & Gruetter, R. (2010). MP2RAGE, a self bias-field corrected sequence for improved segmentation and T1-mapping at high field. NeuroImage, 49(2), 1271–1281. http://doi.org/10.1016/j.neuroimage.2009.10.002
* `The original implementations in MatLab on Github of José Marques <https://github.com/JosePMarques/MP2RAGE-related-scripts>`_ 
