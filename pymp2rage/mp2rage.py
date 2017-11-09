import nibabel as nb
from nilearn import image, masking
import numpy as np
import logging
from bids.grabbids import BIDSLayout
import pandas
import re
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from .utils import MPRAGEfunc_varyingTR, MP2RAGE_lookuptable


class MP2RAGE(object):

    """ This object can calculate a Unified T1-weighted image and a
    quantitative T1 map, based on the magnitude and phase-information of the two
    volumes of a MP2RAGE-sequence (Marques et al., 2010).

    It can also further correct this map for B1 inhomogenieties using a
    B1 map (Marques et al., 2014).

    Args:
        MPRAGE_tr (float): MP2RAGE TR in seconds
        invtimesAB (list of floats): Inversion times in seconds
        flipangleABdegree (list of floats): Flip angle of the two readouts in degrees
        nZslices (list of integers): Slices Per Slab * [PartialFourierInSlice-0.5  0.5]
        FLASH_tr (float): TR of the GRE readout
        sequence (string): Kind of sequence (default is 'normal')
        inversion_efficiency: inversion efficiency of the MP2RAGE PULSE (Default is 0.96, 
                              as measured on a Siemens system).
        B0 (float): Field strength in Tesla
        inv1_combined (filename or Nifti1Image, optional): Magnitude and phase image corresponding to
                                                           first inversion pulse. Should always consist
                                                           of two volumes.
        inv2_combined (filename or Nifti1Image, optional): Magnitude and phase image corresponding to
                                                           second inversion pulse. Should always consist
                                                           of two volumes.
        inv1 (filename or Nifti1Image, optional): Magnitude image of first inversion pulse.
                                                  Should always consist of one volume.
        inv1ph (filename or Nifti1Image, optional): Phase image of first inversion pulse.
                                                    Should always consist of one volume.
        inv2 (filename or Nifti1Image, optional): Magnitude image of second inversion pulse.
                                                  Should always consist of one volume.
        inv2ph (filename or Nifti1Image, optional): Phase image of second inversion pulse.
                                                    Should always consist of one volume.
        B1_fieldmap (filename or Nifti1Image, optional): B1 fieldmap that indicates the ratio or percentage
                                                         of the real versus intended flip angle.
                                                         Can be used to correct T1-weighted image and T1 map
                                                         for B1+ inhomogenieties.
    Attributes:
        t1 (Nifti1Image): Quantitative T1 map
        t1_uni (Nifti1Image): Bias-field corrected T1-weighted map

        t1_masked (Nifti1Image): Quantitative T1 map, masked 
        t1w_uni_masked (Nifti1Image): Bias-field corrected T1-weighted map, masked
    """

    def __init__(self, 
                 MPRAGE_tr=None,
                 invtimesAB=None,
                 flipangleABdegree=None,
                 nZslices=None,
                 FLASH_tr=None,
                 sequence='normal',
                 inversion_efficiency=0.96,
                 B0=7,
                 inv1_combined=None, 
                 inv2_combined=None, 
                 inv1=None, 
                 inv1ph=None, 
                 inv2=None, 
                 inv2ph=None,
                 B1_fieldmap=None): 



        if inv1_combined is not None:
            inv1_combined = image.load_img(inv1_combined, dtype=np.double)

            if inv1_combined.shape[3] != 2:
                raise Exception('inv1_combined should contain two volumes')

            if (inv1 is not None) or (inv1ph is not None):
                raise Exception('*Either* give inv1_combined *or* inv1 and inv1_ph.')

            self.inv1 = image.index_img(inv1_combined, 0)
            self.inv1ph = image.index_img(inv1_combined, 1)

        if inv2_combined is not None:
            inv2_combined = image.load_img(inv2_combined, dtype=np.double)

            if inv2_combined.shape[3] != 2:
                raise Exception('inv2_combined should contain two volumes')

            if (inv1 is not None) or (inv1ph is not None):
                raise Exception('*Either* give inv2_combined *or* inv2 and inv2_ph.')

            self.inv2 = image.index_img(inv2_combined, 0)
            self.inv2ph = image.index_img(inv2_combined, 1)

        if inv1 is not None:
            self.inv1 = image.load_img(inv1, dtype=np.double)

        if inv2 is not None:
            self.inv2 = image.load_img(inv2, dtype=np.double)

        if inv1ph is not None:
            self.inv1ph = image.load_img(inv1ph, dtype=np.double)

        if inv2ph is not None:
            self.inv2ph = image.load_img(inv2ph, dtype=np.double)


        # Normalize phases between 0 and 2 pi
        self.inv1ph = image.math_img('((x - np.max(x))/ - np.ptp(x)) * 2 * np.pi', x=self.inv1ph)
        self.inv2ph = image.math_img('((x - np.max(x))/ - np.ptp(x)) * 2 * np.pi', x=self.inv2ph)

        # Set parameters
        self.MPRAGE_tr = MPRAGE_tr
        self.invtimesAB = invtimesAB
        self.flipangleABdegree = flipangleABdegree
        self.nZslices = nZslices
        self.FLASH_tr = FLASH_tr
        self.sequence = sequence
        self.inversion_efficiency = inversion_efficiency
        self.B0 = B0
        
        # set up t1
        self._t1 = None

        # Preset masked versions
        self._t1w_uni = None
        self._mask = None
        self._inv1_masked = None
        self._inv2_masked = None
        self._t1_masked = None
        self._t1w_uni_masked = None

        if B1_fieldmap is not None:
            self.b1 = nb.load(B1_fieldmap)

            self._t1_b1_corrected = None
            self._t1w_uni_b1_corrected = None
            self._t1_b1_corrected_masked = None
            self._t1w_uni_b1_corrected_masked = None


    @property
    def t1w_uni(self):
        if self._t1w_uni is None:
            self.fit_t1w_uni()

        return self._t1w_uni

    @property
    def t1(self):
        if self._t1 is None:
            self.fit_t1()

        return self._t1

    @property
    def t1(self):
        if self._t1 is None:
            self.fit_t1()

        return self._t1
    
    def fit_t1w_uni(self):
        compINV1 = self.inv1.get_data() * np.exp(self.inv1ph.get_data() * 1j)
        compINV2 = self.inv2.get_data() * np.exp(self.inv2ph.get_data() * 1j)

        # Scale to 4095
        self._t1w_uni = (np.real(compINV1*compINV2/(compINV1**2 + compINV2**2)))*4095+2048

        # Clip anything outside of range
        self._t1w_uni = np.clip(self._t1w_uni, 0, 4095)

        # Convert to nifti-image
        self._t1w_uni = nb.Nifti1Image(self._t1w_uni, self.inv1.affine)

        return self._t1w_uni

    def fit_t1(self):
        if (self.MPRAGE_tr is None) or (self.invtimesAB is None) or (self.flipangleABdegree is None) \
                or (self.nZslices is None) or (self.FLASH_tr is None):
            raise Exception("All sequence parameters (MPRAGE_tr, invtimesAB, flipangleABdegree, nZslices,' \
                            ' and FLASH_TR) have to be provided for T1 fitting")
        
        Intensity, T1Vector, _ = MP2RAGE_lookuptable(self.MPRAGE_tr, self.invtimesAB, self.flipangleABdegree, 
                                                     self.nZslices, self.FLASH_tr, self.sequence, 2,
                                                     self.inversion_efficiency, self.B0)
        
        T1Vector = np.append(T1Vector, T1Vector[-1] + (T1Vector[-1]-T1Vector[-2]))    
        Intensity = np.append(Intensity, -0.5)
        
        
        T1Vector = T1Vector[np.argsort(Intensity)]
        Intensity = np.sort(Intensity)
        
        self._t1 = np.interp(-0.5 + self.t1w_uni.get_data()/4096, Intensity, T1Vector)
        self._t1[np.isnan(self._t1)] = 0
        
        # Convert to milliseconds
        self._t1 *= 1000
        
        # Make image
        self._t1 = nb.Nifti1Image(self._t1, self.t1w_uni.affine)
        
        return self._t1



    def fit_mask(self, modality='INV2', smooth_fwhm=2.5, threshold=None, **kwargs):
        """Fit a mask based on one of the MP2RAGE images (usually INV2).

        This function creates a mask of the brain and skull, so that parts of the image
        that have insufficient signal for proper T1-fitting can be ignored.
        By default, it uses a slightly smoothed version of the INV2-image (to increase
        SNR), and the "Nichols"-method, as implemented in the ``nilearn``-package,
        to remove low-signal areas. The "Nichols"-method looks for the lowest 
        density in the intensity histogram and places a threshold cut there.

        You can also give an arbitrary 'threshold'-parameter to threshold the image
        at a specific value.

        The resulting mask is returned and stored in the ``mask``-attribute of
        the MP2RAGEFitter-object. 

        Args:
            modality (str): Modality to  use for masking operation (defaults to INV2) 
            smooth (float): The size of the smoothing kernel to apply in mm (defaults 
                            to 2.5 mm)
            threshold (float): If not None, the image is thresholded at this (arbitary)
                               number.
            **kwargs: These arguments are forwarded to nilearn's ``compute_epi_mask``

        Returns:
            The computed mask

        """

        im = getattr(self, modality.lower())

        if threshold is None:
            smooth_im = image.smooth_img(im, smooth_fwhm)
            self._mask = masking.compute_epi_mask(smooth_im, **kwargs)
        else:
            self._mask = image.math_img('im > %s' % threshold, im=im)

        return self.mask

    @property
    def mask(self):
        if self._mask is None:
            logging.warning('Mask is not computed yet. Computing the mask now with' \
                            'default settings using nilearn\'s compute_epi_mask)' \
                            'For more control, use the ``fit_mask``-function.')
            self.fit_mask()

        return self._mask


    @property
    def t1_masked(self):
        return image.math_img('t1 * mask', t1=self.t1, mask=self.mask)

    @property
    def t1w_uni_masked(self):
        return image.math_img('t1w_uni * mask', t1w_uni=self.t1w_uni, mask=self.mask)

    @property
    def inv1_masked(self):
        return image.math_img('inv1 * mask', inv1=self.inv1, mask=self.mask)

    @property
    def inv2_masked(self):
        return image.math_img('inv2 * mask', inv2=self.inv2, mask=self.mask)


    @classmethod
    def from_bids(cls, source_dir, subject, **kwargs):
        """ Creates a MP2RAGE-object from a properly organized BIDS-folder.

        The folder should be organized as follows:

        sub-01/anat/:
         * sub-01_inv-1_part-mag_MP2RAGE.nii
         * sub-01_inv-1_part-phase_MP2RAGE.nii
         * sub-01_inv-2_part-mag_MP2RAGE.nii
         * sub-01_inv-2_part-phase_MP2RAGE.nii
         * sub-01_inv-1_MP2RAGE.json
         * sub-01_inv-2_MP2RAGE.json

         The JSON-files should contain all the necessary MP2RAGE sequence parameters
         and should look something like this:

         sub-01/anat/sub-01_inv-1_MP2RAGE.json:
             {
                "InversionTime":0.8,
                "FlipAngle":5,
                "ExcitationRepetitionTime":0.0062,
                "InversionRepetitionTime":5.5,
                "NumberShots":159
             }

         sub-01/anat/sub-01_inv-2_MP2RAGE.json:
             {
                "InversionTime":2.7,
                "FlipAngle":7,
                "ExcitationRepetitionTime":0.0062,
                "InversionRepetitionTime":5.5,
                "NumberShots":159
             }

        A MP2RAGE-object can now be created from the BIDS folder as follows:

        Example:
            >>> import pymp2rage
            >>> mp2rage = pymp2rage.MP2RAGE.from_bids('/data/sourcedata/', '01')

        Args:
            source_dir (BIDS dir): directory containing all necessary files
            subject (str): subject identifier
            **kwargs: additional keywords that are forwarded to get-function of
            BIDSLayout. For example `ses` could be used to select specific session.
        """



        layout = BIDSLayout(source_dir)
        
        filenames = layout.get(subject=subject, return_type='file', type='MP2RAGE', extensions=['.nii', '.nii.gz'], **kwargs)
        
        part_regex = re.compile('part-(mag|phase)')
        inv_regex = re.compile('inv-([0-9]+)')
        
        parts = [part_regex.search(fn).group(1) if part_regex.search(fn) else None for fn in filenames]
        inversion_idx = [int(inv_regex.search(fn).group(1)) if inv_regex.search(fn) else None for fn in filenames]
        
        # Check whether we have everything
        df = pandas.DataFrame({'fn':filenames, 
                               'inv':inversion_idx,
                               'part':parts})
        
        tmp = df[np.in1d(df.inv, [1, 2]) & np.in1d(df.part, ['mag', 'phase'])]
        check = (len(tmp) == 4) & (tmp.groupby(['inv', 'part']).size() == 1).all()
        
        if not check:
            raise ValueError('Did not find exactly one Magnitude and phase image for two' \
                             'inversions. Only found: %s' % tmp.fn.tolist())
        
        
        df = df.set_index(['inv', 'part'])
        
        inv1 = df.loc[1, 'mag'].fn
        inv1ph = df.loc[1, 'phase'].fn
        inv2 = df.loc[2, 'mag'].fn
        inv2ph = df.loc[2, 'phase'].fn

        print('Found following files:\n * inv1, magnitude: {inv1}\n * inv1, phase: {inv1ph}'\
              '\n * inv2, magnitude: {inv2}\n * inv2, phase: {inv2ph}'.format(**locals()))
        
        meta_inv1 = layout.get_metadata(inv1)
        meta_inv2 = layout.get_metadata(inv2)
        
        for key in ['InversionRepetitionTime', 'NumberShots', 'PartialFourier']:
            if key in meta_inv1:
                if meta_inv1[key] != meta_inv2[key]:
                    raise ValueError('%s of INV1 and INV2 are different!' % key)        
        
        MPRAGE_tr = meta_inv1['InversionRepetitionTime']    
        invtimesAB = [meta_inv1['InversionTime'], meta_inv2['InversionTime']]    
        flipangleABdegree = [meta_inv1['FlipAngle'], meta_inv2['FlipAngle']]
        
        if 'PartialFourier' in meta_inv1.keys():
            nZslices = meta_inv1['NumberShots'] * np.array([meta_inv1['PartialFourier'] -.5, 0.5])    
        else: 
            nZslices = meta_inv1['NumberShots']
            
        FLASH_tr = [meta_inv1['ExcitationRepetitionTime'], meta_inv2['ExcitationRepetitionTime']]
        
        B0 = meta_inv1.pop('FieldStrength', 7)
        
        return cls(MPRAGE_tr,
                   invtimesAB,
                   flipangleABdegree,
                   nZslices,
                   FLASH_tr,
                   inv1=inv1,
                   inv1ph=inv1ph,
                   inv2=inv2,
                   inv2ph=inv2ph)


    def write_files(self, path=None, prefix=None, compress=True, masked=False):
        """ Write bias-field corrected T1-weighted image and T1 map to disk 
        as Nifti-files.

        If no filename or directory are given, the filename of INV1 is used
        as a template.

        The resulting files have the following names:
         * <path>/<prefix>_T1.nii[.gz]
         * <path>/<prefix>_T1w.nii[.gz]
         * [<path>/<prefix>_T1_masked.nii[.gz]]
         * [<path>/<prefix>_T1w_masked.nii[.gz]]
        
        Args:
            path (str, Optional): Directory where files should be placed
            prefix (str, Optional): Prefix of final filename (<path>/


        Example:
            >>> import pymp2rage
            >>> mp2rage = pymp2rage.MP2RAGE.from_bids('/data/sourcedata', '01')
            >>> mp2rage.write_files() # This write sub-01_T1w.nii.gz and 
                                      # sub-01_T1map.nii.gz to 
                                      # /data/sourcedata/sub-01/anat

        """

        if path is None:
            path = os.path.dirname(self.inv1.get_filename())
        

        if prefix is None:
            prefix = os.path.split(self.inv1.get_filename())[-1]

            INV_reg = re.compile('_?(INV)-?(1|2)', re.IGNORECASE)
            part_reg = re.compile('_?(part)-?(mag|phase)', re.IGNORECASE)
            MP2RAGE_reg = re.compile('_(ME)?MP2RAGE', re.IGNORECASE)

            for reg in [INV_reg, part_reg, MP2RAGE_reg]:
                prefix = reg.sub('', prefix)

            prefix = os.path.splitext(prefix)[0]

        ext = '.nii.gz' if compress else '.nii'

        t1_filename = os.path.join(path, prefix+'_T1map'+ext)
        print("Writing T1 map to %s" % t1_filename)
        self.t1.to_filename(t1_filename)

        t1w_uni_filename = os.path.join(path, prefix+'_T1w'+ext)
        print("Writing bias-field corrected T1-weighted image to %s" % t1w_uni_filename)
        self.t1w_uni.to_filename(t1w_uni_filename)

        if masked:
            t1_masked_filename = os.path.join(path, prefix+'_T1map_masked'+ext)
            print("Writing masked T1 map to %s" % t1_masked_filename)
            self.t1_masked.to_filename(t1_masked_filename)

            t1w_uni_masked_filename = os.path.join(path, prefix+'_T1w_masked'+ext)
            print("Writing masked bias-field corrected T1-weighted image to %s" % t1w_uni_masked_filename)
            self.t1w_uni_masked.to_filename(t1w_uni_masked_filename)

        if hasattr(self, 't1_b1_corrected'):
            t1_b1_corrected_filename = os.path.join(path, prefix+'_T1map_b1corrected.nii.gz')
            print('Writing B1-corrected T1 map to %s' % t1_b1_corrected_filename)
            self.t1_b1_corrected.to_filename(t1_b1_corrected_filename)

        if hasattr(self, 't1w_uni_b1_corrected'):
            t1w_uni_b1_corrected_filename = os.path.join(path, prefix+'_T1w_b1corrected.nii.gz')
            print('Writing B1-corrected T1-weighted image to %s' % t1w_uni_b1_corrected_filename)
            self.t1w_uni_b1_corrected.to_filename(t1w_uni_b1_corrected_filename)



    def plot_B1_effects(self):
        
        """ This function replicates the plot_MP2RAGEproperties-function
        of the Matlab script by José Marques.
        
        It shows what effect different B1 differences as compared to intended
        flip angle has on the resulting contrast between gray matter (GM),
        white matter (WM), and cerebrospinal fluid (CSF).
        
        
        see:
        https://github.com/JosePMarques/MP2RAGE-related-scripts/blob/master/func/plotMP2RAGEproperties.m"""
        
        
        Signalres = lambda x1, x2: x1*x2/(x2**2+x1**2)
        noiseres = lambda x1, x2: ((x2**2-x1**2)**2 / (x2**2 + x1**2)**3 )**(0.5)

        Contrast = []

        if self.B0 == 3:
            T1WM=0.85
            T1GM=1.35
            T1CSF=2.8
            B1range=np.arange(0.8, 1.21, 0.1)
        else:
            T1WM=1.1
            T1GM=1.85
            T1CSF=3.9
            B1range=np.arange(0.6, 1.41, 0.2)
            
        lines = []

        for B1 in B1range:
            
            effective_flipangle = B1 * np.array(self.flipangleABdegree)
            MP2RAGEamp, T1vector, IntensityBeforeComb = MP2RAGE_lookuptable(self.MPRAGE_tr, 
                                                                            self.invtimesAB, effective_flipangle, 
                                                                            self.nZslices, self.FLASH_tr, 
                                                                            self.sequence, nimages=2,
                                                                            inversion_efficiency=self.inversion_efficiency, 
                                                                            B0=self.B0, all_data=1)
            

            lines.append(plt.plot(MP2RAGEamp, T1vector, color=np.array([0.5]*3)*B1, label='B1 = %.2f' % B1))
            posWM= np.argmin(np.abs(T1WM - T1vector))
            posGM= np.argmin(np.abs(T1GM - T1vector))
            posCSF = np.argmin(np.abs(T1CSF- T1vector))

            Signal= Signalres(IntensityBeforeComb[[posWM,posGM,posCSF],0], IntensityBeforeComb[[posWM,posGM,posCSF],1])
            noise = noiseres(IntensityBeforeComb[[posWM,posGM,posCSF],0],IntensityBeforeComb[[posWM,posGM,posCSF],1])


            Contrast.append(1000 * np.sum((Signal[1:]-Signal[:-1])/np.sqrt(noise[1:]**2+noise[:-1]**2))/np.sqrt(self.MPRAGE_tr))
            
            
        plt.axhline(T1CSF, color='red')
        plt.axhline(T1GM, color='green')
        plt.axhline(T1WM, color='blue')
        
        plt.text(0.35,T1WM,'White Matter')
        plt.text(0.35,T1GM,'Grey Matter')
        
        plt.text(-0.3,(T1CSF+T1GM)/2, 'Contrast over B1 range', va='center')
        plt.text(0,(T1CSF+T1GM)/2,'\n'.join(['%.2f' % c for c in Contrast]), va='center')
        
        plt.legend(loc='upper right')
        
        
        return Contrast

    def correct_for_B1(self, B1=None, check_B1_range=True):
        """ This function corrects the bias-field corrected T1-weighted image (`t1w_uni`-attribute)
        and the quantitative T1 map (`t1`-attribute) for B1 inhomogenieties using a B1 field map. 
        (see Marques and Gruetter, 2013).
        It assumes that the B1 field map is either a ratio of the real and intended 
        flip angle (range of approximately 0 - 2) *or* the percentage of the real
        vs intended flip angle (range of approximately 0 - 200). 
        
        If the B1 map has a different resolution, it is resampled to the resolution
        of INV1 and INV2. *This function assumes your MP2RAGE images and the B1 map
        are in the same space*.

        If the B1 map is not immediately acquired after the MP2RAGE sequence, 
        you should register the (magnitude image corresponding to) the B1 map to 
        INV1/INV2 first. 

        The corrected images are stored in the `t1w_uni_b1_corrected` and the
        `t1_b1_corrected`-attributes as well as returned as a tuple
        
        
        Args:
            B1 (filename): B1 field map, either as a ratio or as a percentage. If 
                           set to None, use self.B1 (set when the MP2RAGE class was 
                           initialized).
            check_B1_range (bool): whether the fuction should check whether the range
                                   of the B1 fieldmap makes sense (centered at 1, range of
                                   roughly 0-2 or 0-200).

        Returns:
            (tuple): tuple containing:

                t1w_uni_b1_corrected: A T1-weighted image corrected for B1 inhomogenieties
                t1_b1_corrected: A quantiative T1-weighted image corrected for B1 inhomogenieties

        
        """
        
        if B1 is None:
            if not hasattr(self, 'b1'):
                raise ValueError('Can not correct for B1 inhomogenieties without B1' \
                                 'fieldmap')
            B1 = self.b1
        else:
            B1 = nb.load(B1)
            self.b1 = B1
            
        if B1.shape != self.inv1.shape:
            logging.warning('B1 map has different resolution from ' \
                            'INV1, I am resampling the B1 map to match INV1...' \
                            'Make sure they are in the same space ()')
            B1 = image.resample_to_img(B1, self.inv1)
        
        if check_B1_range:
            
            if np.median(B1.get_data()) > 10:
                logging.warning('B1 does not seem to be in range of 0-2. ' \
                                'Assuming B1 is measured as a percentage, dividing by ' \
                                '100...')
                B1 = image.math_img('B1 / 100.', B1=B1)


            if (np.abs(np.median(B1.get_data()) - 1) > .25):
                raise ValueError('Median of B1 is far from 1. The scale of this B1 map '\
                                 'is most likely wrong.')
            


        # creates a lookup table of MP2RAGE intensities as a function of B1 and T1    
        B1_vector = np.arange(0.005, 1.9, 0.05)
        T1_vector =  np.arange(0.5, 5.2001, 0.05)
        
        MP2RAGEmatrix = np.zeros((len(B1_vector), len(T1_vector)))
            
        for k, b1val in enumerate(B1_vector):
            
            effective_flipangle = b1val * np.array(self.flipangleABdegree)
            
            Intensity, T1vector, _ = MP2RAGE_lookuptable(self.MPRAGE_tr, 
                                                         self.invtimesAB, effective_flipangle, 
                                                         self.nZslices, self.FLASH_tr, 
                                                         self.sequence, nimages=2,
                                                         inversion_efficiency=self.inversion_efficiency, 
                                                         B0=self.B0,
                                                         all_data=0)
            
    #         f = interpolate.interp1d(np.sort(T1vector), Intensity[np.argsort(T1vector)],
            f = interpolate.interp1d(T1vector, Intensity, 
                                     bounds_error=False, fill_value=np.nan, )
            MP2RAGEmatrix[k, :] = f(T1_vector)
        
    #     return MP2RAGEmatrix

        # make the matrix  MP2RAGEMatrix into T1_matrix(B1,ratio)
        npoints=40;
        MP2RAGE_vector=np.linspace(-0.5,0.5,npoints);
        
        T1matrix = np.zeros((len(B1_vector), npoints))

        for k, b1val in enumerate(B1_vector):
            
            
            if np.isnan(MP2RAGEmatrix[k,:]).any():

                signal = MP2RAGEmatrix[k,:].copy()        
                signal[np.isnan(signal)] = np.linspace(-0.5,-1,np.isnan(signal).sum())

                f = interpolate.interp1d(signal, T1_vector, bounds_error=False, fill_value='extrapolate')#fill_value='extrapolate')

                T1matrix[k,:] = f(MP2RAGE_vector)
                
            else:   
                signal = MP2RAGEmatrix[k,:]
                f = interpolate.interp1d(np.sort(MP2RAGEmatrix[k,:]), T1_vector[np.argsort(MP2RAGEmatrix[k,:])], 'cubic', 
                                         bounds_error=False, fill_value='extrapolate')
                T1matrix[k, :] = f(MP2RAGE_vector)
                
                 
        # *** Create correted T1 map ***
        # Make interpolation function that gives T1, given B1 and T1w signal
        f = interpolate.RectBivariateSpline(B1_vector, MP2RAGE_vector, T1matrix, kx=1, ky=1)
        
        
        x = B1.get_data()    
        
        # Rescale T1w signal to [-.5, .5]
        y = self.t1w_uni.get_data() / 4095 - .5
        
        # Precache corrected T1 map
        t1c = np.zeros_like(x)
        
        # Make a mask that excludes non-interesting voxels
        mask = (x != 0) & (y != 0) & ~np.isnan(y)
        
        # Interpolate T1-corrected map
        t1c[mask] = f(x[mask], y[mask], grid=False)
        self.t1_b1_corrected = nb.Nifti1Image(t1c * 1000, self.t1.affine)
        
        # *** Create corrected T1-weighted image ***
        Intensity, T1vector, _ = MP2RAGE_lookuptable(self.MPRAGE_tr, 
                                                     self.invtimesAB, self.flipangleABdegree, 
                                                     self.nZslices, self.FLASH_tr, 
                                                     self.sequence, nimages=2,
                                                     inversion_efficiency=self.inversion_efficiency, 
                                                     B0=self.B0,
                                                     all_data=0)
        
        f = interpolate.interp1d(T1vector, Intensity, bounds_error=False, fill_value=-0.5)
        t1w_uni_corrected = (f(t1c) + .5) * 4095    
        self.t1w_uni_b1_corrected = nb.Nifti1Image(t1w_uni_corrected, self.t1w_uni.affine)
        
        return self.t1_b1_corrected, self.t1w_uni_b1_corrected


class MEMP2RAGE(MP2RAGE):
    """ This is an extension of the MP2RAGE-class that can deal with multi-echo
    data. """

    def __init__(self, 
                 echo_times,
                 MPRAGE_tr=None,
                 invtimesAB=None,
                 flipangleABdegree=None,
                 nZslices=None,
                 FLASH_tr=None,
                 sequence='normal',
                 inversion_efficiency=0.96,
                 B0=7,
                 inv1=None, 
                 inv1ph=None, 
                 inv2=None, 
                 inv2ph=None,
                 B1_fieldmap=None): 

        
        if type(inv2) is list:
            inv2 = image.concat_imgs(inv2)

        if type(inv2ph) is list:
            inv2ph = image.concat_imgs(inv2ph)
        
        self.t2starw_echoes = inv2
        self.inv2_echo_times = np.array(echo_times)
        self.n_echoes = len(echo_times)

        if inv2ph is not None:
            self.t2starw_echoes_phase = inv2ph

        if self.t2starw_echoes.shape[-1] != self.n_echoes:
            raise ValueError('Length of echo_times should correspond to the number of echoes'\
                             'in INV2')

        
        inv2 = image.index_img(self.t2starw_echoes, 0)
        inv2ph = image.index_img(self.t2starw_echoes_phase, 0)


        self._s0 = None
        self._t2star = None
        self._t2starw = None

        super(MEMP2RAGE, self).__init__(MPRAGE_tr=MPRAGE_tr,
                                        invtimesAB=invtimesAB,
                                        flipangleABdegree=flipangleABdegree,
                                        nZslices=nZslices,
                                        FLASH_tr=FLASH_tr,
                                        sequence=sequence,
                                        inversion_efficiency=inversion_efficiency,
                                        B0=B0,
                                        inv1=inv1, 
                                        inv1ph=inv1ph, 
                                        inv2=inv2, 
                                        inv2ph=inv2ph,
                                        B1_fieldmap=B1_fieldmap) 

    def fit_t2star(self, min_t2star=0, max_t2star=300):

        tmp = np.log(self.t2starw_echoes.get_data())
        idx = (tmp > 0).all(-1)

        s0 = np.zeros(self.t2starw_echoes.shape[:3])
        t2star = np.zeros(self.t2starw_echoes.shape[:3])

        x = np.concatenate((np.ones((self.n_echoes, 1)), -self.inv2_echo_times[..., np.newaxis]), 1)

        beta, _, _, _ = np.linalg.lstsq(x, tmp[idx].T)

        s0[idx] = np.exp(beta[0])
        t2star[idx] = 1./beta[1]

        s0 = s0 / np.percentile(s0, 95) * 4095

        t2star[t2star < min_t2star] = min_t2star
        t2star[t2star > max_t2star] = max_t2star

        self._s0 = image.new_img_like(self.t2starw_echoes, s0)
        self._t2star = image.new_img_like(self.t2starw_echoes, t2star)

        return self._t2star

    @property
    def t2star(self):
        if self._t2star is None:
            self.fit_t2star()

        return self._t2star

    @property
    def s0(self):
        if self._s0 is None:
            self.fit_t2star()

        return self._s0

    @property
    def t2starw(self):
        if self._t2starw is None:
            self._t2starw = image.mean_img(self.t2starw_echoes)
            self._t2starw = image.math_img('im / np.percentile(im, 95) * 4095', im=self._t2starw)

        return self._t2starw

    @classmethod
    def from_bids(cls, source_dir, subject, **kwargs):
    
        """ Creates a MEMP2RAGE-object from a properly organized BIDS-folder.

        The folder should be organized similar to this example:

        sub-01/anat/:
        # The first inversion time volumes
         * sub-01_inv-1_part-mag_MEMP2RAGE.nii
         * sub-01_inv-1_part-phase_MEMP2RAGE.nii
         
        # The four echoes of the second inversion (magnitude)
         * sub-01_inv-2_part-mag_echo-1_MEMP2RAGE.nii
         * sub-01_inv-2_part-mag_echo-2_MEMP2RAGE.nii
         * sub-01_inv-2_part-mag_echo-3_MEMP2RAGE.nii
         * sub-01_inv-2_part-mag_echo-4_MEMP2RAGE.nii
         
        # The four echoes of the second inversion (phase)         
         * sub-01_inv-2_part-phase_echo-1_MEMP2RAGE.nii
         * sub-01_inv-2_part-phase_echo-2_MEMP2RAGE.nii
         * sub-01_inv-2_part-phase_echo-3_MEMP2RAGE.nii
         * sub-01_inv-2_part-phase_echo-4_MEMP2RAGE.nii

        # The json describing the parameters of the first inversion pulse
         * sub-01_inv-1_MEMP2RAGE.json
         
        # The json describing the parameters of the second inversion pulse
         * sub-01_inv-2_echo-1_MEMP2RAGE.json
         * sub-01_inv-2_echo-2_MEMP2RAGE.json
         * sub-01_inv-2_echo-3_MEMP2RAGE.json
         * sub-01_inv-2_echo-4_MEMP2RAGE.json

         The JSON-files should contain all the necessary MP2RAGE sequence parameters
         and should look something like this:

         sub-01/anat/sub-01_inv-1_MEMP2RAGE.json:
             {
                "InversionTime":0.67,
                "FlipAngle":7,
                "ExcitationRepetitionTime":0.0062,
                "InversionRepetitionTime":6.723,
                "NumberShots":150
             }

         sub-01/anat/sub-01_inv-2_echo-1_MEMP2RAGE.json:
             {
                "InversionTime":3.855,
                "FlipAngle":6,
                "ExcitationRepetitionTime":0.0320,
                "InversionRepetitionTime":6.723,
                "NumberShots":150,
                "EchoTime": 6.0
             }
             
         sub-01/anat/sub-01_inv-2_echo-2_MEMP2RAGE.json:
             {
                "InversionTime":3.855,
                "FlipAngle":6,
                "ExcitationRepetitionTime":0.0320,
                "InversionRepetitionTime":6.723,
                "NumberShots":150,
                "EchoTime": 14.5
             }
             
         sub-01/anat/sub-01_inv-2_echo-3_MEMP2RAGE.json:
             {
                "InversionTime":3.855,
                "FlipAngle":6,
                "ExcitationRepetitionTime":0.0320,
                "InversionRepetitionTime":6.723,
                "NumberShots":150,
                "EchoTime": 23
             }
             
         sub-01/anat/sub-01_inv-2_echo-4_MEMP2RAGE.json:
             {
                "InversionTime":3.855,
                "FlipAngle":6,
                "ExcitationRepetitionTime":0.0320,
                "InversionRepetitionTime":6.723,
                "NumberShots":150,
                "EchoTime": 31.5
             }             

        A MP2RAGE-object can now be created from the BIDS folder as follows:

        Example:
            >>> import pymp2rage
            >>> mp2rage = pymp2rage.MEMP2RAGE.from_bids('/data/sourcedata/', '01')

        Args:
            source_dir (BIDS dir): directory containing all necessary files
            subject (str): subject identifier
            **kwargs: additional keywords that are forwarded to get-function of
            BIDSLayout. For example `ses` could be used to select specific session.
        """
        
        layout = BIDSLayout(source_dir)
        
        filenames = layout.get(subject=subject, return_type='file', type='MEMP2RAGE', extensions=['.nii', '.nii.gz'], **kwargs)
        
        part_regex = re.compile('part-(mag|phase)')
        inv_regex = re.compile('inv-([0-9]+)')
        echo_regex = re.compile('echo-([0-9]+)')
        
        parts = [part_regex.search(fn).group(1) if part_regex.search(fn) else None for fn in filenames]
        inversion_idx = [int(inv_regex.search(fn).group(1)) if inv_regex.search(fn) else None for fn in filenames]
        echo_idx = [int(echo_regex.search(fn).group(1)) if echo_regex.search(fn) else 1 for fn in filenames]
        
        # Check whether we have everything
        df = pandas.DataFrame({'fn':filenames, 
                               'inv':inversion_idx,
                               'part':parts,
                               'echo':echo_idx})
        
        tmp = df[np.in1d(df.inv, [1, 2]) & np.in1d(df.part, ['mag', 'phase']) & ((df.echo == 1))]
        

        check = (len(tmp) == 4) & (tmp.groupby(['inv', 'part']).size() == 1).all()
        
        if not check:
            raise ValueError('Did not find exactly one Magnitude and phase image for two' \
                             'inversions. Only found: %s' % tmp.fn.tolist())
        
        
        df = df.set_index(['inv', 'part', 'echo'])
        df.sort_index(inplace=True)
        
        inv1 = df.loc[1, 'mag', 1].fn
        inv1ph = df.loc[1, 'phase', 1].fn
        inv2 = df.loc[2, 'mag', 1].fn
        inv2ph = df.loc[2, 'phase', 1].fn

        print('Found following files for MP2RAGE:\n * inv1, magnitude: {inv1}\n * inv1, phase: {inv1ph}'\
              '\n * inv2, magnitude: {inv2}\n * inv2, phase: {inv2ph}'.format(**locals()))
        
        
        echo_indices = df.index.get_level_values(2).unique()
        print('Found four echoes:')
        for echo in echo_indices:
            print(' * {}'.format(df.loc[2, 'mag', echo].fn))
        
        meta_inv1 = layout.get_metadata(inv1)
        meta_inv2 = layout.get_metadata(inv2)
        
        for key in ['InversionRepetitionTime', 'NumberShots', 'PartialFourier']:
            if key in meta_inv1:
                if meta_inv1[key] != meta_inv2[key]:
                    raise ValueError('%s of INV1 and INV2 are different!' % key)        
        
        MPRAGE_tr = meta_inv1['InversionRepetitionTime']    
        invtimesAB = [meta_inv1['InversionTime'], meta_inv2['InversionTime']]    
        flipangleABdegree = [meta_inv1['FlipAngle'], meta_inv2['FlipAngle']]
        
        if 'PartialFourier' in meta_inv1.keys():
            nZslices = meta_inv1['NumberShots'] * np.array([meta_inv1['PartialFourier'] -.5, 0.5])    
        else: 
            nZslices = meta_inv1['NumberShots']
            
        FLASH_tr = [meta_inv1['ExcitationRepetitionTime'], meta_inv2['ExcitationRepetitionTime']]
        
        B0 = meta_inv1.pop('FieldStrength', 7)
        
        
        echo_times = []
            
        for echo in echo_indices:
            te = layout.get_metadata(df.loc[2, 'mag', echo].fn)['EchoTime']
            echo_times.append(te)
        
        return cls(echo_times,
                   MPRAGE_tr,
                   invtimesAB,
                   flipangleABdegree,
                   nZslices,
                   FLASH_tr,
                   inv1=inv1,
                   inv1ph=inv1ph,
                   inv2=df.loc[2, 'mag'].fn.tolist(),
                   inv2ph=df.loc[2, 'phase'].fn.tolist())
