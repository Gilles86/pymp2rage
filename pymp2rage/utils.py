import numpy as np
import os.path as op


def MPRAGEfunc_varyingTR(MPRAGE_tr, inversiontimes, nZslices, 
                          FLASH_tr, flipangle, sequence, T1s, 
                          nimages=2,
                          B0=7, M0=1, inversionefficiency=0.96):

    if sequence == 'normal':
        normalsequence = True
        waterexcitation = False
    else:
        normalsequence = False
        waterexcitation = True

    nZslices = np.atleast_1d(nZslices)
    inversiontimes = np.atleast_1d(inversiontimes)
    FLASH_tr = np.atleast_1d(FLASH_tr)
    flipangle = np.atleast_1d(flipangle)

    FatWaterCSppm=3.3 # ppm
    gamma=42.576 #MHz/T
    pulseSpace=1/2/(FatWaterCSppm*B0*gamma) #

    fliprad = flipangle/180*np.pi

    if len(fliprad) != nimages:
        fliprad = np.repeat(fliprad, nimages)

    if len(FLASH_tr) != nimages:
        FLASH_tr = np.repeat(FLASH_tr, nimages)        

    if len(nZslices) == 2:
        nZ_bef=nZslices[0]
        nZ_aft=nZslices[1]
        nZslices=sum(nZslices);

    elif len(nZslices)==1:
        nZ_bef=nZslices / 2
        nZ_aft=nZslices / 2

    if normalsequence:
        E_1 = np.exp(-FLASH_tr / T1s)
        TA = nZslices * FLASH_tr
        TA_bef = nZ_bef * FLASH_tr
        TA_aft = nZ_aft * FLASH_tr

        TD = np.zeros(nimages+1)
        E_TD = np.zeros(nimages+1)

        TD[0] = inversiontimes[0]-TA_bef[0]
        E_TD[0] = np.exp(-TD[0] / T1s)

        TD[nimages] =MPRAGE_tr - inversiontimes[nimages-1] - TA_aft[-1]
        E_TD[nimages] = np.exp(-TD[nimages] / T1s)


        if nimages > 1:
            TD[1:nimages] = inversiontimes[1:] - inversiontimes[:-1] - (TA_aft[:-1] + TA_bef[1:])
            E_TD[1:nimages] = np.exp(-TD[1:nimages] / T1s)

        cosalfaE1 = np.cos(fliprad) * E_1    
        oneminusE1 = 1 - E_1
        sinalfa = np.sin(fliprad)

    MZsteadystate = 1. / (1 + inversionefficiency * (np.prod(cosalfaE1))**(nZslices) * np.prod(E_TD))

    MZsteadystatenumerator = M0 * (1 - E_TD[0])


    for i in np.arange(nimages):
        MZsteadystatenumerator = MZsteadystatenumerator*cosalfaE1[i]**nZslices + M0 * (1-E_1[i]) * (1-(cosalfaE1[i])**nZslices) / (1-cosalfaE1[i])        
        MZsteadystatenumerator = MZsteadystatenumerator*E_TD[i+1]+M0*(1-E_TD[i+1])

    MZsteadystate = MZsteadystate * MZsteadystatenumerator


    signal = np.zeros(nimages)

    m = 0
    temp = (-inversionefficiency*MZsteadystate*E_TD[m] + M0 * (1-E_TD[m])) * (cosalfaE1[m])**(nZ_bef) + \
           M0 * (1 - E_1[m]) * (1 - (cosalfaE1[m])**(nZ_bef)) \
           / (1-(cosalfaE1[m]))

    signal[0] = sinalfa[m] * temp


    for m in range(1, nimages):
        temp = temp * (cosalfaE1[m-1])**(nZ_aft) + \
               M0 * (1 - E_1[m-1]) * (1 - (cosalfaE1[m-1])**(nZ_aft)) \
              / (1-(cosalfaE1[m-1]))

        temp = (temp * E_TD[m] + M0 * (1 - E_TD[m])) * (cosalfaE1[m])**(nZ_bef) + \
               M0 * (1-E_1[m]) * (1 - (cosalfaE1[m])**(nZ_bef)) \
               / (1 - (cosalfaE1[m]))

        signal[m] = sinalfa[m]*temp

    return signal        


def MP2RAGE_lookuptable(MPRAGE_tr, invtimesAB, flipangleABdegree, nZslices, FLASH_tr, 
                     sequence, nimages=2, B0=7, M0=1, inversion_efficiency=0.96, all_data=0,
                        T1vector=None):
# first extra parameter is the inversion efficiency
# second extra parameter is the alldata
#   if ==1 all data is shown
#   if ==0 only the monotonic part is shown



    invtimesa, invtimesb = invtimesAB
    B1vector = 1

    flipanglea, flipangleb = flipangleABdegree

    if T1vector is None:
        T1vector = np.arange(0.05, 5.05, 0.05)

    FLASH_tr = np.atleast_1d(FLASH_tr)

    if len(FLASH_tr) == 1:
        FLASH_tr = np.repeat(FLASH_tr, nimages)


    nZslices = np.atleast_1d(nZslices)

    if len(nZslices)==2:        
        nZ_bef, nZ_aft = nZslices
        nZslices2 = np.sum(nZslices)

    elif len(nZslices) == 1:
        nZ_bef = nZ_aft = nZslices / 2
        nZslices2 = nZslices

    Signal = np.zeros((len(T1vector), 2))

    for j, T1 in enumerate(T1vector):
        if ((np.diff(invtimesAB) >= nZ_bef * FLASH_tr[1] + nZ_aft*FLASH_tr[0]) and \
           (invtimesa >= nZ_bef*FLASH_tr[0]) and \
           (invtimesb <= (MPRAGE_tr-nZ_aft*FLASH_tr[1]))):
            Signal[j, :] = MPRAGEfunc_varyingTR(MPRAGE_tr, invtimesAB, nZslices2, FLASH_tr, [flipanglea, flipangleb], sequence, T1, nimages, B0, M0, inversion_efficiency)


        else:
            Signal[j,:] = 0


    Intensity = np.squeeze(np.real(Signal[..., 0] * np.conj(Signal[..., 1])) / (np.abs(Signal[... ,0])**2 + np.abs(Signal[...,1])**2))

    if all_data == 0:
        minindex = np.argmax(Intensity)
        maxindex = np.argmin(Intensity)
        Intensity = Intensity[minindex:maxindex+1]
        T1vector = T1vector[minindex:maxindex+1]
        IntensityBeforeComb = Signal[minindex:maxindex+1]
    else:
        IntensityBeforeComb = Signal
    return Intensity, T1vector, IntensityBeforeComb


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.
    Parameters
    ----------
    fname : str
        file or path name
    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname
    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'
    >>> fname
    'subject'
    >>> ext
    '.nii.gz'
    """

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and \
                (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext
