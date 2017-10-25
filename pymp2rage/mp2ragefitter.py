import nibabel as nb
from nilearn import image
import numpy as np


class MP2RAGEFitter(object):


    def __init__(self, inv1_combined=None, inv2_combined=None, inv1=None, inv1ph=None, inv2=None, inv2ph=None):


        if inv1_combined is not None:
            inv1_combined = image.load_img(inv1_combined, dtype=np.double)

            if inv1_combined.shape[3] != 2:
                raise Exception('inv1_combined should contain two volumes')

            if (inv1 is not None) or (inv1ph is not None):
                raise Exception('*Either* give inv1_combined *or* inv1 and inv1_ph.')

            self.inv1 = image.index_img(inv2_combined, 0)
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


    def fit_mp2rage(self):
        compINV1 = self.inv1.get_data() * np.exp(self.inv1ph.get_data() * 1j)
        compINV2 = self.inv2.get_data() * np.exp(self.inv2ph.get_data() * 1j)

        # Scale to 4095
        self.MP2RAGE = (np.real(compINV1*compINV2/(compINV1**2 + compINV2**2)))*4095+2048

        # Clip anything outside of range
        self.MP2RAGE = np.clip(self.MP2RAGE, 0, 4095)

        # Convert to nifti-image
        self.MP2RAGE = nb.Nifti1Image(self.MP2RAGE, self.inv1.affine)


        return self.MP2RAGE


    @staticmethod
    def MPRAGEfunc_varyingTR(nimages, MPRAGE_tr, inversiontimes, nZslices, 
                              FLASH_tr, flipangle, sequence, T1s, 
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

    @staticmethod
    def MP2RAGE_lookuptable(nimages, MPRAGE_tr, invtimesAB, flipangleABdegree, nZslices, FLASH_tr, 
                         sequence, inversion_efficiency=0.96, all_data=0):
    # first extra parameter is the inversion efficiency
    # second extra parameter is the alldata
    #   if ==1 all data is shown
    #   if ==0 only the monotonic part is shown

    
    
        invtimesa, invtimesb = invtimesAB
        B1vector = 1

        flipanglea, flipangleb = flipangleABdegree

        T1vector = np.arange(0.05, 4.05, 0.05)



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
                Signal[j, :] = MP2RAGEFitter.MPRAGEfunc_varyingTR(nimages, MPRAGE_tr, invtimesAB, nZslices2, FLASH_tr, [flipanglea, flipangleb], sequence, T1)


            else:
                Signal[j,:] = 0


        Intensity = np.squeeze(np.real(Signal[..., 0] * np.conj(Signal[..., 1])) / (np.abs(Signal[... ,0])**2 + np.abs(Signal[...,1])**2))

        if all_data == 0:
            minindex = np.argmax(Intensity)
            maxindex = np.argmin(Intensity)
            Intensity = Intensity[minindex:maxindex]
            T1vector = T1vector[minindex:maxindex];
            IntensityBeforeComb = Signal[minindex:maxindex]
        else:
            IntensityBeforeComb = Signal
        return Intensity, T1vector, IntensityBeforeComb
