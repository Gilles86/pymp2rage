import nibabel as nb
from nilearn import image


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
