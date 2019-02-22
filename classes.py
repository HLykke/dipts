__all__ = ['TimeSeries'
        ,'DipImage']

from typing import List, Tuple
import imageio
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import convolve2d

plt.rcParams['figure.dpi'] = 350


class TimeSeries(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        obj = np.asarray(input_array, *args, **kwargs).view(cls)
        # obj.info = info
        return obj

    def acvf(self, max_shift: int = None):
        """
        Auto covariance function.
        :param max_shift: M
        :return:
        """
        n = len(self)
        xmean = np.nanmean(self)
        if max_shift is None:
            max_shift = int(n - 1)
        res = TimeSeries(np.zeros(max_shift))
        for i, h in enumerate(np.arange(max_shift)):
            res[i] = (self[h:] - xmean) @ (self[:n - h] - xmean) / (n - h)
        return res

    def acf(self, max_shift: int = None):
    """
    Auto correlation function.
    :param max_shift: (int). Cannot be larger than len(x).
    :return: (np.ndarray of shape (1, n))
    """
    _acvf = self.acvf(max_shift)
    return TimeSeries(_acvf / _acvf[0])

    def plot(self, x=None, xlim: List[int] = None, ylim: List[int] = None, clear=True, *args, **kwargs):
        """Basic plot just to be able to see whats going on. """
        if 'matplotlib.pyplot' not in sys.modules:
            pass
        fig = plt.figure()

        if clear:
            fig.clf()

        if x is not None:
            plt.plot(self, x, *args, **kwargs)
        else:
            plt.plot(self, *args, **kwargs)

        if xlim is not None:
            plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        plt.show()
        plt.close(fig)

    def linreg(self):
        pass

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # self.info = getattr(obj, 'info', None)


class DipImage(np.ndarray):
    """Class with functions learned in FYS-2010: Digital image processing. """

    def __new__(cls, input_array, *args, **kwargs):
        obj = np.asarray(input_array, *args, **kwargs).view(cls)
        # obj.info = info
        return obj

    def im2double(self):
        """
        Normalizes all elements in im (np.ndarray) to range [0, 1] and converts to float64.
        """
        im = self.astype(np.float64)
        if max(np.unique(im)) > 1:
            return im / 255
        im = DipImage(im)
        return im

    def im2uint8(self):
        new = self.copy()
        if max(np.unique(self)) <= 1:
            new = self * 255
        return new.astype(np.uint8)

    def histogram(self, nbins=256, show: bool = False) -> Tuple[np.ndarray]:
        """returns bins and the count of pixels in each bin.
        If show set to true, the image with a histogram plot is also shown."""
        counts, binedges = np.histogram(self, nbins)  # bins, r
        im = self.im2uint8()
        bins = np.arange(nbins)

        if show:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self, 'gray')
            ax[1].stem(bins, counts, markerfmt=' ')
            plt.show()
        return bins, counts  # bins, f(r)

    def histogram_equalization(self, L=256, show_hist: bool = False):
        """Transformation of histogram of image(r) to uniform pdf."""
        bins, counts = self.histogram(nbins=L)
        cumu_counts = np.cumsum(counts)

        # normalized so that sum_i p(i) = 1
        # and s = (L-1) cumsum(p(r))
        new_bins = ((L - 1) * cumu_counts) // cumu_counts[-1]

        new_counts = np.zeros_like(counts)
        new_counts[new_bins] = counts[bins]
        # new_counts[~new_counts] = 0
        if show_hist:
            plt.stem(np.arange(L), counts, markerfmt=' ')
            plt.show()
            plt.stem(np.arange(L), new_counts, markerfmt=' ')
            plt.show()

        im_T = DipImage(np.zeros_like(self))
        im = self.im2uint8()
        for i in range(L):
            im_T[im == bins[i]] = new_bins[i]
        return im_T

    def hist_conversion(self, Pz, L=256, show_hist: bool = False):
        """Pz is the pdf to convert to."""
        raise NotImplementedError()

    def gamma_transform(self, gamma, c=1):
        if max(np.unique(self)) > 1:
            self = self.im2double()
            new_im = self.gamma_transform(gamma, c)
            return new_im.im2uint8()
        return DipImage(c * self ** gamma)

    def intensity_stretch(self, L=256, minimum=0, inplace=False):
        new_im = ((self - np.min(self)) / (np.max(self) - np.min(self)) * (L-1)) + minimum
        if inplace:
            self[:] = new_im
        return DipImage(new_im)

    def blur(self, sfilter=None, filter_side_length=5, inplace=False):
        if sfilter is None:
            sfilter = np.ones((filter_side_length, filter_side_length))
        blurred = convolve2d(self.im2double(), sfilter, mode='same')
        new_image = DipImage(blurred)
        new_image.intensity_stretch(inplace=True)
        if inplace:
            self[:] = new_image
        return new_image

    def unsharpen(self, sfilter=None, alpha=0.2, filter_side_length=5):
        """sharp_parts = image - blurred_image. sfilter or filter_side_length are specifications for the blurring.
        sharpened_image = image + alpha*sharp_parts. """
        shp = (self - self.blur(sfilter, filter_side_length))
        fig, ax = plt.subplots(1, 1)
        p_im = ax.imshow(shp, 'gray', vmin=0)
        plt.colorbar(p_im)
        plt.show()
        return (self + alpha * shp).intensity_stretch()

    def _laplace(self, sfilter=None, mode='1', inplace=False):
        if sfilter is None:
            if mode == '1':
                sfilter = np.array([[0, 1, 0]
                                 , [1,-4, 1]
                                 , [0, 1, 0]])
            elif mode == '2':
                sfilter = np.array([[1, 1, 1]
                                , [1, -8, 1]
                                , [1, 1, 1]])
            else:
                raise ValueError("If not specified sfilter, mode must be either '1' or '2'.")

        im = convolve2d(self, sfilter, mode='same')
        if inplace:
            self[:] = im
        return DipImage(im)

    def laplace_sharpen(self, alpha=0.1, **kwargs):
        shp = self._laplace(**kwargs)
        sharpened = self - alpha * shp
        return DipImage(sharpened).intensity_stretch()

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # self.info = getattr(obj, 'info', None)


def example():
    im = imageio.imread(
        '/home/harald/Documents/UiT/UiT_Courses/FYS-2010/data/ch2/'
        'DIP3E_Original_Images_CH02/Fig0222(a)(face).tif')
    # im = imageio.imread('/home/harald/Documents/UiT/UiT_Courses/'
    #                     'FYS-2010/data/ch3/DIP3E_Original_Images_CH03/Fig0316(2)(2nd_from_top).tif')
    # im = imageio.imread('/home/harald/Documents/UiT/UiT_Courses/'
    #                     'FYS-2010/data/ch3/DIP3E_Original_Images_CH03/Fig0314(a)(100-dollars).tif')
    image = DipImage(im)
    image = image.im2double()
    #
    # # testing histogram
    # image.histogram(show=True)
    #
    # # testing histogram_equalization
    # im2 = image.histogram_equalization(show_hist=True)
    # plt.subplot(121)
    # plt.imshow(image, 'gray')
    # plt.subplot(122)
    # plt.imshow(im2, 'gray')
    # plt.show()

    im2 = image.histogram_equalization(show_hist=False)
    # im2.histogram(show=True)
    im3 = im2.blur(filter_side_length=2)
    im4 = im2.unsharpen(filter_side_length=3, alpha=0.5)
    im5 = im2.laplace_sharpen(mode='2', alpha=0.05)

    fig, ax = plt.subplots(1, 2)
    plt_im1 = ax[0].imshow(im2, 'gray', vmin=0, vmax=255)
    plt_im2 = ax[1].imshow(im5, 'gray', vmin=0, vmax=255)
    # plt.colorbar(plt_im1)
    # plt.colorbar(plt_im2)
    plt.show()


# def test_DipImage():
#     pass


if __name__ == '__main__':
    example()
