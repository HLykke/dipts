__all__ = ['TimeSeries'
    , 'DipImage']

import sys
from typing import List, Tuple
import pyfftw

import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, rotate
from scipy.signal import convolve2d, medfilt2d

plt.rcParams['figure.dpi'] = 350
plt.rcParams['image.cmap'] = 'gray'


class TimeSeries(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        obj = np.asarray(input_array, *args, **kwargs).view(cls)
        # obj.info = info
        obj.dt = 1
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

    def fft(self, pad=None, return_f=False, shift=True):
        f = np.fft.fftfreq(len(self), d=self.dt)
        if pad is not None:
            hat = np.fft.fft(self, n=pad) #((pad+len(self))//2)*2)
        else:
            hat = np.fft.fft(self)
        if shift:
            hat = np.fft.fftshift(hat)
        if return_f:
            return f, TimeSeries(hat)
        return TimeSeries(abs(hat))

    def ifft(self, return_t=False):
        t = np.arange(len(self))
        x = np.fft.ifftshift(self)
        x = np.fft.ifft(x)
        if return_t:
            return t, TimeSeries(x)
        return TimeSeries(x)

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

    def polyreg(self, dim=1):
        # Todo: STA,
        raise NotImplementedError

    def linreg(self):
        # Todo: STA, use polyreg(1)
        raise NotImplementedError

    def Wb(self, length=None, dt=None):
        if length is None:
            length = len(self)
        if dt is None:
            dt = self.dt
        f = np.fft.fftfreq(length, d=self.dt)
        N = len(self)
        res = self.dt**2/N * np.sin(N*np.pi*f*self.dt)**2/np.sin(np.pi*f*self.dt)**2
        return res

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'dt', None)


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
        if np.any(self.imag):
            im = self.astype(np.complex)
        else:
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
        counts, binedges = np.histogram(self, bins=nbins)  # bins, r
        im = self.im2uint8()
        bins = np.arange(nbins)

        if show:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self, 'gray')
            ax[1].stem(bins, counts, markerfmt=' ')
            plt.show()
        return bins, counts  # bins, f(r)

    def histogram_equalization(self, L=256, Pz=None, show_hist: bool = False, inplace: bool = False):
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
            plt.stem(np.arange(L), counts, markerfmt=' ', linefmt='red')
            # plt.show()
            plt.stem(np.arange(L), new_counts, markerfmt=' ')
            plt.show()

        im_T = DipImage(np.zeros_like(self))
        im = self.im2uint8()
        for i in range(L):
            im_T[im == bins[i]] = new_bins[i]

        if inplace:
            self[:] = im_T
        return im_T

    def histogram_conversion(self, Ps= lambda x: x**2, L=256, show_hist: bool = False):  # Todo
        """Pz is the pdf to convert to."""
        ps_cum = np.cumsum(Ps(np.arange(L)))
        # print(ps_cum)
        raise NotImplementedError()

    def rotate(self, angle=90, inplace=False):
        im = rotate(self, angle=angle)
        if inplace:
            self[:] = im
        return DipImage(im)

    def gamma_transform(self, gamma, c=1):
        """Gamma transforms image to c*self^gamma."""
        if max(np.unique(self)) > 1:
            self = self.im2double()
            new_im = self.gamma_transform(gamma, c)
            return new_im.im2uint8()
        return DipImage(c * self ** gamma)

    def intensity_stretch(self, L=256, minimum=0, inplace=False):
        """Puts intensities ndarray into range minimum-L."""
        new_im = ((self - np.min(self)) / (np.max(self) - np.min(self)) * (L - 1)) + minimum
        if inplace:
            self[:] = new_im
        return DipImage(new_im)

    def blur(self, sfilter=None, filter_side_length=None, inplace=False, sigma=5):
        """Blurs image by averaging. Gaussian by default, specify std_dev of filter with sigma. """
        if filter_side_length is not None or sfilter is not None:
            if sfilter is None:
                sfilter = np.ones((filter_side_length, filter_side_length))
            blurred = convolve2d(self.im2double(), sfilter, mode='same')
            new_image = DipImage(blurred)
            new_image.intensity_stretch(inplace=True)
        else:
            new_image = gaussian_filter(self, sigma=sigma)
        if inplace:
            self[:] = new_image

        return new_image

    def unsharpen(self, sfilter=None, alpha=0.2, sigma=5, return_diff: bool = False):
        """sharp_parts = image - blurred_image. sfilter or filter_side_length are specifications for the blurring.
        sharpened_image = image + alpha*sharp_parts. """
        shp = (self - self.blur(sfilter=sfilter, sigma=sigma))
        # fig, ax = plt.subplots(1, 1)
        # p_im = ax.imshow(shp, 'gray', vmin=0)
        # plt.colorbar(p_im)
        # plt.show()
        image = (self + alpha * shp).intensity_stretch()
        if return_diff:
            return image, shp
        return image

    def _laplace(self, sfilter=None, sigma=1, mode='gaussian laplace', inplace=False):

        if mode == 'gaussian laplace':
            im = gaussian_laplace(-self, sigma=sigma)
            # plt.imshow(im)
            # plt.show()

        elif sfilter is None:
            if mode == '1':
                sfilter = np.array([[0, 1, 0]
                                       , [1, -4, 1]
                                       , [0, 1, 0]])
            elif mode == '2':
                sfilter = np.array([[1, 1, 1]
                                       , [1, -8, 1]
                                       , [1, 1, 1]])


            else:
                raise ValueError("If not specified sfilter, mode must be either '1', '2' or 'gaussian laplace'.")

            im = convolve2d(self, sfilter, mode='same')

        if inplace:
            self[:] = im
        return DipImage(im)

    def laplace_sharpen(self, alpha=0.2, return_diff: bool = False, inplace=False, **kwargs):
        shp = self._laplace(**kwargs)
        # plt.imshow(shp)
        # plt.show()
        sharpened = self - alpha * shp
        image = sharpened.intensity_stretch()
        if inplace:
            self[:] = image
        if return_diff:
            return image, shp
        return image

    def median_filter(self, filter_side_length=3, inplace=True):
        a = medfilt2d(self, filter_side_length)
        if inplace:
            self[:] = a
        return a

    def impad(self, pad=None, inplace=True):
        """Zero-padding of image. Pads zeros only after image in both directions.
        By default pads smallest number of zeros that is greater than 2*image,
        but is a 2-expential. Ex: An image of (3,14) --> (8, 32). """
        oldshape = np.array(self.shape)  # Make it list to be able to change the values
        if pad is None:
            # The smallest size larger than double the original size in both directions that is a 2-exponential.
            newshape = np.int32(2**(np.ceil(np.log2(oldshape * 2)))) - self.shape
            # print(newshape + self.shape)
            im = np.pad(self
                        , ((0, newshape[0]), (0, newshape[1]))
                        , mode='constant'
                        , constant_values=(0, 0))  # np.pad(self, ((up, down), (left, right)))
        else:
            if type(pad) == tuple:
                im = np.pad(self, ((0, pad[0]), (0, pad[1])), mode='constant', constant_values=0)
            im = np.pad(self, pad, mode='constant', constant_values=0)
        # testimage = DipImage(im)
        # testimage.show()
        image = DipImage(im)
        # if inplace:
        #     self[:] = image
        return image

    def fft(self, inplace=False, log=False):
        im = self.im2double()
        if self.ndim == 1:
            im_hat = np.fft.fft(im)
        elif self.ndim == 2:
            im_hat = np.fft.fft2(im)
        # Alternative way to shift is:
        # DFT[f(u, v)*(-1)**(u+v)]
        im_hat = np.fft.fftshift(im_hat)
        if log:
            im_hat = np.log(np.abs(im_hat))
        image = DipImage(im_hat)
        if inplace:
            self[:] = image
        return image

    def ifft(self, inplace=False):
        im = self.im2double()
        im = np.fft.ifftshift(self)
        if self.ndim == 1:
            im = np.fft.ifft(im)
        elif self.ndim == 2:
            im = np.fft.ifft2(im)
        image = DipImage(im)
        if inplace:
            self[:] = image
        return image

    def lowpass(self, d=None, sfilter=None, mode='gaussian', sigma=1, zone_plate=False):
        """Lowpass-filter of self. Self must be in time-domain.
        d is distance from center in pixels. Isotropic filter.
        Chose either a mode or specify a filter.

        Possible modes:
            - gaussian
            - ideal
        """
        if mode == 'gaussian':
            # todo: Make bandpass here. Isolate mask so zone_plate can be shown.
            mask = np.ones_like(self)
            mask = DipImage(gaussian_filter(mask.fft().real, sigma=10))
            image = DipImage(mask.fft()*self.fft()).ifft()
            mask.show()
            return DipImage(image)
        return self.bandpass(band=(0, d), sfilter=sfilter, mode=mode, zone_plate=zone_plate)

    def highpass(self, d, sfilter=None, mode='gaussian', zone_plate=False):
        # Todo: Gaussian, blackman?, zone_plate
        """Highpass-filter of self. Self must be in time-domain.
        d is distance from center in pixels. Isotropic filter.
        Chose either a mode or specify a filter.

        Possible modes:
            - gaussian
            - ideal
        """
        if mode == 'gaussian':
            # todo: Make bandpass here. Isolate mask so zone_plate can be shown.
            image = gaussian_filter(self, sigma=sigma)
            return DipImage(image)
        return self.bandpass(band=(d, np.inf), sfilter=sfilter, mode=mode, zone_plate=zone_plate)

    def bandpass(self, band: tuple, shape=None, sfilter = None
                 , mode: str = 'gaussian', sigma=1, zone_plate=False, inplace=False):
        """Band pass filter.

        :param band: Range of what frequencies to let pass thought.
        :param sfilter: If you wish, you may specify your own filter.
        :param mode: 'gaussian' or 'ideal'.
        :param zone_plate: Show what effect the filter has on a zone_plate.
        :return: the filtered version of self.
        """
        # Todo: Gaussian, blackman?, zone_plate

        # Signal to be modified
        dft = self.fft()

        if shape is None:
            shape = self.shape
        if mode=='ideal':
            lenx = shape[0]//2
            leny = shape[1]//2
            print(lenx, leny)
            y, x = np.ogrid[-lenx: lenx, -leny: leny]
            # print(y, x)
            if band is None:
                raise ValueError('When mode "gaussian" or "ideal" is chosen'
                                 ', the radius-parameter, band, must be specified.')
            # Making filter mask. 00000011111111000000000 in radius.
            sfilter = (band[0] ** 2 <= x ** 2 + y ** 2) & (x ** 2 + y ** 2 <= band[1]**2)

        image = DipImage(dft * sfilter).ifft()

        if mode == 'gaussian':
            # todo: Make bandpass here. Isolate mask so zone_plate can be shown.
            mask = np.ones_like(self)
            mask = ~gaussian_filter(mask, sigma=sigma)
            print(mask)
            # raise NotImplementedError

        if zone_plate:
            sfilter = DipImage(sfilter)
            print(sfilter.shape)
            sfilter.show()
            sfilter.zoneplate()
        if inplace:
            self[:] = np.abs(image)
        return image

    def zoneplate(self, shape=None):
        """See how a filter behaves. Let self be the filter of interest.
        Specify the shape of the desired zone-plate.

        Example:
            f = DipImage(gaussian_filter(f, 1.3))
            f.zoneplate(shape=(256, 256))

        """
        def z(x, y):
            return 1/2*(1 + np.cos(x**2 + y**2))
        if shape is None:
            m, n = self.shape
        else:
            m, n = shape
        y, x = np.linspace(-10, 10, m), np.linspace(-10, 10, n)
        meshs = np.meshgrid(x, y)

        blank_plate = DipImage(np.abs(z(*meshs)))

        # dft = self.fft()
        plate = np.abs(self * blank_plate)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.abs(blank_plate))
        ax[1].imshow(np.abs(plate))
        plt.show()

    def show(self):
        if np.count_nonzero(self.imag):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.real)
            ax[1].imshow(self.imag)
            plt.show()
        else:
            plt.imshow(self.real, cmap='gray')
            plt.show()

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # self.info = getattr(obj, 'info', None)


def example():
    im = imageio.imread(
        '/home/harald/Documents/UiT/UiT_Courses/FYS-2010/data/ch2/'
        'DIP3E_Original_Images_CH02/Fig0222(a)(face).tif')

    im = DipImage(im)
    # gamma transform
    for j, i in enumerate([1, 0.6, 0.4, 0.2]):
        plt.subplot(2, 2, j+1)
        plt.imshow(im.gamma_transform(i).im2double(), vmax=1, vmin=0)
    plt.show()

    # plt.imshow(im)
    # plt.show()
    # im = imageio.imread('/home/harald/Documents/UiT/UiT_Courses/'
    #                     'FYS-2010/data/ch3/DIP3E_Original_Images_CH03/Fig0316(2)(2nd_from_top).tif')
    # im = imageio.imread('/home/harald/Documents/UiT/UiT_Courses/'
    #                     'FYS-2010/data/ch3/DIP3E_Original_Images_CH03/Fig0314(a)(100-dollars).tif')
    # image = DipImage(im)
    # print(type(image))
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

    # im2 = image.histogram_equalization(show_hist=False)
    # # im2.histogram(show=True)
    # im3 = im2.blur(filter_side_length=2)
    # im4 = im2.unsharpen(filter_side_length=3, alpha=0.5)
    # im5 = im2.laplace_sharpen(mode='2', alpha=0.05)
    #
    # fig, ax = plt.subplots(1, 2)
    # plt_im1 = ax[0].imshow(im2, 'gray', vmin=0, vmax=255)
    # plt_im2 = ax[1].imshow(im5, 'gray', vmin=0, vmax=255)
    # # plt.colorbar(plt_im1)
    # # plt.colorbar(plt_im2)
    # plt.show()


def t_DipImage():
    im = imageio.imread(
        '/home/harald/Documents/UiT/UiT_Courses/FYS-2010/data/ch4/Fig0441(a)(characters_test_pattern).tif')
    im = DipImage(im)
    im.show()
    # im.show()
    # im = im.impad().fft(log=True)
    # im.show()

    # bandpassed = im.bandpass(band=(50, 150), mode='ideal')
    # bandpassed.show()
    im.lowpass(sigma=3)


def t_TimeSeries():
    # Perfect one
    f = 5
    x = TimeSeries(np.sin(2*np.pi*f*np.linspace(0, 1, 2**12)))
    y = TimeSeries(np.sin(2*np.pi*f*np.linspace(0, 1, 10)))
    t = np.arange(2**12)
    # wb = y.fft(shift=False).Wb(length=2**14)
    # plt.plot(t, x.fft(shift=True)/2**14)
    plt.plot(t, y.fft(pad=2**12))
    # plt.plot(wb)
    # a = x.fft().Wb(length=2000)
    # plt.plot(a)
    plt.show()


if __name__ == '__main__':
    # example()
    t_DipImage()
    # t_TimeSeries()