__all__ = ['TimeSeries'
    , 'DipImage']

import sys
from typing import List, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, rotate
from scipy.signal import convolve2d, medfilt2d
from scipy import misc
from validation import validate_io_types

plt.rcParams['figure.dpi'] = 350
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True


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

    def ccvf(self, other, max_shift: int = None):
        """
        Cross covariance function.
        :param other: Timeseries.
        :param max_shift: M
        :return:
        """
        n = np.min(len(self), len(other))
        xmean = np.nanmean(self)
        ymean = np.nanmean(other)
        if max_shift is None:
            max_shift = int(n - 1)
        res = TimeSeries(np.zeros(max_shift))
        for i, h in enumerate(np.arange(max_shift)):
            res[i] = (self[h:] - xmean) @ (other[:n - h] - ymean) / (n - h)
        return res

    def ccf(self, other, max_shift: int = None):
        _ccvf = self.ccvf(other, max_shift=max_shift)
        return _ccvf / np.max(_ccvf)

    def fft(self, pad=None, return_f=False, shift=True):
        f = np.fft.fftfreq(len(self), d=self.dt)
        if pad is not None:
            hat = np.fft.fft(self, n=pad)  # ((pad+len(self))//2)*2)
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

    def plot(self, x=None, xlim: List[int] = None, ylim: List[int] = None
             , clear=True, domain='time'
             , xticksteps=5, xtickrotation=0, *args, **kwargs):
        """Basic plot just to be able to see whats going on. """
        if 'matplotlib.pyplot' not in sys.modules:
            pass
        fig = plt.figure()

        if clear:
            fig.clf()

        if x is not None:
            plt.plot(x, self, *args, **kwargs)
        else:
            plt.plot(self, *args, **kwargs)

        if xlim is not None:
            plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        steps = xticksteps
        if domain == 'omega':
            myxticks = []
            for i in range(-steps // 2 + 1, steps // 2 + 1):
                pi = r'\pi'
                if abs(i) == 1:
                    i = ''
                myxticks.append(r"$\frac{%s %s}{%s}$" % (i, pi, steps // 2))
            myxticks[steps // 2] = 0
            myxticks[0] = r"$-\pi$"
            myxticks[-1] = r"$\pi$"

            plt.xticks(np.linspace(0, len(self), steps), myxticks, rotation=xtickrotation)

        elif domain == 'normal':
            print()
            pass

        elif domain == 'freq':
            # print("self.dt = ", self.dt)
            plt.xticks(np.linspace(0, len(self), steps)
                       , np.linspace(-1 / (2 * self.dt), 1 / (2 * self.dt), len(self)), rotation=xtickrotation)

        plt.show()
        plt.close(fig)

    def stem(self):
        plt.stem(self)
        plt.show()

    def polyreg(self, t=None, deg=1):
        # Todo: STA,
        if t is None:
            t = np.arange(len(self))
        p = np.polyfit(t, self, deg=deg)
        y_hat = np.polyval(p, t)
        return y_hat
        # raise NotImplementedError

    def linreg(self, t=None):
        # Todo: STA, use polyreg(1)
        y_hat = self.polyreg(t)
        return y_hat

    def periodogram(self, dt=1, w=None):
        """Estimate of Power spectral density of x.
        Optional to add window w."""
        N = len(self)
        if w is None:
            w = 1
            U = 1
        elif w == 'hann':
            w = np.hann(len(self))
        elif w == 'hamming':
            w = np.hamming(len(self))
        else:
            U = 1 / N * np.sum(w ** 2)

        S = dt / (N * U) * np.abs(np.fft.fft(w * self)) ** 2
        S = np.fft.fftshift(S)
        return TimeSeries(S)

    def wosa(self, m: int, overlap=0.5):
        """Weighted overlapped segment averaging of x.
        Which is an estimator of Power spectral density of x.

        :param x: Time series
        :param m: Segment length
        :param overlap: (float in interval [0, 1]) Segment overlap.
        :return: (np.ndarray). S_estimate_wosa.
        """
        # todo: zeropad instead of cutting away the end.
        N = len(self)
        print(N)
        # soln. to n((1-overlap)m)+overlap*m < N wrt n where n is int.
        n_seg = int((N - (m * overlap)) / (m * (1 - overlap)))  # Number of segments
        segments = np.zeros((m, n_seg))
        window = np.hamming(m)
        for i in range(n_seg):
            start = int((1 - overlap) * m * i)
            segments[:, i] = S_per(self[start: start + m], w=window)
        result = np.sum(segments, axis=1)
        return result

    def Wb(self, length=None, dt=None):
        if length is None:
            length = len(self)
        if dt is None:
            dt = self.dt
        f = np.fft.fftfreq(length, d=self.dt)
        N = len(self)
        res = self.dt ** 2 / N * np.sin(N * np.pi * f * self.dt) ** 2 / np.sin(np.pi * f * self.dt) ** 2
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

    def resize(self, new_shape, mode='nearest'):
        """Resize an image using different methods.
        The options are: 'nearest'
                    (Todo: 'linear' or 'cubic')

        Nearest neighbor works a little weird because of the way numpy.round and numpy.rint works.
        If a number is in the middle between two integers it will be rounded to the
        nearest even number, not away from 0 (which is the normal mathematical rule).
        There are good motivations for this, but it works strange in this case.
        """
        # Tuples are immutable, but I need to do calculations with them.
        old_shape = np.array(self.shape)
        new_shape = np.array(new_shape)

        image = np.ones(new_shape)*-1
        scaling = new_shape / old_shape

        image = np.zeros(new_shape)
        i_x, i_y = np.mgrid[0:new_shape[0], 0:new_shape[1]]

        # This is where things get strange because of round-to-nearest-even rule of numpy.
        i_x = np.round(i_x* 1/scaling[0])
        i_y = np.round(i_y* 1/scaling[1])
        i_x = (i_x).astype(int)
        i_y = (i_y).astype(int)
        # To make sure no inices are above the highest possible one.
        i_x[i_x >= old_shape[0]] = old_shape[0] - 1
        i_y[i_y >= old_shape[1]] = old_shape[1] - 1
        image = self[i_x, i_y]
        return image

    def im2double(self):
        """
        Normalizes all elements in im (np.ndarray) to range [0, 1] and converts to float64.
        """
        if np.any(self.imag):
            im = self.astype(np.complex)
        else:
            im = self.real
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

    def histogram_conversion(self, Ps=lambda x: x ** 2, L=256, show_hist: bool = False):  # Todo
        """Pz is the pdf to convert to."""
        ps_cum = np.cumsum(Ps(np.arange(L)))
        # print(ps_cum)
        raise NotImplementedError()

    def rotate(self, angle=90, inplace=False):
        im = rotate(self, angle=angle, reshape=False)
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

    def impad(self, pad=None):
        """Zero-padding of image. Pads zeros only after image in both directions.
        By default pads smallest number of zeros that is greater than 2*image,
        but is a 2-expential. Ex: An image of (3,14) --> (8, 32). """
        oldshape = np.array(self.shape)  # Make it list to be able to change the values
        if pad is None:
            # The smallest size larger than double the original size in both directions that is a 2-exponential.
            newshape = np.int32(2 ** (np.ceil(np.log2(oldshape * 2)))) - self.shape
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

        # Does not work because self is of different shape
        # if inplace:
        #     self[:] = image
        return image

    def fft(self, pad=None, inplace=False, log=False):
        """Pads by default to smallest 2-exponentials larger than 2*im.shape."""
        im = self.im2double()
        if pad is not False:
            im = im.impad(pad)
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
            mask = DipImage(gaussian_filter(mask.fft().real, sigma=sigma))
            image = DipImage(mask.fft() * self.fft()).ifft()
            mask.show()
            return DipImage(image)
        return self.bandpass(band=(0, d), sfilter=sfilter, mode=mode, zone_plate=zone_plate)

    def highpass(self, d=None, sfilter=None, mode='gaussian', sigma=1, zone_plate=False, return_mask=False):
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
            mask = np.ones_like(self)
            mask = DipImage(1 - gaussian_filter(mask.fft().real, sigma=sigma))
            image = DipImage(mask.fft() * self.fft()).ifft()
            mask.show()
            if return_mask:
                return DipImage(image), mask
            else:
                return DipImage(image)
        return self.bandpass(band=(d, np.inf), sfilter=sfilter, mode=mode, zone_plate=zone_plate,
                             return_mask=return_mask)

    def bandpass(self, band: Tuple[float, float] = (0, np.inf), pad: Tuple = None
                 , sfilter=None, return_mask=False, mode: str = 'gaussian', zone_plate=False
                 , inplace=False, **kwargs):
        """Band pass filter.

        :param pad: Default is padding lowest 2-exponent larger than
                douple the axis length in both x and y directions. Set False for no padding.
        :param band: Range of what frequencies to let pass thought.
        :param sfilter: If you wish, you may specify your own filter.
        :param mode: 'gaussian' or 'ideal'.
        :param zone_plate: Show what effect the filter has on a zone_plate.
        :param sigma: (tuple). Sigma of highpass and sigma of lowpass. (....|-----|....).
        :param inplace: If True, replaces the current image with the bandpassed image.
        :param kwargs: Passed on to _crosscar (params rx and ry) or to _mnormal (params cov, mean)
        :return: the filtered version of self.
        """
        # Todo: zone_plate
        # Signal to be modified
        dft = self.fft()

        if pad is None:
            shape = dft.shape

        # boxcar filter
        if mode == 'ideal':
            mask = self._boxcar(*band, shape=shape, **kwargs)

        if mode == 'crosscar':
            mask = self._crosscar(shape=shape, **kwargs)

        if mode == 'gaussian':
            # mask = self._mnormal(shape, **kwargs)
            mask = self._normal(shape=shape, **kwargs)

        if mode == 'butterworth':
            mask = self._butterworth(*band, shape=shape, **kwargs)

        if zone_plate:
            sfilter = DipImage(sfilter)
            print(sfilter.shape)
            sfilter.show()
            sfilter.zoneplate()

        image = DipImage(dft * mask).ifft()
        m, n = self.shape
        image = image[:m, :n]

        if inplace:
            self[:] = image.real

        if return_mask:
            return image.real, mask
        else:
            return image.real

    def zoneplate(self, shape=None):
        """See how a filter behaves. Let self be the filter of interest.
        Specify the shape of the desired zone-plate.

        Example:
            f = DipImage(gaussian_filter(f, 1.3))
            f.zoneplate(shape=(256, 256))

        """

        def z(x, y):
            return 1 / 2 * (1 + np.cos(x ** 2 + y ** 2))

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

    def show(self, toscreen: bool=True, save: str=None):
        if np.count_nonzero(self.imag):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.real)
            ax[1].imshow(self.imag)
        else:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(self.real, cmap='gray')
        if toscreen:
            plt.show()
        if save is not None:
            plt.savefig(save)
        plt.close(fig)

    def _mnormal(self, cov=np.diag([1, 1]), mean=None, shape=None):
        # Todo: finish
        raise NotImplementedError("{} is not working yet.".format(self._mnormal.__name__))
        # Shape fixing
        cov = np.array(cov)
        dft = self.fft()
        nx, ny = np.array(dft.shape) // 2
        if shape is None:
            shape = self.shape
        # Setting middle of image to default mean value.
        if mean is None:
            mean = np.array([nx, ny])
            # print(mean)

        # if given a single number for cov
        # print("Before: ", cov)
        if cov.ndim < 2:
            cov = np.diag(cov)
            # print("After: ", cov)

        mask = np.mgrid[-nx:nx, -ny:ny]
        mask = DipImage(self.d_multivariate_normal(mask, mean, cov))
        # mask.show()

    def _normal(self, sigma: Tuple[float, float] = (0, np.inf), shape=None):
        """Gaussian bandpass filter in 2d.
        If shape is not specified shape of self will be used.
        :param sigma: Lower cutoff-frequency is approx. 2*sigma[0]
                and higher cutoff-frequency is approx. 2*sigma[1].
        :param shape: Shape of returned bandpass filter."""
        if shape is None:
            shape = self.shape
        nx = shape[0] // 2
        ny = shape[1] // 2
        y, x = np.ogrid[-nx: nx, -ny: ny]
        distance2 = x ** 2 + y ** 2
        # Assuming covariance matrix is diagonal and sigma_x = sigma_y.
        highpass = - np.exp(- distance2 / (2 * sigma[0] ** 2)) / (np.sqrt(2 * np.pi) * sigma[1])
        lowpass = np.exp(- distance2 / (2 * sigma[1] ** 2)) / (np.sqrt(2 * np.pi) * sigma[1])
        mask = lowpass + highpass
        return DipImage(mask)

    def _boxcar(self, rmin=0, rmax=np.inf, shape=None):
        """Ideal bandpass filter in 2d. If shape is not specified shape of self will be used.
        :param rmin: Lower cutoff-frequency
        :param rmax: Higher cutoff-frequency
        :param shape: Shape of returned bandpass filter."""
        if shape is None:
            shape = self.shape
        nx = shape[0] // 2
        ny = shape[1] // 2
        # print(lenx, leny)
        y, x = np.ogrid[-nx: nx, -ny: ny]
        # print(y, x)
        # Making filter mask. 00000011111111000000000 in radius.
        mask = (rmin ** 2 <= x ** 2 + y ** 2) & (x ** 2 + y ** 2 <= rmax ** 2)
        return TimeSeries(mask)

    def _butterworth(self, rmin=0, rmax=np.inf, n=3, shape=None):
        """Butterworth bandpass filter in 2d.
        If shape is not specified shape of self will be used.
        :param rmin: Lower cutoff-frequency
        :param rmax: Higher cutoff-frequency
        :param shape: Shape of returned bandpass filter."""
        if shape is None:
            shape = self.shape
        nx = shape[0] // 2
        ny = shape[1] // 2
        y, x = np.ogrid[-nx: nx, -ny: ny]
        distance2 = x ** 2 + y ** 2
        d02 = rmin ** 2
        d12 = rmax ** 2
        highpass = - 1 / (1 + (distance2 / d02) ** n)
        lowpass = 1 / (1 + (distance2 / d12) ** n)
        mask = highpass + lowpass
        return DipImage(mask)

    def _crosscar(self, rx, ry, shape=None, center=None, rot=0):
        """Pass filter in 2d of a cross. """
        if shape is None:
            shape = self.shape
        # Placing center of cross
        if center is None:
            center = np.array(shape) // 2

        mask = np.zeros(shape)

        mask[center[0] - rx: center[0] + rx, :] = 1
        mask[:, center[1] - ry: center[1] + ry] = 1
        return DipImage(mask).rotate(rot)

    @staticmethod
    def d_multivariate_normal(X, mu, cov):
        print("Shapes:")
        print("\tmu: ", mu.shape)
        print("\tcov: ", cov.shape)
        print("\tX: ", X.shape, '\n')
        print("mu = ", mu)
        print("cov = ", cov)
        print("X = ", X)

        a = 1 / (np.sqrt(2 * np.pi) ** len(mu) * np.linalg.det(cov))
        exponent = -1 / 2 * (X - mu).T @ np.linalg.inv(cov) @ (X - mu)
        return a * np.exp(exponent)

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
        plt.subplot(2, 2, j + 1)
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

    image = im.bandpass(band=(50, 160), n=3, mode='butterworth')
    image.show()
    # im.show()

    # im2 = im.bandpass(band=(0, 100), mode='ideal')
    # im2.show()

    # im.show()
    # im = im.impad().fft(log=True)
    # im.show()

    # bandpassed = im.bandpass(band=(50, 150), mode='ideal')
    # bandpassed.show()


def t_TimeSeries():
    # # Perfect one
    # f = 5
    # x = TimeSeries(np.sin(2 * np.pi * f * np.linspace(0, 1, 2 ** 12)))
    # y = TimeSeries(np.sin(2 * np.pi * f * np.linspace(0, 1, 10)))
    # t = np.arange(2 ** 12)
    # # wb = y.fft(shift=False).Wb(length=2**14)
    # # plt.plot(t, x.fft(shift=True)/2**14)
    # plt.plot(t, y.fft(pad=2 ** 12))
    # # plt.plot(wb)
    # # a = x.fft().Wb(length=2000)
    # # plt.plot(a)
    # plt.show()

    # regression
    N = 10
    y = TimeSeries(0.09 * np.arange(N) ** (1 / 3) + np.random.randn(N))
    y_hat = y.polyreg(2)
    plt.plot(y)
    plt.plot(y_hat)
    plt.show()


if __name__ == '__main__':
    # example()
    # t_DipImage()
    # t_TimeSeries()
    pass
