import numpy, copy
import scipy.constants as codata

from srxraylib.util.data_structures import ScaledArray, ScaledMatrix

from wofry.propagator.wavefront import Wavefront, WavefrontDimension

from wofry.propagator.util.gaussian_schell_model import GaussianSchellModel1D

# needed for h5 i/o
import os
import sys
import time
from wofry.propagator.polarization import Polarization
try:
    import h5py
except:
    raise ImportError("h5py not available: input/output to files not working")
# --------------------------------------------------
# Wavefront 1D
# --------------------------------------------------

class GenericWavefront1D(Wavefront):

    def __init__(self, wavelength=1e-10, electric_field_array=None, electric_field_array_pi=None):
        self._wavelength = wavelength
        self._electric_field_array = electric_field_array
        self._electric_field_array_pi = electric_field_array_pi

    def get_dimension(self):
        return WavefrontDimension.ONE

    def is_polarized(self):
        if self._electric_field_array_pi is None:
            return False
        else:
            return True

    def duplicate(self):
        if self.is_polarized():
            return GenericWavefront1D(wavelength=self._wavelength,
                                      electric_field_array=ScaledArray(np_array=copy.copy(self._electric_field_array.np_array),
                                                                       scale=copy.copy(self._electric_field_array.scale)),
                                      electric_field_array_pi=ScaledArray(np_array=copy.copy(self._electric_field_array_pi.np_array),
                                                                       scale=copy.copy(self._electric_field_array_pi.scale)) )
        else:
            return GenericWavefront1D(wavelength=self._wavelength,
                                      electric_field_array=ScaledArray(np_array=copy.copy(self._electric_field_array.np_array),
                                                                       scale=copy.copy(self._electric_field_array.scale)))

    @classmethod
    def initialize_wavefront(cls, wavelength=1e-10, number_of_points=1000, polarization=Polarization.SIGMA):

        sA = ScaledArray.initialize(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex))

        if ((polarization == Polarization.PI) or (polarization == Polarization.TOTAL)):
            sA_pi = ScaledArray.initialize(np_array=numpy.full(number_of_points, (0.0 + 0.0j), dtype=complex))
        else:
            sA_pi = None

        return GenericWavefront1D(wavelength, sA, sA_pi)

    @classmethod
    def initialize_wavefront_from_steps(cls, x_start=-1.0, x_step=0.002, number_of_points=1000, wavelength=1e-10, polarization=Polarization.SIGMA):

        sA = ScaledArray.initialize_from_steps(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         initial_scale_value=x_start,
                                                                         scale_step=x_step)
        if ((polarization == Polarization.PI) or (polarization == Polarization.TOTAL)):
            sA_pi = ScaledArray.initialize_from_steps(np_array=numpy.full(number_of_points, (0.0 + 0.0j), dtype=complex),
                                                                         initial_scale_value=x_start,
                                                                         scale_step=x_step)
        else:
            sA_pi = None

        return GenericWavefront1D(wavelength, sA, sA_pi)

    @classmethod
    def initialize_wavefront_from_range(cls, x_min=0.0, x_max=0.0, number_of_points=1000, wavelength=1e-10, polarization=Polarization.SIGMA ):

        sA = ScaledArray.initialize_from_range(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         min_scale_value=x_min,
                                                                         max_scale_value=x_max)

        if ((polarization == Polarization.PI) or (polarization == Polarization.TOTAL)):
            sA_pi = ScaledArray.initialize_from_range(np_array=numpy.full(number_of_points, (0.0 + 0.0j), dtype=complex),
                                                                             min_scale_value=x_min,
                                                                             max_scale_value=x_max)
        else:
            sA_pi = None
        return GenericWavefront1D(wavelength, sA, sA_pi )

    @classmethod
    def initialize_wavefront_from_arrays(cls, x_array, y_array, y_array_pi=None, wavelength=1e-10):
        if x_array.size != y_array.size:
            raise Exception("Unmatched shapes for x and y")

        sA = ScaledArray.initialize_from_steps(np_array=y_array,
                                               initial_scale_value=x_array[0],
                                               scale_step=numpy.abs(x_array[1]-x_array[0]))

        if y_array_pi is not None:
            sA_pi = ScaledArray.initialize_from_steps(np_array=y_array_pi,
                                                   initial_scale_value=x_array[0],
                                                   scale_step=numpy.abs(x_array[1]-x_array[0]))
        else:
            sA_pi = None

        return GenericWavefront1D(wavelength, sA, sA_pi)


    # main parameters

    # grid

    def size(self):
        return self._electric_field_array.size()

    def delta(self):
        return self._electric_field_array.delta()

    def offset(self):
        return self._electric_field_array.offset()

    def get_abscissas(self):
        return self._electric_field_array.scale

    def get_mesh_x(self):
        return self.get_abscissas()

    # photon energy

    def get_wavelength(self):
        return self._wavelength

    def get_wavenumber(self):
        return 2*numpy.pi/self._wavelength

    def get_photon_energy(self):
        m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
        return  m2ev / self._wavelength


    # wavefront

    def get_complex_amplitude(self, polarization=Polarization.SIGMA):

        if polarization == Polarization.SIGMA:
            return self._electric_field_array.np_array
        elif polarization == Polarization.PI:
            if self.is_polarized():
                return self._electric_field_array_pi.np_array
            else:
                raise Exception("Wavefront is not polarized.")
        else:
            raise Exception("Only 0=SIGMA and 1=PI are valid polarization values.")

    def get_amplitude(self, polarization=Polarization.SIGMA):
        return numpy.absolute(self.get_complex_amplitude(polarization=polarization))

    def get_phase(self,from_minimum_intensity=0.0,unwrap=0, polarization=Polarization.SIGMA):
        phase = numpy.angle(self.get_complex_amplitude(polarization=polarization))
        if (from_minimum_intensity > 0.0):
            intensity = self.get_intensity()
            intensity /= intensity.max()
            bad_indices = numpy.where(intensity < from_minimum_intensity )
            phase[bad_indices] = 0.0
        if unwrap:
            phase = numpy.unwrap(phase)
        return phase

    def get_intensity(self, polarization=Polarization.SIGMA):
        if polarization == Polarization.TOTAL:
            if self.is_polarized():
                return self.get_amplitude(polarization=Polarization.SIGMA)**2 + \
                       self.get_amplitude(polarization=Polarization.PI)**2
            else:
                return self.get_amplitude(polarization=Polarization.SIGMA)**2
        else:
            return self.get_amplitude(polarization=polarization)**2

    def get_integrated_intensity(self, polarization=Polarization.SIGMA):
        return self.get_intensity(polarization=polarization).sum() * (self.get_abscissas()[1] - self.get_abscissas()[0])

    # interpolated values

    def get_interpolated_complex_amplitude(self, abscissa_value, polarization=Polarization.SIGMA): # singular

        if polarization == Polarization.SIGMA:
            return self._electric_field_array.interpolate_value(abscissa_value)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                return self._electric_field_array_pi.interpolate_value(abscissa_value)
            else:
                raise Exception("Wavefront is not polarized.")
        else:
            raise Exception("Only 0=SIGMA and 1=PI are valid polarization values.")


    def get_interpolated_complex_amplitudes(self, abscissa_values, polarization=Polarization.SIGMA): # plural

        if polarization == Polarization.SIGMA:
            return self._electric_field_array.interpolate_values(abscissa_values)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                return self._electric_field_array_pi.interpolate_values(abscissa_values)
            else:
                raise Exception("Wavefront is not polarized.")
        else:
            raise Exception("Only 0=SIGMA and 1=PI are valid polarization values.")



    def get_interpolated_amplitude(self, abscissa_value, polarization=Polarization.SIGMA): # singular!
        return numpy.absolute(self.get_interpolated_complex_amplitude(abscissa_value,polarization=polarization))

    def get_interpolated_amplitudes(self, abscissa_values, polarization=Polarization.SIGMA): # plural!
        return numpy.absolute(self.get_interpolated_complex_amplitudes(abscissa_values,polarization=polarization))

    def get_interpolated_phase(self, abscissa_value, polarization=Polarization.SIGMA): # singular!
        complex_amplitude = self.get_interpolated_complex_amplitude(abscissa_value, polarization=polarization)
        return numpy.arctan2(numpy.imag(complex_amplitude), numpy.real(complex_amplitude))

    def get_interpolated_phases(self, abscissa_values, polarization=Polarization.SIGMA): # plural!
        complex_amplitudes = self.get_interpolated_complex_amplitudes(abscissa_values, polarization=polarization)
        return numpy.arctan2(numpy.imag(complex_amplitudes), numpy.real(complex_amplitudes))

    def get_interpolated_intensity(self, abscissa_value, polarization=Polarization.SIGMA):
        if polarization == Polarization.TOTAL:
            interpolated_complex_amplitude = self.get_interpolated_amplitude(abscissa_value,polarization=Polarization.SIGMA)
            if self.is_polarized():
                interpolated_complex_amplitude_pi = self.get_interpolated_amplitude(abscissa_value,polarization=Polarization.PI)
                return numpy.abs(interpolated_complex_amplitude)**2 + numpy.abs(interpolated_complex_amplitude_pi)**2
            else:
                return numpy.abs(interpolated_complex_amplitude)**2
        elif polarization == Polarization.SIGMA:
            interpolated_complex_amplitude = self.get_interpolated_amplitude(abscissa_value,polarization=Polarization.SIGMA)
            return numpy.abs(interpolated_complex_amplitude)**2
        elif polarization == Polarization.PI:
            interpolated_complex_amplitude_pi = self.get_interpolated_amplitude(abscissa_value,polarization=Polarization.PI)
            return numpy.abs(interpolated_complex_amplitude_pi)**2
        else:
            raise Exception("Wrong polarization value.")


    def get_interpolated_intensities(self, abscissa_values, polarization=Polarization.SIGMA):
        # return self.get_interpolated_amplitudes(abscissa_values,polarization=Polarization.SIGMA)**2

        if polarization == Polarization.TOTAL:
            interpolated_complex_amplitude = self.get_interpolated_complex_amplitude(abscissa_values,polarization=Polarization.SIGMA)
            if self.is_polarized():
                interpolated_complex_amplitude_pi = self.get_interpolated_complex_amplitude(abscissa_values,polarization=Polarization.PI)
                return numpy.abs(interpolated_complex_amplitude)**2 + numpy.abs(interpolated_complex_amplitude_pi)**2
            else:
                return numpy.abs(interpolated_complex_amplitude)**2
        elif polarization == Polarization.SIGMA:
            interpolated_complex_amplitude = self.get_interpolated_complex_amplitude(abscissa_values,polarization=Polarization.SIGMA)
            return numpy.abs(interpolated_complex_amplitude)**2
        elif polarization == Polarization.PI:
            interpolated_complex_amplitude_pi = self.get_interpolated_complex_amplitude(abscissa_values,polarization=Polarization.PI)
            return numpy.abs(interpolated_complex_amplitude_pi)**2
        else:
            raise Exception("Wrong polarization value.")


    # modifiers

    def set_wavelength(self, wavelength):
        self._wavelength = wavelength

    def set_wavenumber(self, wavenumber):
        self._wavelength = 2 * numpy.pi / wavenumber

    def set_photon_energy(self, photon_energy):
        m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
        self._wavelength = m2ev / photon_energy

    def set_complex_amplitude(self, complex_amplitude, complex_amplitude_pi=None):
        if complex_amplitude.size != self._electric_field_array.size():
            raise Exception("Complex amplitude array has different dimension")

        self._electric_field_array.np_array = complex_amplitude

        if complex_amplitude_pi is not None:
            if self.is_polarized():
                if complex_amplitude_pi.size != self._electric_field_array_pi.size():
                    raise Exception("Complex amplitude array has different dimension")

                self._electric_field_array_pi.np_array = complex_amplitude_pi
            else:
                raise Exception('Cannot set PI-polarized complex amplitude to a non-polarized wavefront.')

    def set_pi_complex_amplitude_to_zero(self):
        if self.is_polarized():
            self._electric_field_array_pi.np_array *= 0.0

    def set_plane_wave_from_complex_amplitude(self, complex_amplitude=(1.0 + 0.0j), inclination=0.0):
        self._electric_field_array.np_array = numpy.full(self._electric_field_array.size(), complex_amplitude, dtype=complex)
        if inclination != 0.0:
            self.add_phase_shifts( self.get_wavenumber() * self._electric_field_array.scale * numpy.tan(inclination) )
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    def set_plane_wave_from_amplitude_and_phase(self, amplitude=1.0, phase=0.0, inclination=0.0):
        self.set_plane_wave_from_complex_amplitude(amplitude*numpy.cos(phase) + 1.0j*amplitude*numpy.sin(phase))
        if inclination != 0.0:
            self.add_phase_shifts( self.get_wavenumber() * self._electric_field_array.scale * numpy.tan(inclination) )
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    def set_spherical_wave(self, radius=1.0, center=0.0, complex_amplitude=1.0):
        if radius == 0: raise Exception("Radius cannot be zero")

        self._electric_field_array.np_array = complex_amplitude * numpy.exp(-1.0j * self.get_wavenumber() *
                                            ( (self._electric_field_array.scale - center)** 2) / (-2 * radius))
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    def set_gaussian_hermite_mode(self, sigma_x, mode_x, amplitude=1.0, shift=0.0, beta=100.0):
        a1D = GaussianSchellModel1D(amplitude, sigma_x, beta*sigma_x)

        real_amplitude = a1D.phi(mode_x, self.get_abscissas() - shift)
        eigenvalue = a1D.beta(mode_x)

        self.set_complex_amplitude(numpy.sqrt(eigenvalue)*real_amplitude+0.0j)
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    # note that amplitude is for "amplitude" not for intensity!
    def set_gaussian(self, sigma_x, amplitude=1.0, shift=0.0):
        self.set_gaussian_hermite_mode(sigma_x, 0, amplitude=amplitude, shift=shift)
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    def add_phase_shift(self, phase_shift, polarization=Polarization.SIGMA):
        if polarization == Polarization.SIGMA:
            self._electric_field_array.np_array *= numpy.exp(1.0j * phase_shift)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                self._electric_field_array_pi.np_array *= numpy.exp(1.0j * phase_shift)
            else:
                raise Exception("Wavefront is not polarized")
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA or 1=PI are valid)")

    def add_phase_shifts(self, phase_shifts, polarization=Polarization.SIGMA):

        if polarization == Polarization.SIGMA:
            if phase_shifts.size != self._electric_field_array.size():
                raise Exception("Phase Shifts array has different dimension")
            self._electric_field_array.np_array =  numpy.multiply(self._electric_field_array.np_array, numpy.exp(1.0j * phase_shifts))
        elif polarization == Polarization.PI:
            if self.is_polarized():
                if phase_shifts.size != self._electric_field_array_pi.size():
                    raise Exception("Phase Shifts array has different dimension")
                self._electric_field_array_pi.np_array =  numpy.multiply(self._electric_field_array_pi.np_array, numpy.exp(1.0j * phase_shifts))
            else:
                raise Exception("Wavefront is not polarized")
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA or 1=PI are valid)")


    def rescale_amplitude(self, factor, polarization=Polarization.SIGMA):

        if polarization == Polarization.SIGMA:
            self._electric_field_array.np_array *= factor
        elif polarization == Polarization.PI:
            if self.is_polarized():
                self._electric_field_array_pi.np_array *= factor
            else:
                raise Exception("Wavefront is not polarized")
        elif polarization == Polarization.TOTAL:
            self.rescale_amplitude(factor, polarization=Polarization.SIGMA)
            self.rescale_amplitude(factor, polarization=Polarization.PI)
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA, 1=PI or 3=TOTAL are valid)")


    def rescale_amplitudes(self, factors, polarization=Polarization.SIGMA):


        if polarization == Polarization.SIGMA:
            if factors.size != self._electric_field_array.size(): raise Exception("Factors array has different dimension")
            self._electric_field_array.np_array =  numpy.multiply(self._electric_field_array.np_array, factors)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                if factors.size != self._electric_field_array_pi.size(): raise Exception("Factors array has different dimension")
                self._electric_field_array_pi.np_array =  numpy.multiply(self._electric_field_array_pi.np_array, factors)
            else:
                raise Exception("Wavefront is not polarized")
        elif polarization == Polarization.TOTAL:
            self.rescale_amplitudes(factors, polarization=Polarization.SIGMA)
            self.rescale_amplitudes(factors, polarization=Polarization.PI)
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA, 1=PI or 3=TOTAL are valid)")


    def clip(self, x_min, x_max, negative=False):
        window = numpy.ones(self._electric_field_array.size())

        if not negative:
            lower_window = numpy.where(self.get_abscissas() < x_min)
            upper_window = numpy.where(self.get_abscissas() > x_max)

            if len(lower_window) > 0: window[lower_window] = 0
            if len(upper_window) > 0: window[upper_window] = 0
        else:
            window_indices = numpy.where((self.get_abscissas() >= x_min) & (self.get_abscissas() <= x_max))

            if len(window_indices) > 0:
                window[window_indices] = 0.0

        if self.is_polarized():
            self.rescale_amplitudes(window,polarization=Polarization.TOTAL)
        else:
            self.rescale_amplitudes(window,polarization=Polarization.SIGMA)

    def is_identical(self,wfr,decimal=7):
        from numpy.testing import assert_array_almost_equal
        try:
            assert_array_almost_equal(self.get_complex_amplitude(),wfr.get_complex_amplitude(),decimal)
            assert(self.is_polarized() == wfr.is_polarized())
            if self.is_polarized():
                assert_array_almost_equal(self.get_complex_amplitude(polarization=Polarization.PI),
                                          wfr.get_complex_amplitude(polarization=Polarization.PI),decimal)

            assert_array_almost_equal(self.get_abscissas(),wfr.get_abscissas(),decimal)
            assert_array_almost_equal(self.get_photon_energy(),wfr.get_photon_energy(),decimal)
        except:
            return False

        return True


    #
    # auxiliary methods get main wavefront phase curvature (radius)
    #
    def _figure_of_merit(self,radius,weight_with_intensity=True):
        """
        Computes a "figure of merit" for finding the wavefront curvature.
        A low value of the figure of metit means that the entered radius (checked)
        corresponds to the radius of the wavefront.

        If wavefront is polarized, the pi component is ignored.

        :param radius:
        :param weight_with_intensity:
        :return: a positive scalar with the figure of merit
        """
        x = self.get_abscissas()
        new_phase = 1.0 * self.get_wavenumber() * (x**2) / (-2 * radius)

        wavefront2 = self.duplicate()
        wavefront2.add_phase_shifts(new_phase)

        if weight_with_intensity:
            out = numpy.abs(wavefront2.get_phase()*wavefront2.get_intensity()).sum()
        else:
            out = numpy.abs(wavefront2.get_phase()).sum()

        return out

    def scan_wavefront_curvature(self,rmin=-10000.0,rmax=10000.0,rpoints=100):

        radii = numpy.linspace(rmax,rmin,rpoints)
        fig_of_mer = numpy.zeros_like(radii)

        for i,radius in enumerate(radii):
            fig_of_mer[i] =self._figure_of_merit(radius)

        return radii,fig_of_mer


    def guess_wavefront_curvature(self,rmin=-10000.0,rmax=10000.0,rpoints=100):
        from scipy.optimize import minimize

        radii,fig_of_mer = self.scan_wavefront_curvature(rmin=rmin,rmax=rmax,rpoints=rpoints)

        res = minimize(self._figure_of_merit, radii[numpy.argmin(fig_of_mer)], args=self, method='powell',options={'xtol': 1e-8, 'disp': True})

        return res.x

    #
    # auxiliary function to dump h5 files
    #
    def _dump_arr_2_hdf5(self,_arr,_calculation, _filename, _subgroupname):
        """
        Auxiliary routine to save_h5_file
        :param _arr: (usually 2D) array to be saved on the hdf5 file inside the _subgroupname
        :param _calculation
        :param _filename: path to file for saving the wavefront
        :param _subgroupname: container mechanism by which HDF5 files are organised
        """
        sys.stdout.flush()
        f = h5py.File(_filename, 'a')
        try:
            f1 = f.create_group(_subgroupname)
        except:
            f1 = f[_subgroupname]
        fdata = f1.create_dataset(_calculation, data=_arr)
        f.close()

    def save_h5_file(self,filename,subgroupname="wfr",intensity=True,phase=False,overwrite=True,verbose=False):
        """
        Auxiliary function to write wavefront data into a hdf5 generic file.
        When using the append mode to write h5 files, overwriting does not work and makes the code crash. To avoid this
        issue, try/except is used. If by any chance a file should be overwritten, it is firstly deleted and re-written.
        :param self: input / output resulting Wavefront structure (instance of GenericWavefront2D);
        :param filename: path to file for saving the wavefront
        :param subgroupname: container mechanism by which HDF5 files are organised
        :param intensity: writes intensity for sigma and pi polarisation (default=True)
        :param amplitude:
        :param phase:
        :param overwrite: flag that should always be set to True to avoid infinity loop on the recursive part of the function.
        :param verbose: if True, print some file i/o messages
        """
        if overwrite:
            try:
                os.remove(filename)
            except:
                pass


        try:
            if not os.path.isfile(filename):  # if file doesn't exist, create it.
                sys.stdout.flush()
                f = h5py.File(filename, 'w')
                # point to the default data to be plotted
                f.attrs['default']          = 'entry'
                # give the HDF5 root some more attributes
                f.attrs['file_name']        = filename
                f.attrs['file_time']        = time.time()
                f.attrs['creator']          = 'oasys-wofry'
                f.attrs['HDF5_Version']     = h5py.version.hdf5_version
                f.attrs['h5py_version']     = h5py.version.version
                f.close()

            # always writes complex amplitude
            x_polarization = self.get_complex_amplitude()       # sigma
            self._dump_arr_2_hdf5(x_polarization, "wfr_complex_amplitude_s", filename, subgroupname)

            if self.is_polarized():
                y_polarization = self.get_complex_amplitude(polarization=Polarization.PI)       # pi
                self._dump_arr_2_hdf5(y_polarization, "wfr_complex_amplitude_p", filename, subgroupname)

            if intensity:
                if self.is_polarized():
                    self._dump_arr_2_hdf5(self.get_intensity(polarization=Polarization.TOTAL),"intensity/wfr_intensity", filename, subgroupname)
                    self._dump_arr_2_hdf5(self.get_intensity(polarization=Polarization.SIGMA),"intensity/wfr_intensity_s", filename, subgroupname)
                    self._dump_arr_2_hdf5(self.get_intensity(polarization=Polarization.PI),"intensity/wfr_intensity_p", filename, subgroupname)
                else:
                    self._dump_arr_2_hdf5(self.get_intensity(),"intensity/wfr_intensity", filename, subgroupname)

            if phase:
                if self.is_polarized():
                    self._dump_arr_2_hdf5(self.get_phase(polarization=Polarization.SIGMA)-self.get_phase(polarization=Polarization.PI),
                                          "phase/wfr_phase", filename, subgroupname)
                    self._dump_arr_2_hdf5(self.get_phase(polarization=Polarization.SIGMA),"phase/wfr_phase_s", filename, subgroupname)
                    self._dump_arr_2_hdf5(self.get_phase(polarization=Polarization.PI),"phase/wfr_phase_p", filename, subgroupname)
                else:
                    self._dump_arr_2_hdf5(self.get_phase(polarization=Polarization.SIGMA),"phase/wfr_phase", filename, subgroupname)


            # add mesh and SRW information
            f = h5py.File(filename, 'a')
            f1 = f[subgroupname]

            # point to the default data to be plotted
            f1.attrs['NX_class'] = 'NXentry'
            f1.attrs['default'] = 'intensity'

            # TODO: add self interpreting decoder
            # f1["wfr_method"] = "WOFRY"
            f1["wfr_dimension"] = 1
            f1["wfr_photon_energy"] = self.get_photon_energy()
            x = self.get_abscissas()


            f1["wfr_mesh"] =  numpy.array([x[0],x[-1],x.size])


            # Add NX plot attribites for automatic plot with silx view
            myflags = [intensity,phase]
            mylabels = ['intensity','phase']
            for i,label in enumerate(mylabels):
                if myflags[i]:
                    f2 = f1[mylabels[i]]
                    f2.attrs['NX_class'] = 'NXdata'
                    f2.attrs['signal'] = 'wfr_%s'%(mylabels[i])
                    f2.attrs['axes'] = b'axis_x'

                    f3 = f2["wfr_%s"%(mylabels[i])]

                    # axis data
                    ds = f2.create_dataset('axis_x', data=1e6*x)
                    ds.attrs['units'] = 'microns'
                    ds.attrs['long_name'] = 'Pixel Size (microns)'    # suggested Y axis plot label
            f.close()

        except:
            # TODO: check exit??
            if overwrite is not True:
                raise Exception("Bad input argument")
            os.remove(filename)
            if verbose: print("save_h5_file: file deleted %s"%filename)
            self.save_h5_file(filename,subgroupname, intensity=intensity, phase=phase, overwrite=False)

        if verbose: print("save_h5_file: written/updated %s data in file: %s"%(subgroupname,filename))

    @classmethod
    def load_h5_file(cls,filename,filepath="wfr"):

        try:
            f = h5py.File(filename, 'r')
            mesh = f[filepath+"/wfr_mesh"][()]
            complex_amplitude_s = f[filepath+"/wfr_complex_amplitude_s"][()]
            energy = f[filepath + "/wfr_photon_energy"][()]
            try:
                complex_amplitude_p = f[filepath + "/wfr_complex_amplitude_p"][()]
            except:
                complex_amplitude_p = None
            f.close()
        except:
            raise Exception("Failed to load 2D wavefront from h5 file: "+filename)

        wfr = cls.initialize_wavefront_from_arrays(
                            numpy.linspace(mesh[0],mesh[1],int(mesh[2])),
                            complex_amplitude_s,complex_amplitude_p)
        wfr.set_photon_energy(energy)
        return wfr


if __name__ == "__main__":
    # w = GenericWavefront1D.initialize_wavefront_from_steps(polarization=Polarization.TOTAL)
    # w.save_h5_file("/tmp/wf.h5",subgroupname="wfr",intensity=True,phase=False,overwrite=True,verbose=True)
    # w2 = GenericWavefront1D.load_h5_file("/tmp/wf.h5",filepath="wfr")
    # assert(w2.is_identical(w))
    pass