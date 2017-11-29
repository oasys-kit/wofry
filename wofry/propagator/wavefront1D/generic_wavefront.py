import numpy, copy
import scipy.constants as codata

from srxraylib.util.data_structures import ScaledArray, ScaledMatrix

from wofry.propagator.wavefront import Wavefront, WavefrontDimension

from wofry.propagator.util.gaussian_schell_model import GaussianSchellModel1D

# --------------------------------------------------
# Wavefront 1D
# --------------------------------------------------

class GenericWavefront1D(Wavefront):

    def get_dimension(self):
        return WavefrontDimension.ONE

    def __init__(self, wavelength=1e-10, electric_field_array=None):
        self._wavelength = wavelength
        self._electric_field_array = electric_field_array

    def duplicate(self):
        return GenericWavefront1D(wavelength=self._wavelength,
                                  electric_field_array=ScaledArray(np_array=copy.copy(self._electric_field_array.np_array),
                                                                   scale=copy.copy(self._electric_field_array.scale)))

    @classmethod
    def initialize_wavefront(cls, wavelength=1e-10, number_of_points=1000):
        return GenericWavefront1D(wavelength, ScaledArray.initialize(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex)))

    @classmethod
    def initialize_wavefront_from_steps(cls, x_start=0.0, x_step=0.0, number_of_points=1000, wavelength=1e-10):
        return GenericWavefront1D(wavelength, ScaledArray.initialize_from_steps(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         initial_scale_value=x_start,
                                                                         scale_step=x_step))
    @classmethod
    def initialize_wavefront_from_range(cls, x_min=0.0, x_max=0.0, number_of_points=1000, wavelength=1e-10 ):
        return GenericWavefront1D(wavelength, ScaledArray.initialize_from_range(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         min_scale_value=x_min,
                                                                         max_scale_value=x_max))

    @classmethod
    def initialize_wavefront_from_arrays(cls, x_array, y_array, wavelength=1e-10,):
        if x_array.size != y_array.size:
            raise Exception("Unmatched shapes for x and y")

        return GenericWavefront1D(wavelength, ScaledArray.initialize_from_steps(np_array=y_array,
                                                                                initial_scale_value=x_array[0],
                                                                                scale_step=numpy.abs(x_array[1]-x_array[0])))


    # main parameters

    def size(self):
        return self._electric_field_array.size()

    def delta(self):
        return self._electric_field_array.delta()

    def offset(self):
        return self._electric_field_array.offset()

    def get_wavelength(self):
        return self._wavelength

    def get_wavenumber(self):
        return 2*numpy.pi/self._wavelength

    def get_abscissas(self):
        return self._electric_field_array.scale

    def get_mesh_x(self):
        return self.get_abscissas()

    def get_complex_amplitude(self):
        return self._electric_field_array.np_array

    def get_amplitude(self):
        return numpy.absolute(self.get_complex_amplitude())

    def get_phase(self,from_minimum_intensity=0.0):
        phase = numpy.angle(self.get_complex_amplitude())
        if (from_minimum_intensity > 0.0):
            intensity = self.get_intensity()
            intensity /= intensity.max()
            bad_indices = numpy.where(intensity < from_minimum_intensity )
            phase[bad_indices] = 0.0

        return phase

    def get_intensity(self):
        return self.get_amplitude()**2

    # interpolated values

    def get_interpolated_complex_amplitude(self, abscissa_value): # singular
        return self._electric_field_array.interpolate_value(abscissa_value)

    def get_interpolated_complex_amplitudes(self, abscissa_values): # plural
        return self._electric_field_array.interpolate_values(abscissa_values)

    def get_interpolated_amplitude(self, abscissa_value): # singular!
        return numpy.absolute(self.get_interpolated_complex_amplitude(abscissa_value))

    def get_interpolated_amplitudes(self, abscissa_values): # plural!
        return numpy.absolute(self.get_interpolated_complex_amplitudes(abscissa_values))

    def get_interpolated_phase(self, abscissa_value): # singular!
        complex_amplitude = self.get_interpolated_complex_amplitude(abscissa_value)
        return numpy.arctan2(numpy.imag(complex_amplitude), numpy.real(complex_amplitude))

    def get_interpolated_phases(self, abscissa_values): # plural!
        complex_amplitudes = self.get_interpolated_complex_amplitudes(abscissa_values)
        return numpy.arctan2(numpy.imag(complex_amplitudes), numpy.real(complex_amplitudes))

    def get_interpolated_intensity(self, abscissa_value):
        return self.get_interpolated_amplitude(abscissa_value)**2

    def get_interpolated_intensities(self, abscissa_values):
        return self.get_interpolated_amplitudes(abscissa_values)**2

    # modifiers

    def set_wavelength(self, wavelength):
        self._wavelength = wavelength

    def set_wavenumber(self, wavenumber):
        self._wavelength = 2 * numpy.pi / wavenumber

    def set_photon_energy(self, photon_energy):
        m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
        self._wavelength = m2ev / photon_energy

    def set_complex_amplitude(self, complex_amplitude):
        if complex_amplitude.size != self._electric_field_array.size():
            raise Exception("Complex amplitude array has different dimension")

        self._electric_field_array.np_array = complex_amplitude

    def set_plane_wave_from_complex_amplitude(self, complex_amplitude=(1.0 + 0.0j)):
        self._electric_field_array.np_array = numpy.full(self._electric_field_array.size(), complex_amplitude, dtype=complex)

    def set_plane_wave_from_amplitude_and_phase(self, amplitude=1.0, phase=0.0):
        self.set_plane_wave_from_complex_amplitude(amplitude*numpy.cos(phase) + 1.0j*amplitude*numpy.sin(phase))

    def set_spherical_wave(self, radius=1.0, complex_amplitude=1.0):
        if radius == 0: raise Exception("Radius cannot be zero")

        self._electric_field_array.np_array = (complex_amplitude / (-radius)) * numpy.exp(-1.0j * self.get_wavenumber() *
                                                                                          (self._electric_field_array.scale ** 2) / (-2 * radius))


    def set_gaussian_hermite_mode(self, sigma_x, mode_x, amplitude=1.0):
        a1D = GaussianSchellModel1D(amplitude, sigma_x, 100.0*sigma_x)

        real_amplitude = a1D.phi(mode_x, self.get_abscissas())

        self.set_complex_amplitude(real_amplitude)

    # note that amplitude is for "amplitude" not for intensity!
    def set_gaussian(self, sigma_x, amplitude=1.0):
        self.set_gaussian_hermite_mode(sigma_x, 0, amplitude=amplitude)

    def add_phase_shift(self, phase_shift):
        self._electric_field_array.np_array *= numpy.exp(1.0j * phase_shift)

    def add_phase_shifts(self, phase_shifts):
        if phase_shifts.size != self._electric_field_array.size():
            raise Exception("Phase Shifts array has different dimension")

        self._electric_field_array.np_array =  numpy.multiply(self._electric_field_array.np_array, numpy.exp(1.0j * phase_shifts))

    def rescale_amplitude(self, factor):
        self._electric_field_array.np_array *= factor

    def rescale_amplitudes(self, factors):
        if factors.size != self._electric_field_array.size(): raise Exception("Factors array has different dimension")

        self._electric_field_array.np_array =  numpy.multiply(self._electric_field_array.np_array, factors)

    def clip(self, x_min, x_max, negative=False):
        window = numpy.ones(self._electric_field_array.size())

        if not negative:
            lower_window = numpy.where(self.get_abscissas() < x_min)
            upper_window = numpy.where(self.get_abscissas() > x_max)

            if len(lower_window) > 0: window[lower_window] = 0
            if len(upper_window) > 0: window[upper_window] = 0
        else:
            window = numpy.where(x_min <= self.get_abscissas() <= x_max)

            if len(window) > 0: window[window] = 0

        self.rescale_amplitudes(window)
