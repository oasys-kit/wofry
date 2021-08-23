import numpy
import scipy.constants as codata

from srxraylib.util.data_structures import ScaledMatrix, ScaledArray

from wofry.propagator.wavefront import Wavefront, WavefrontDimension
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D

from wofry.propagator.util.gaussian_schell_model import GaussianSchellModel2D

import copy


from wofry.propagator.polarization import Polarization
# class Polarization:
#     SIGMA = 0
#     PI = 1
#     TOTAL = 3

# needed for h5 i/o
import os
import sys
import time
try:
    import h5py
except:
    raise ImportError("h5py not available: input/output to files not working")
# --------------------------------------------------
# Wavefront 2D
# --------------------------------------------------




class GenericWavefront2D(Wavefront):
    XX = 0
    YY = 1


    def __init__(self, wavelength=1e-10, electric_field_matrix=None, electric_field_matrix_pi=None):
        self._wavelength = wavelength
        self._electric_field_matrix = electric_field_matrix
        self._electric_field_matrix_pi = electric_field_matrix_pi

    def get_dimension(self):
        return WavefrontDimension.TWO

    def is_polarized(self):
        if self._electric_field_matrix_pi is None:
            return False
        else:
            return True

    def duplicate(self):

        if self.is_polarized():
            return GenericWavefront2D(wavelength=self._wavelength,
                                      electric_field_matrix=ScaledMatrix(x_coord=   copy.copy(self._electric_field_matrix.x_coord),
                                                                         y_coord=   copy.copy(self._electric_field_matrix.y_coord),
                                                                         z_values=  copy.copy(self._electric_field_matrix.z_values),
                                                                         interpolator=        self._electric_field_matrix.interpolator),
                                      electric_field_matrix_pi=ScaledMatrix(x_coord=copy.copy(self._electric_field_matrix_pi.x_coord),
                                                                         y_coord=   copy.copy(self._electric_field_matrix_pi.y_coord),
                                                                         z_values=  copy.copy(self._electric_field_matrix_pi.z_values),
                                                                         interpolator=        self._electric_field_matrix_pi.interpolator),
                                      )
        else:
            return GenericWavefront2D(wavelength=self._wavelength,
                                      electric_field_matrix=ScaledMatrix(x_coord=copy.copy(self._electric_field_matrix.x_coord),
                                                                         y_coord=copy.copy(self._electric_field_matrix.y_coord),
                                                                         z_values=copy.copy(self._electric_field_matrix.z_values),
                                                                         interpolator=self._electric_field_matrix.interpolator))

    @classmethod
    def initialize_wavefront(cls, number_of_points=(100,100), wavelength=1e-10, polarization=Polarization.SIGMA):

        sM = ScaledMatrix.initialize(np_array_z=numpy.full(number_of_points, (1.0 + 0.0j),dtype=complex),
                                                          interpolator=False)

        if ((polarization == Polarization.PI) or (polarization == Polarization.TOTAL)):
            sM_pi = ScaledMatrix.initialize(np_array_z=numpy.full(number_of_points, (0.0 + 0.0j),dtype=complex),
                                                          interpolator=False)
        else:
            sM_pi = None

        return GenericWavefront2D(wavelength, sM, sM_pi)

    @classmethod
    def initialize_wavefront_from_steps(cls, x_start=0.0, x_step=1.0, y_start=0.0, y_step=1.0,
                                        number_of_points=(100,100), wavelength=1e-10,
                                        polarization=Polarization.SIGMA):
        sM = ScaledMatrix.initialize_from_steps(numpy.full(number_of_points,(1.0 + 0.0j),dtype=complex),
                                                x_start,
                                                x_step,
                                                y_start,
                                                y_step,
                                                interpolator=False)

        if ((polarization == Polarization.PI) or (polarization == Polarization.TOTAL)):
            sM_pi = ScaledMatrix.initialize_from_steps(numpy.full(number_of_points,(0.0 + 0.0j),dtype=complex),
                                                x_start,
                                                x_step,
                                                y_start,
                                                y_step,
                                                interpolator=False)
        else:
            sM_pi = None

        return GenericWavefront2D(wavelength,sM, sM_pi)

    @classmethod
    def initialize_wavefront_from_range(cls, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0,
                                        number_of_points=(100,100), wavelength=1e-10,
                                        polarization=Polarization.SIGMA ):

        sM = ScaledMatrix.initialize_from_range( \
                    numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                    x_min,x_max,y_min,y_max,interpolator=False)

        if ((polarization == Polarization.PI) or (polarization == Polarization.TOTAL)):
            sM_pi = ScaledMatrix.initialize_from_range( \
                    numpy.full(number_of_points, (0.0 + 0.0j), dtype=complex),
                    x_min,x_max,y_min,y_max,interpolator=False)
        else:
            sM_pi = None

        return GenericWavefront2D(wavelength, sM, sM_pi)

    @classmethod
    def initialize_wavefront_from_arrays(cls,x_array, y_array,  z_array, z_array_pi=None, wavelength=1e-10):
        sh = z_array.shape

        if sh[0] != x_array.size:
            raise Exception("Unmatched shapes for x")
        
        if sh[1] != y_array.size:
            raise Exception("Unmatched shapes for y")

        sM = ScaledMatrix.initialize_from_steps(
                    z_array,x_array[0],numpy.abs(x_array[1]-x_array[0]),
                            y_array[0],numpy.abs(y_array[1]-y_array[0]),interpolator=False)

        if z_array_pi is None:
            sM_pi = None
        else:
            sh = z_array_pi.shape
            if sh[0] != x_array.size:
                raise Exception("Unmatched shapes for x (Pi Polarization)")

            if sh[1] != y_array.size:
                raise Exception("Unmatched shapes for y (Pi Polarization)")

            sM_pi = ScaledMatrix.initialize_from_steps(
                        z_array_pi,x_array[0],numpy.abs(x_array[1]-x_array[0]),
                                y_array[0],numpy.abs(y_array[1]-y_array[0]),interpolator=False)

        return GenericWavefront2D(wavelength, sM, sM_pi)



    # TODO: polarization
    def get_Wavefront1D_from_profile(self, axis, coordinate):
        # swap axis - changed giovanni+manuel
        if axis == 1: # fixed X
            index = numpy.argmin(numpy.abs(self._electric_field_matrix.x_coord - coordinate))

            return GenericWavefront1D(wavelength=self._wavelength,
                                      electric_field_array=ScaledArray(scale=self._electric_field_matrix.y_coord,
                                                                       np_array=self._electric_field_matrix.z_values[index, :]))
        elif axis == 0:
            index = numpy.argmin(numpy.abs(self._electric_field_matrix.y_coord - coordinate))

            return GenericWavefront1D(wavelength=self._wavelength,
                                      electric_field_array=ScaledArray(scale=self._electric_field_matrix.x_coord,
                                                                       np_array=self._electric_field_matrix.z_values[:, index]))

    #TODO
    def get_Wavefront1D_from_histogram(self, axis):
        raise NotImplementedError("Not yet implemented!")


    # TODO: check polarization
    @classmethod
    def combine_1D_wavefronts_into_2D(cls, wavefront_h, wavefront_v, normalize_to=0, wavelength=None, polarization=Polarization.SIGMA):
        """
        Create a 2D wavefront by doing the outer product of two 1D wavefront.

        :param wavefront_h: horozontal wavefront
        :param wavefront_v: vertical wavefront
        :param normalize_to: 0=horizontal, 1=vertical, 2=One, 3=No normalization
        :param wavelength: wavelength for combined wavefront. If None (default), use the average
        :param polarization: build the 2D wavefront using 1D wavefront polarization 0=Sigma, 1=Pi, 2=Total
        :return:
        """
        if wavelength is None:
            wavelength = (wavefront_h.get_wavelength() + wavefront_v.get_wavelength())/2


        wavefront_2D = GenericWavefront2D.initialize_wavefront_from_steps(x_start=wavefront_h.offset(),
                                                                          x_step=wavefront_h.delta(),
                                                                          y_start=wavefront_v.offset(),
                                                                          y_step=wavefront_v.delta(),
                                                                          number_of_points=(wavefront_h.size(), wavefront_v.size()))


        complex_amplitude = numpy.outer(wavefront_h.get_complex_amplitude(polarization=polarization),
                                        wavefront_v.get_complex_amplitude(polarization=polarization))


        wavefront_2D.set_complex_amplitude(complex_amplitude)
        wavefront_2D.set_wavelength(wavelength)


        if normalize_to == 0:
            wavefront_2D.rescale_amplitude(numpy.sqrt(wavefront_h.get_integrated_intensity(polarization=polarization) / \
                                           wavefront_2D.get_integrated_intensity(polarization=polarization)), \
                                           polarization=polarization)
        elif normalize_to == 1:
            wavefront_2D.rescale_amplitude(numpy.sqrt(wavefront_v.get_integrated_intensity(polarization=polarization) / \
                                           wavefront_2D.get_integrated_intensity(polarization=polarization)),
                                           polarization=polarization)
        elif normalize_to == 2: # One
            wavefront_2D.rescale_amplitude(numpy.sqrt(1.0/wavefront_2D.get_integrated_intensity(polarization=polarization)), \
                                           polarization=polarization)
        elif normalize_to == 3: # None
            pass



        return wavefront_2D

    #
    # main parameters
    #

    # grid

    def size(self):
        return self._electric_field_matrix.shape()

    def delta(self):
        x = self.get_coordinate_x()
        y = self.get_coordinate_y()
        return numpy.abs(x[1]-x[0]),numpy.abs(y[1]-y[0])
    #
    def offset(self):
        return self.get_coordinate_x()[0],self.get_coordinate_y()[0]


    def get_coordinate_x(self):
        return self._electric_field_matrix.get_x_values()

    def get_coordinate_y(self):
        return self._electric_field_matrix.get_y_values()


    def get_mesh_indices_x(self):
        return numpy.outer( numpy.arange(0,self.size()[0]), numpy.ones(self.size()[1]))

    def get_mesh_indices_y(self):
        return numpy.outer( numpy.ones(self.size()[0]), numpy.arange(0,self.size()[1]))

    def get_mask_grid(self,width_in_pixels=(1,1),number_of_lines=(1,1)):
        """

        :param width_in_pixels: (pixels_for_horizontal_lines,pixels_for_vertical_lines
        :param number_of_lines: (number_of_horizontal_lines, number_of_vertical_lines)
        :return:
        """

        indices_x = self.get_mesh_indices_x()
        indices_y = self.get_mesh_indices_y()

        used_indices = numpy.zeros( self.size(),dtype=int)

        for i in range(number_of_lines[1]):
            used_indices[ numpy.where( numpy.abs(indices_x - (i+1)*self.size()[0]/(1+number_of_lines[1])) <= (width_in_pixels[1]-1) )] = 1
        for i in range(number_of_lines[0]):
            used_indices[ numpy.where( numpy.abs(indices_y - (i+1)*self.size()[1]/(1+number_of_lines[0])) <= (width_in_pixels[0]-1) )] = 1

        return used_indices

    # only for 2D
    def get_mesh_x(self):
        XY = numpy.meshgrid(self.get_coordinate_x(),self.get_coordinate_y())
        return XY[0].T

    def get_mesh_y(self):
        XY = numpy.meshgrid(self.get_coordinate_x(),self.get_coordinate_y())
        return XY[1].T

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
            return self._electric_field_matrix.get_z_values()
        elif polarization == Polarization.PI:
            if self.is_polarized():
                return self._electric_field_matrix_pi.get_z_values()
            else:
                raise Exception("Wavefront is not polarized.")
        else:
            raise Exception("Only 0=SIGMA and 1=PI are valid polarization values.")


    def get_amplitude(self, polarization=Polarization.SIGMA):
        return numpy.absolute(self.get_complex_amplitude(polarization=polarization))

    def get_phase(self,from_minimum_intensity=0.0,unwrap=0, polarization=Polarization.SIGMA):

        """

        :param from_minimum_intensity: set to zero phase values at pixels where intensity
                                        is less than from_minimum_intensity threshold
        :param unwrap: Flag to unwrap the returned phase:
            0: No unwrap (default)
            1: Unwrap only in Horizontal axis.
            2: Unwrap only in Vertical axis.
            3: Unwrap first in H, then in V.
            4: Unwrap first in V, then in H.
        :param polarization: 0=Sigma, 1=Pi
        :return: the phase in a numpy array
        """
        # return numpy.arctan2(numpy.imag(self.get_complex_amplitude()), numpy.real(self.get_complex_amplitude()))
        phase = numpy.angle( self.get_complex_amplitude(polarization=polarization) )

        if (from_minimum_intensity > 0.0):
            intensity = self.get_intensity()
            intensity /= intensity.max()
            bad_indices = numpy.where(intensity < from_minimum_intensity )
            phase[bad_indices] = 0.0

        if unwrap > 0:
            if unwrap == 1: # x only
                phase = numpy.unwrap(phase,axis=0)
            elif unwrap == 2: # y only
                phase = numpy.unwrap(phase,axis=1)
            elif unwrap == 3: # x and y
                phase = numpy.unwrap(numpy.unwrap(phase,axis=0),axis=1)
            elif unwrap == 4: # y and x
                phase = numpy.unwrap(numpy.unwrap(phase,axis=1),axis=0)
            else:
                raise Exception(NotImplemented)

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
        deltas =    (self.get_coordinate_x()[1] - self.get_coordinate_x()[0]) * \
                    (self.get_coordinate_y()[1] - self.get_coordinate_y()[0])
        return self.get_intensity(polarization=polarization).sum() * deltas



    # interpolated values

    def get_interpolated_complex_amplitude(self, x_value,y_value, polarization=Polarization.SIGMA):

        if polarization == Polarization.SIGMA:
            return self._electric_field_matrix.interpolate_value(x_value,y_value)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                return self._electric_field_matrix_pi.interpolate_value(x_value,y_value)
            else:
                raise Exception("Wavefront is not polarized.")
        else:
            raise Exception("Only 0=SIGMA and 1=PI are valid polarization values.")

    def get_interpolated_amplitude(self, x_value,y_value, polarization=Polarization.SIGMA): # singular!
        interpolated_complex_amplitude = self.get_interpolated_complex_amplitude(x_value,y_value,polarization=polarization)
        return numpy.abs(interpolated_complex_amplitude)

    def get_interpolated_phase(self, x_value,y_value, polarization=Polarization.SIGMA): # singular!
        interpolated_complex_amplitude = self.get_interpolated_complex_amplitude(x_value,y_value,polarization=polarization)
        return numpy.arctan2(numpy.imag(interpolated_complex_amplitude), numpy.real(interpolated_complex_amplitude))


    def get_interpolated_intensity(self, x_value,y_value, polarization=Polarization.SIGMA):
        if polarization == Polarization.TOTAL:
            interpolated_complex_amplitude = self.get_interpolated_complex_amplitude(x_value,y_value,polarization=Polarization.SIGMA)
            if self.is_polarized():
                interpolated_complex_amplitude_pi = self.get_interpolated_complex_amplitude(x_value,y_value,polarization=Polarization.PI)
                return numpy.abs(interpolated_complex_amplitude)**2 + numpy.abs(interpolated_complex_amplitude_pi)**2
            else:
                return numpy.abs(interpolated_complex_amplitude)**2
        elif polarization == Polarization.SIGMA:
            interpolated_complex_amplitude = self.get_interpolated_complex_amplitude(x_value,y_value,polarization=Polarization.SIGMA)
            return numpy.abs(interpolated_complex_amplitude)**2
        elif polarization == Polarization.PI:
            interpolated_complex_amplitude_pi = self.get_interpolated_complex_amplitude(x_value,y_value,polarization=Polarization.PI)
            return numpy.abs(interpolated_complex_amplitude_pi)**2
        else:
            raise Exception("Wrong polarization value.")

    # plural (a bit redundant, but kept the same interfacs as wavefront 1D)
    def get_interpolated_complex_amplitudes(self, x_value,y_value, polarization=Polarization.SIGMA):
        return self.get_interpolated_complex_amplitude(x_value, y_value, polarization=polarization)

    def get_interpolated_amplitudes(self, x_value,y_value, polarization=Polarization.SIGMA): # plural!
        return self.get_interpolated_amplitude(x_value,y_value, polarization=polarization)

    def get_interpolated_phases(self, x_value,y_value, polarization=Polarization.SIGMA): # plural!
        return self.get_interpolated_phase(x_value,y_value, polarization=polarization)

    def get_interpolated_intensities(self, x_value,y_value, polarization=Polarization.SIGMA):
        return self.get_interpolated_intensity(x_value,y_value, polarization=polarization)

    def get_interpolated(self,x_value,y_value,toreturn='complex_amplitude'):
        if toreturn == 'complex_amplitude':
            return self.get_interpolated_amplitudes()
        elif toreturn == 'amplitude':
            return self.get_interpolated_amplitudes()
        elif toreturn == 'phase':
            return self.get_interpolated_phases()
        elif toreturn == 'intensity':
            return self.get_interpolated_intensities()
        else:
            raise Exception('Unknown return string')


    # modifiers

    def set_wavelength(self,wavelength):
        self._wavelength = wavelength

    def set_wavenumber(self,wavenumber):
        self._wavelength = 2*numpy.pi / wavenumber

    def set_photon_energy(self,photon_energy):
        m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
        self._wavelength = m2ev / photon_energy

    def set_complex_amplitude(self,complex_amplitude, complex_amplitude_pi=None):
        if self._electric_field_matrix.shape() != complex_amplitude.shape:
            raise Exception("Incompatible shape")
        self._electric_field_matrix.set_z_values(complex_amplitude)

        if complex_amplitude_pi is not None:
            if self.is_polarized():
                if self._electric_field_matrix_pi.shape() != complex_amplitude_pi.shape:
                    raise Exception("Incompatible shape")
                self._electric_field_matrix_pi.set_z_values(complex_amplitude_pi)
            else:
                raise Exception('Cannot set PI-polarized complex amplitude to a non-polarized wavefront.')


    def set_pi_complex_amplitude_to_zero(self):
        if self.is_polarized():
            new_value = self._electric_field_matrix_pi.get_z_values()
            self._electric_field_matrix_pi.set_z_values(new_value*0.0)

    # TODO: add inclination like for 1D
    def set_plane_wave_from_complex_amplitude(self, complex_amplitude=(1.0 + 0.0j)):
        new_value = self._electric_field_matrix.get_z_values()
        new_value *= 0.0
        new_value += complex_amplitude
        self._electric_field_matrix.set_z_values(new_value)
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()


    # TODO: add inclination like for 1D
    def set_plane_wave_from_amplitude_and_phase(self, amplitude=1.0, phase=0.0):
        self.set_plane_wave_from_complex_amplitude(amplitude*numpy.cos(phase) + 1.0j*amplitude*numpy.sin(phase))
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    # TODO: add center like for 1D
    def set_spherical_wave(self,  radius=1.0, complex_amplitude=1.0,):
        """

        :param radius:  Positive radius is divergent wavefront, negative radius is convergent
        :param complex_amplitude:
        :return:
        """
        if radius == 0:
            raise Exception("Radius cannot be zero")
        new_value = complex_amplitude * numpy.exp( -1.0j * self.get_wavenumber() *
                                (self.get_mesh_x()**2 + self.get_mesh_y()**2) / (-2*radius) )

        # new_value = numpy.exp(-1.0j * self.get_wavenumber() *
        #                         (self.get_mesh_x()**2+self.get_mesh_y()**2)/(-2*radius))
        self._electric_field_matrix.set_z_values(new_value)
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    def set_gaussian_hermite_mode(self, sigma_x, sigma_y, nx, ny, amplitude=1.0, center_x=0.0, center_y=0.0,
                                  betax=100.0,betay=100):
        x = self.get_coordinate_x()
        y = self.get_coordinate_y()

        a2D = GaussianSchellModel2D(amplitude, sigma_x, betax*sigma_x, sigma_y, betay*sigma_y)
        Phi = a2D.phi_nm(nx, ny, x-center_x, y-center_y) + 0j
        eigenvalue = a2D.beta(nx,ny)

        self.set_complex_amplitude(numpy.sqrt(eigenvalue)*Phi)
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()

    # note that amplitude is for "amplitude" not for intensity!
    def set_gaussian(self, sigma_x, sigma_y, amplitude=1.0, center_x=0.0, center_y=0.0):
        self.set_gaussian_hermite_mode(sigma_x, sigma_y, 0, 0, amplitude=amplitude, center_x=center_x, center_y=center_y)
        # if polarized, set arbitrary PI component to zero
        self.set_pi_complex_amplitude_to_zero()


    def add_phase_shift(self, phase_shift, polarization=Polarization.SIGMA):
        if polarization == Polarization.SIGMA:
            new_value = self._electric_field_matrix.get_z_values()
            new_value *= numpy.exp(1.0j*phase_shift)
            self._electric_field_matrix.set_z_values(new_value)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                new_value = self._electric_field_matrix_pi.get_z_values()
                new_value *= numpy.exp(1.0j*phase_shift)
                self._electric_field_matrix_pi.set_z_values(new_value)
            else:
                raise Exception("Wavefront is not polarized")
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA or 1=PI are valid)")

    def add_phase_shifts(self, phase_shifts, polarization=Polarization.SIGMA):

        if polarization == Polarization.SIGMA:
            if phase_shifts.shape != self._electric_field_matrix.shape():
                raise Exception("Phase Shifts array has different dimension")
            new_value = self._electric_field_matrix.get_z_values()
            new_value *= numpy.exp(1.0j*phase_shifts)
            self._electric_field_matrix.set_z_values(new_value)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                if phase_shifts.shape != self._electric_field_matrix_pi.shape():
                    raise Exception("Phase Shifts array has different dimension")
                new_value = self._electric_field_matrix_pi.get_z_values()
                new_value *= numpy.exp(1.0j*phase_shifts)
                self._electric_field_matrix_pi.set_z_values(new_value)
            else:
                raise Exception("Wavefront is not polarized")
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA or 1=PI are valid)")

        # if phase_shifts.shape != self._electric_field_matrix.shape():
        #     raise Exception("Phase Shifts array has different dimension")
        # new_value = self._electric_field_matrix.get_z_values()
        # new_value *= numpy.exp(1.0j*phase_shifts)
        # self._electric_field_matrix.set_z_values(new_value)

    def rescale_amplitude(self, factor, polarization=Polarization.SIGMA):

        if polarization == Polarization.SIGMA:
            new_value = self._electric_field_matrix.get_z_values()
            new_value *= factor
            self._electric_field_matrix.set_z_values(new_value)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                new_value = self._electric_field_matrix_pi.get_z_values()
                new_value *= factor
                self._electric_field_matrix_pi.set_z_values(new_value)
            else:
                raise Exception("Wavefront is not polarized")
        elif polarization == Polarization.TOTAL:
            self.rescale_amplitude(factor, polarization=Polarization.SIGMA)
            self.rescale_amplitude(factor, polarization=Polarization.PI)
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA, 1=PI or 3=TOTAL are valid)")

    def rescale_amplitudes(self, factors, polarization=Polarization.SIGMA):

        if polarization == Polarization.SIGMA:
            if factors.shape != self._electric_field_matrix.shape():
                raise Exception("Factors array has different dimension")
            new_value = self._electric_field_matrix.get_z_values()
            new_value *= factors
            self._electric_field_matrix.set_z_values(new_value)
        elif polarization == Polarization.PI:
            if self.is_polarized():
                if factors.shape != self._electric_field_matrix_pi.shape():
                    raise Exception("Factors array has different dimension")
                new_value = self._electric_field_matrix_pi.get_z_values()
                new_value *= factors
                self._electric_field_matrix_pi.set_z_values(new_value)
            else:
                raise Exception("Wavefront is not polarized")
        elif polarization == Polarization.TOTAL:
            self.rescale_amplitudes(factors, polarization=Polarization.SIGMA)
            self.rescale_amplitudes(factors, polarization=Polarization.PI)
        else:
            raise Exception("Invalid polarization value (only 0=SIGMA, 1=PI or 3=TOTAL are valid)")

    def rebin(self,expansion_points_horizontal, expansion_points_vertical, expansion_range_horizontal, expansion_range_vertical,
              keep_the_same_intensity=0,set_extrapolation_to_zero=0):

        x0 = self.get_coordinate_x()
        y0 = self.get_coordinate_y()

        x1 = numpy.linspace(x0[0]*expansion_range_horizontal,x0[-1]*expansion_range_horizontal,int(x0.size*expansion_points_horizontal))
        y1 = numpy.linspace(y0[0]*expansion_range_vertical  ,y0[-1]*expansion_range_vertical,  int(y0.size*expansion_points_vertical))

        X1 = numpy.outer(x1,numpy.ones_like(y1))
        Y1 = numpy.outer(numpy.ones_like(x1),y1)

        z1 = self.get_interpolated_complex_amplitudes(X1,Y1)

        if set_extrapolation_to_zero:
            z1[numpy.where( X1 < x0[0])] = 0.0
            z1[numpy.where( X1 > x0[-1])] = 0.0
            z1[numpy.where( Y1 < y0[0])] = 0.0
            z1[numpy.where( Y1 > y0[-1])] = 0.0


        if keep_the_same_intensity:
            if self.get_intensity(polarization=Polarization.SIGMA).sum() > 1e-10:
                z1 /= numpy.sqrt( (numpy.abs(z1)**2).sum() / self.get_intensity(polarization=Polarization.SIGMA).sum() )
        #
        if self.is_polarized():
            z1_pi = self.get_interpolated_complex_amplitudes(X1,Y1,polarization=Polarization.PI)

            if set_extrapolation_to_zero:
                z1_pi[numpy.where( X1 < x0[0])] = 0.0
                z1_pi[numpy.where( X1 > x0[-1])] = 0.0
                z1_pi[numpy.where( Y1 < y0[0])] = 0.0
                z1_pi[numpy.where( Y1 > y0[-1])] = 0.0

            if keep_the_same_intensity:
                if self.get_intensity(polarization=Polarization.PI).sum() > 1e-10:
                    z1_pi /= numpy.sqrt( (numpy.abs(z1_pi)**2).sum() / self.get_intensity(polarization=Polarization.PI).sum() )
        else:
            z1_pi = None

        z1_pi = None
        new_wf = GenericWavefront2D.initialize_wavefront_from_arrays(x1,y1,z1,z1_pi,wavelength=self.get_wavelength())

        return new_wf

    def clip_window(self,window):

        if self.is_polarized():
            self.rescale_amplitude(window,polarization=Polarization.TOTAL)
        else:
            self.rescale_amplitudes(window,polarization=Polarization.SIGMA)

    # todo: rename to rectangle!
    def clip_square(self, x_min, x_max, y_min, y_max, negative=False, apply_to_wavefront=True):

        if not negative:
            window = numpy.ones(self._electric_field_matrix.shape())
            lower_window_x = numpy.where(self.get_coordinate_x() < x_min)
            upper_window_x = numpy.where(self.get_coordinate_x() > x_max)
            lower_window_y = numpy.where(self.get_coordinate_y() < y_min)
            upper_window_y = numpy.where(self.get_coordinate_y() > y_max)

            if len(lower_window_x) > 0: window[lower_window_x,:] = 0
            if len(upper_window_x) > 0: window[upper_window_x,:] = 0
            if len(lower_window_y) > 0: window[:,lower_window_y] = 0
            if len(upper_window_y) > 0: window[:,upper_window_y] = 0
        else:
            window = numpy.ones(self._electric_field_matrix.shape())
            window2 = numpy.ones_like(window)
            window_x = numpy.where( (x_min <= self.get_coordinate_x()) & ( self.get_coordinate_x() <= x_max))
            window_y = numpy.where( (y_min <= self.get_coordinate_y()) & ( self.get_coordinate_y() <= y_max))

            if len(window_x) > 0: window[window_x,:] = 0.0
            if len(window_y) > 0: window2[:,window_y] = 0.0

            window += window2
            window_good = numpy.where( window > 0 )
            if len(window_good) > 0: window[window_good] = 1.0

        if apply_to_wavefront:
            self.clip_window(window)

        return window


    def clip_circle(self, radius, x_center=0.0, y_center=0.0, negative=False, apply_to_wavefront=True):
        window = numpy.zeros(self._electric_field_matrix.shape())
        X = self.get_mesh_x()
        Y = self.get_mesh_y()
        distance_to_center = numpy.sqrt( (X-x_center)**2 + (Y-y_center)**2 )
        if negative:
            indices_good = numpy.where(distance_to_center >= radius)
        else:
            indices_good = numpy.where(distance_to_center <= radius)
        window[indices_good] = 1.0

        if apply_to_wavefront:
            self.clip_window(window)

        return window

    def clip_ellipse(self, axis_a, axis_b, x_center=0.0, y_center=0.0, negative=False, apply_to_wavefront=True):
        window = numpy.zeros(self._electric_field_matrix.shape())
        X = self.get_mesh_x()
        Y = self.get_mesh_y()

        TESTX= (X - x_center)**2 / (axis_a/2)**2 + (Y - y_center)**2 / (axis_b/2)**2 - 1.0
        TESTX *= -1.0

        if negative:
            indices_good = numpy.where(TESTX  <= 0.0)
        else:
            indices_good = numpy.where(TESTX >= 0.0)
        window[indices_good] = 1.0

        if apply_to_wavefront:
            self.clip_window(window)

        return window


    def is_identical(self,wfr,decimal=7):
        from numpy.testing import assert_array_almost_equal
        try:
            assert_array_almost_equal(self.get_complex_amplitude(),wfr.get_complex_amplitude(),decimal)
            assert(self.is_polarized() == wfr.is_polarized())
            if self.is_polarized():
                assert_array_almost_equal(self.get_complex_amplitude(polarization=Polarization.PI),
                                          wfr.get_complex_amplitude(polarization=Polarization.PI),decimal)
            assert_array_almost_equal(self.get_coordinate_x(),wfr.get_coordinate_x(),decimal)
            assert_array_almost_equal(self.get_coordinate_y(),wfr.get_coordinate_y(),decimal)
            assert_array_almost_equal(self.get_photon_energy(),wfr.get_photon_energy(),decimal)
        except:
            return False

        return True

    # auxiliary function to dump h5 files
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
            x_polarization = self.get_complex_amplitude(polarization=Polarization.SIGMA)       # sigma
            self._dump_arr_2_hdf5(x_polarization.T, "wfr_complex_amplitude_s", filename, subgroupname)

            if self.is_polarized():
                y_polarization = self.get_complex_amplitude(polarization=Polarization.PI)       # pi
                self._dump_arr_2_hdf5(y_polarization.T, "wfr_complex_amplitude_p", filename, subgroupname)


            if intensity:
                if self.is_polarized():
                    self._dump_arr_2_hdf5(self.get_intensity(polarization=Polarization.TOTAL).T,"wfr_intensity",   filename, subgroupname+"/intensity")
                    self._dump_arr_2_hdf5(self.get_intensity(polarization=Polarization.SIGMA).T,"wfr_intensity_s", filename, subgroupname+"/intensity")
                    self._dump_arr_2_hdf5(self.get_intensity(polarization=Polarization.PI).T,"wfr_intensity_p",    filename, subgroupname+"/intensity")
                else:
                    self._dump_arr_2_hdf5(self.get_intensity(polarization=Polarization.SIGMA).T,"wfr_intensity",   filename, subgroupname+"/intensity")

            if phase:
                if self.is_polarized():
                    self._dump_arr_2_hdf5( \
                        self.get_phase(polarization=Polarization.SIGMA).T - self.get_phase(polarization=Polarization.PI).T,
                                        "wfr_phase", filename, subgroupname+"/phase")
                    self._dump_arr_2_hdf5(self.get_phase(polarization=Polarization.SIGMA).T,"wfr_phase_s", filename, subgroupname+"/phase")
                    self._dump_arr_2_hdf5(self.get_phase(polarization=Polarization.PI).T,"wfr_phase_p", filename, subgroupname+"/phase")
                else:
                    self._dump_arr_2_hdf5(self.get_phase(polarization=Polarization.SIGMA).T,"wfr_phase",   filename, subgroupname+"/phase")



            # add mesh and SRW information




            # add mesh and SRW information


            f = h5py.File(filename, 'a')
            f1 = f[subgroupname]

            # point to the default data to be plotted
            f1.attrs['NX_class'] = 'NXentry'
            f1.attrs['default'] = 'intensity'

            # TODO: add self interpreting decoder
            # f1["wfr_method"] = "WOFRY"
            f1["wfr_dimension"] = 2
            f1["wfr_photon_energy"] = self.get_photon_energy()
            x = self.get_coordinate_x()
            y = self.get_coordinate_y()

            f1["wfr_mesh_X"] =  numpy.array([x[0],x[-1],x.size])
            f1["wfr_mesh_Y"] =  numpy.array([y[0],y[-1],y.size])

            # Add NX plot attribites for automatic plot with silx view
            myflags = [intensity,phase]
            mylabels = ['intensity','phase']
            for i,label in enumerate(mylabels):
                if myflags[i]:
                    f2 = f1[mylabels[i]]
                    f2.attrs['NX_class'] = 'NXdata'
                    f2.attrs['signal'] = 'wfr_%s'%(mylabels[i])
                    f2.attrs['axes'] = [b'axis_y', b'axis_x']

                    f3 = f2["wfr_%s"%(mylabels[i])]
                    f3.attrs['interpretation'] = 'image'

                    # X axis data
                    ds = f2.create_dataset('axis_y', data=1e6*y)
                    ds.attrs['units'] = 'microns'
                    ds.attrs['long_name'] = 'Y Pixel Size (microns)'    # suggested X axis plot label

                    # Y axis data
                    ds = f2.create_dataset('axis_x', data=1e6*x)
                    ds.attrs['units'] = 'microns'
                    ds.attrs['long_name'] = 'X Pixel Size (microns)'    # suggested Y axis plot label
            f.close()

        except:
            # TODO: check exit??
            if overwrite is not True:
                raise Exception("Bad input argument")
            os.remove(filename)
            if verbose: print("save_h5_file: file deleted %s"%filename)
            self.save_h5_file(filename,subgroupname, intensity=intensity, phase=phase, overwrite=False)

        if verbose: print("save_h5_file: witten/updated %s data in file: %s"%(subgroupname,filename))

    @classmethod
    def load_h5_file(cls,filename,filepath="wfr"):

        try:
            f = h5py.File(filename, 'r')
            mesh_X = f[filepath+"/wfr_mesh_X"][()]
            mesh_Y = f[filepath+"/wfr_mesh_Y"][()]
            complex_amplitude_s = f[filepath+"/wfr_complex_amplitude_s"][()].T
            try:
                complex_amplitude_p = f[filepath + "/wfr_complex_amplitude_p"][()].T
            except:
                complex_amplitude_p = None
            wfr = cls.initialize_wavefront_from_arrays(
                                x_array=numpy.linspace(mesh_X[0],mesh_X[1],int(mesh_X[2])),
                                y_array=numpy.linspace(mesh_Y[0],mesh_Y[1],int(mesh_Y[2])),
                                z_array=complex_amplitude_s,
                                z_array_pi=complex_amplitude_p)
            wfr.set_photon_energy(f[filepath+"/wfr_photon_energy"][()])
            f.close()
            return wfr
        except:
            raise Exception("Failed to load 2D wavefront to h5 file: "+filename)

if __name__ == "__main__":
    w = GenericWavefront2D.initialize_wavefront_from_steps(polarization=Polarization.TOTAL)
    w.save_h5_file("/tmp/wf.h5",subgroupname="wfr",intensity=True,phase=False,overwrite=True,verbose=True)
    w2 = GenericWavefront2D.load_h5_file("/tmp/wf.h5",filepath="wfr")
    assert(w2.is_identical(w))
    pass
