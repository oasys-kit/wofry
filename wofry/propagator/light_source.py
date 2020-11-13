import numpy

from wofry.beamline.decorators import LightSourceDecorator
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from syned.storage_ring.light_source import LightSource


class WOLightSource(LightSource, LightSourceDecorator):
    def __init__(self,
                 name                = "Undefined",
                 electron_beam       = None,
                 magnetic_structure  = None,
                 dimension           = 1,
                 initialize_from     = 0,
                 range_from_h        = -1e-6,
                 range_to_h          = 1e-6,
                 range_from_v        = -1e-6,
                 range_to_v          = 1e-6,
                 steps_start_h       = -1e-6,
                 steps_step_h        = 1e-8,
                 steps_start_v       = -1e-6,
                 steps_step_v        = 1e-8,
                 number_of_points_h  = 200,
                 number_of_points_v  = 100,
                 energy              = 10000.0,
                 sigma_h             = 1e-7,
                 sigma_v             = 1e-7,
                 amplitude           = 1.0,
                 kind_of_wave        = 0, #  0=plane, 1=spherical, 2=Gaussian, 3=Gaussian-Hermite
                 n_h                 = 0,
                 n_v                 = 0,
                 beta_h              = 1.0,
                 beta_v              = 1.0,
                 units               = 0,
                 wavelength          = 1e-10,
                 initialize_amplitude = 0,
                 complex_amplitude_re = 1.0,
                 complex_amplitude_im = 0.0,
                 phase                = 0.0,
                 radius               = 100.0,
                 center               = 0.0,
                 inclination          = 0.0,
                 gaussian_shift       = 0.0,
                 add_random_phase     = 0,
                 ):

        super().__init__(name=name, electron_beam=electron_beam, magnetic_structure=magnetic_structure)

        self._dimension =  dimension
        self._initialize_from =  initialize_from
        self._range_from_h =  range_from_h
        self._range_to_h =  range_to_h
        self._range_from_v =  range_from_v
        self._range_to_v =  range_to_v
        self._steps_start_h =  steps_start_h
        self._steps_step_h =  steps_step_h
        self._steps_start_v =  steps_start_v
        self._steps_step_v =  steps_step_v
        self._number_of_points_h =  number_of_points_h
        self._number_of_points_v =  number_of_points_v
        self._energy =  energy
        self._sigma_h =  sigma_h
        self._sigma_v =  sigma_v
        self._amplitude =  amplitude
        self._kind_of_wave =  kind_of_wave
        self._n_h =  n_h
        self._n_v =  n_v
        self._beta_h =  beta_h
        self._beta_v =  beta_v

        self._units = units
        self._wavelength = wavelength
        self._initialize_amplitude = initialize_amplitude
        self._complex_amplitude_re = complex_amplitude_re
        self._complex_amplitude_im = complex_amplitude_im
        self._phase = phase
        self._radius = radius
        self._center = center
        self._inclination = inclination
        self._gaussian_shift = gaussian_shift
        self._add_random_phase = add_random_phase

        self.dimension = dimension
        self._set_support_text([
                    # ("name"      ,           "to define ", "" ),
                    ("dimension"      , "dimension ", "" ),
            ] )

    def get_dimension(self):
        return self._dimension

    # from Wofry Decorator
    def get_wavefront(self):

        #
        # If making changes here, don't forget to do changes in to_python_code() as well...
        #
        if self._initialize_from == 0:
            if self._dimension == 1:
                wf = GenericWavefront1D.initialize_wavefront_from_range(
                    x_min=self._range_from_h, x_max=self._range_to_h,
                    number_of_points=self._number_of_points_h)
            elif self._dimension == 2:
                wf = GenericWavefront2D.initialize_wavefront_from_range(
                    x_min=self._range_from_h, x_max=self._range_to_h,
                    y_min=self._range_from_v, y_max=self._range_to_v,
                    number_of_points=(self._number_of_points_h, self._number_of_points_v))
        else:
            if self._dimension == 1:
                wf = GenericWavefront1D.initialize_wavefront_from_steps(
                    x_start=self._steps_start_h, x_step=self._steps_step_h,
                    number_of_points=self._number_of_points_h)
            elif self._dimension == 2:
                wf = GenericWavefront2D.initialize_wavefront_from_steps(
                    x_start=self._steps_start_h, x_step=self._steps_step_h,
                    y_start=self._steps_start_v, y_step=self._steps_step_v,
                    number_of_points=(self._number_of_points_h, self._number_of_points_v))


        if self._units == 0:
            wf.set_photon_energy(self._energy)
        else:
            wf.set_wavelength(self._wavelength)

        if self._kind_of_wave == 0:  # plane
            if self._dimension == 1:
                if self._initialize_amplitude == 0:
                    wf.set_plane_wave_from_complex_amplitude(complex_amplitude=complex(
                        self._complex_amplitude_re, self._complex_amplitude_im), inclination=self._inclination)
                else:
                    wf.set_plane_wave_from_amplitude_and_phase(amplitude=self._amplitude, phase=self._phase,
                                                                             inclination=self._inclination)
            elif self._dimension == 2:
                if self._initialize_amplitude == 0:
                    wf.set_plane_wave_from_complex_amplitude(complex_amplitude=complex(
                        self._complex_amplitude_re, self._complex_amplitude_im))
                else:
                    wf.set_plane_wave_from_amplitude_and_phase(amplitude=self._amplitude, phase=self._phase)

        elif self._kind_of_wave == 1:  # spheric
            if self._dimension == 1:
                wf.set_spherical_wave(radius=self._radius, center=self._center,
                                                    complex_amplitude=complex(self._complex_amplitude_re,
                                                                              self._complex_amplitude_im))
            elif self._dimension == 2:
                wf.set_spherical_wave(radius=self._radius,
                                                    complex_amplitude=complex(self._complex_amplitude_re,
                                                                              self._complex_amplitude_im))

        elif self._kind_of_wave == 2:  # gaussian
            if self._dimension == 1:
                wf.set_gaussian(sigma_x=self._sigma_h, amplitude=self._amplitude, shift=self._gaussian_shift)
            elif self._dimension == 2:
                wf.set_gaussian(sigma_x=self._sigma_h, sigma_y=self._sigma_v, amplitude=self._amplitude)
        elif self._kind_of_wave == 3:  # g.s.m.
            if self._dimension == 1:
                wf.set_gaussian_hermite_mode(
                                            sigma_x=self._sigma_h,
                                            mode_x=self._n_h,
                                            amplitude=self._amplitude,
                                            beta=self._beta_h,
                                            shift=self._gaussian_shift,
                                            )
            elif self._dimension == 2:
                wf.set_gaussian_hermite_mode(
                                            sigma_x=self._sigma_h,
                                            sigma_y=self._sigma_v,
                                            amplitude=self._amplitude,
                                            nx=self._n_h,
                                            ny=self._n_v,
                                            betax=self._beta_h,
                                            betay=self._beta_v,
                                            )

        if self._add_random_phase:
            wf.add_phase_shifts(2 * numpy.pi * numpy.random.random(wf.size()))


        return wf

    def to_python_code(self, do_plot=True, add_import_section=False):

        txt = ""

        txt += "#"
        txt += "\n# create output_wavefront\n#"
        txt += "\n#"

        if self._dimension == 1:
            if add_import_section: txt += "\nfrom wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D"

            if self._initialize_from == 0:
                txt += "\noutput_wavefront = GenericWavefront1D.initialize_wavefront_from_range(x_min=%g,x_max=%g,number_of_points=%d)"%\
                (self._range_from_h,self._range_to_h,self._number_of_points_h)

            else:
                txt += "\noutput_wavefront = GenericWavefront1D.initialize_wavefront_from_steps(x_start=%g, x_step=%g,number_of_points=%d)"%\
                       (self._steps_start_h,self._steps_step_h,self._number_of_points_h)
        elif self._dimension == 2:
            if add_import_section: txt += "\nfrom wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D"

            if self._initialize_from == 0:
                txt += "\noutput_wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=%g,x_max=%g,y_min=%g,y_max=%g,number_of_points=(%d,%d))" % \
                       (self._range_from_h, self._range_to_h, self._range_from_v,  self._range_to_v, self._number_of_points_h, self._number_of_points_v)

            else:
                txt += "\noutput_wavefront = GenericWavefront2D.initialize_wavefront_from_steps(x_start=%g,x_step=%g,y_start=%g,y_step=%g,number_of_points=(%d,%d))" % \
                       (self._steps_start_h, self._steps_step_h, self._steps_start_v, self._steps_step_v, self._number_of_points_h, self._number_of_points_v)

        if self._units == 0:
            txt += "\noutput_wavefront.set_photon_energy(%g)"%(self._energy)
        else:
            txt += "\noutput_wavefront.set_wavelength(%g)"%(self._wavelength)



        if self._kind_of_wave == 0: #plane
            if self._dimension == 1:
                if self._initialize_amplitude == 0:
                    txt += "\noutput_wavefront.set_plane_wave_from_complex_amplitude(complex_amplitude=complex(%g,%g),inclination=%g)"%\
                           (self._complex_amplitude_re, self._complex_amplitude_im, self._inclination)
                else:
                    txt += "\noutput_wavefront.set_plane_wave_from_amplitude_and_phase(amplitude=%g,phase=%g,inclination=%g)"%(self.amplitude,self.phase,self.inclination)
            elif self._dimension == 2:
                if self._initialize_amplitude == 0:
                    txt += "\noutput_wavefront.set_plane_wave_from_complex_amplitude(complex_amplitude=complex(%g,%g))"%\
                           (self._complex_amplitude_re, self._complex_amplitude_im)
                else:
                    txt += "\noutput_wavefront.set_plane_wave_from_amplitude_and_phase(amplitude=%g,phase=%g)"%(self.amplitude,self.phase)

        elif self._kind_of_wave == 1: # spheric
            if self._dimension == 1:
                txt += "\noutput_wavefront.set_spherical_wave(radius=%g,center=%g,complex_amplitude=complex(%g, %g))"%\
                       (self._radius, self._center, self._complex_amplitude_re, self._complex_amplitude_im)
            elif self._dimension == 2:
                txt += "\noutput_wavefront.set_spherical_wave(radius=%g,complex_amplitude=complex(%g, %g))"%\
                       (self._radius, self._complex_amplitude_re, self._complex_amplitude_im)

        elif self._kind_of_wave == 2: # gaussian
            if self._dimension == 1:
                txt += "\noutput_wavefront.set_gaussian(sigma_x=%g, amplitude=%g,shift=%g)"%\
                       (self._sigma_h, self._amplitude, self._gaussian_shift)
            elif self._dimension == 2:
                txt += "\noutput_wavefront.set_gaussian(sigma_x=%g, sigma_y=%g, amplitude=%g)"%\
                       (self._sigma_h, self._sigma_v, self._amplitude)

        elif self._kind_of_wave == 3: # g.s.m.
            if self._dimension == 1:
                txt += "\noutput_wavefront.set_gaussian_hermite_mode(sigma_x=%g,amplitude=%g,mode_x=%d,shift=%g,beta=%g)"%\
                       (self._sigma_h, self._amplitude, self._n_h, self._gaussian_shift, self._beta_h)
            elif self._dimension == 2:
                txt += "\noutput_wavefront.set_gaussian_hermite_mode(sigma_x=%g,sigma_y=%g,amplitude=%g,nx=%d,ny=%d,betax=%g,betay=%g)"%\
                       (self._sigma_h, self._sigma_v, self._amplitude, self._n_h, self._n_v, self._beta_h, self._beta_v)

        if self._dimension == 1:
            if self._add_random_phase:
                txt += "\noutput_wavefront.add_phase_shifts(2*numpy.pi*numpy.random.random(output_wavefront.size()))"

        if do_plot:
            if self.get_dimension() == 1:
                if add_import_section: txt += "\n\n\nfrom srxraylib.plot.gol import plot"
                txt += "\n\n\nplot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='SOURCE')"
            elif self._dimension == 2:
                if add_import_section: txt += "\n\n\nfrom srxraylib.plot.gol import plot_image"
                txt += "\n\n\nplot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='SOURCE')"

        return txt
