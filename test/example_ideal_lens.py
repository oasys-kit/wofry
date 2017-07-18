"""

    This example shows the focusing of an ideal lens in 1:1 configuration
    for different sources (see main program at the bottom)

    The systems are:
        'convergent spherical'
        'divergent spherical with lens'
        'plane with lens'
        'Gaussian with lens'
        'Hermite with lens'
        'Undulator with lens'

"""

USE_PROPAGATOR = 'fft' # possible values: 'fft' 'convolution' 'srw'

import numpy

from scipy.special import hermite
import scipy.constants as codata

# this is undulator block
from pySRU.ElectronBeam import ElectronBeam
from pySRU.MagneticStructureUndulatorPlane import MagneticStructureUndulatorPlane
from pySRU.TrajectoryFactory import TrajectoryFactory, TRAJECTORY_METHOD_ANALYTIC,TRAJECTORY_METHOD_ODE
from pySRU.RadiationFactory import RadiationFactory,RADIATION_METHOD_NEAR_FIELD, RADIATION_METHOD_APPROX_FARFIELD
from pySRU.SourceUndulatorPlane import SourceUndulatorPlane

# source
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
# beamline
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement
from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen as Screen
from wofry.beamline.optical_elements.ideal_elements.lens import WOIdealLens, WOIdealLens1D
# propagator
from wofry.propagator.propagator import PropagationManager, PropagationParameters
from wofry.propagator.propagator import PropagationElements

propagator = PropagationManager.Instance()
if USE_PROPAGATOR == 'fft':
    from wofry.propagator.propagators2D.fresnel import Fresnel2D
    propagator.add_propagator(Fresnel2D())
if USE_PROPAGATOR == 'convolution':
    from wofry.propagator.propagators2D.fresnel import FresnelConvolution2D
    propagator.add_propagator(FresnelConvolution2D())

if USE_PROPAGATOR == 'srw':
    from wofrysrw.propagator.propagators2D.srw_fresnel import FresnelSRW
    propagator.add_propagator(FresnelSRW())


# global variables
codata_mee = numpy.array(codata.physical_constants["electron mass energy equivalent in MeV"][0])
m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)

do_plot = True

if do_plot:
    from srxraylib.plot.gol import plot,plot_image,plot_table

#
# auxiliar functions
#
def line_fwhm(line):
    #
    #CALCULATE fwhm in number of abscissas bins (supposed on a regular grid)
    #
    tt = numpy.where(line>=max(line)*0.5)
    if line[tt].size > 1:
        # binSize = x[1]-x[0]
        FWHM = (tt[0][-1]-tt[0][0])
        return FWHM
    else:
        return -1


def propagation_in_vacuum(wf,propagation_distance=30.0,defocus_factor=1.0,propagation_steps=1):
    #
    # define image plane
    #
    propagation_elements = PropagationElements()
    #

    if propagation_steps == 1:
        propagation_elements.add_beamline_element(BeamlineElement(optical_element=Screen(),
                                                              coordinates=ElementCoordinates(p=0, q=propagation_distance)))
    else:
        for i in range(propagation_steps):
            propagation_elements.add_beamline_element(BeamlineElement(optical_element=Screen(),
                                                              coordinates=ElementCoordinates(p=0, q=propagation_distance/propagation_steps)))
    propagation_parameters = PropagationParameters(wavefront=wf,
                                                   propagation_elements=propagation_elements)


    if USE_PROPAGATOR == 'fft':
        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        wf1 = propagator.do_propagation(propagation_parameters, Fresnel2D.HANDLER_NAME)
    elif USE_PROPAGATOR == 'srw':
        propagation_parameters.set_additional_parameters("srw_autosetting", False)
        wf1 = propagator.do_propagation(propagation_parameters, FresnelSRW.HANDLER_NAME)
    elif USE_PROPAGATOR == 'convolution':
        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        wf1 = propagator.do_propagation(propagation_parameters, FresnelConvolution2D.HANDLER_NAME)
    else:
        raise Exception("Not implemented method: %s"%USE_PROPAGATOR)

    horizontal_profile = wf1.get_intensity()[:,int(wf1.size()[1]/2)]
    horizontal_profile /= horizontal_profile.max()
    print("FWHM of the horizontal profile: %g um"%(1e6*line_fwhm(horizontal_profile)*wf1.delta()[0]))
    vertical_profile = wf1.get_intensity()[int(wf1.size()[0]/2),:]
    vertical_profile /= vertical_profile.max()
    print("FWHM of the vertical profile: %g um"%(1e6*line_fwhm(vertical_profile)*wf1.delta()[1]))

    print("Output intensity: ",wf1.get_intensity().sum())
    return wf1,wf1.get_coordinate_x(),horizontal_profile


def apply_lens(wf,focal_length):
    propagation_elements = PropagationElements()
    propagation_elements.add_beamline_element(BeamlineElement(optical_element=
        WOIdealLens("IdealLens",focal_x=focal_length, focal_y=focal_length),
        coordinates=ElementCoordinates(p=0.0, q=0.0)))

    propagation_parameters = PropagationParameters(wavefront=wf,
                                                   propagation_elements=propagation_elements)

    if USE_PROPAGATOR == 'fft':
        wfout = propagator.do_propagation(propagation_parameters, Fresnel2D.HANDLER_NAME)
    elif USE_PROPAGATOR == 'convolution':
        wfout = propagator.do_propagation(propagation_parameters, FresnelConvolution2D.HANDLER_NAME)

    return wfout

#
# main function
#

def main(mode_wavefront_before_lens):


    #                               \ |  /
    #   *                           | | |                      *
    #                               / | \
    #   <-------    d  ---------------><---------   d   ------->
    #   d is propagation_distance
    # wavefron names at different positions
    #   wf1                     wf2     wf3                   wf4


    lens_diameter = 0.002 # 0.001 # 0.002

    if mode_wavefront_before_lens == 'Undulator with lens':
        npixels_x = 512
    else:
        npixels_x = int(2048*1.5)

    pixelsize_x = lens_diameter / npixels_x
    print("pixelsize: ",pixelsize_x)


    pixelsize_y = pixelsize_x
    npixels_y = npixels_x

    wavelength = 1.24e-10
    propagation_distance = 30.0
    defocus_factor = 1.0 # 1.0 is at focus
    propagation_steps = 1

    # for Gaussian source
    sigma_x = lens_diameter / 400 # 5e-6
    sigma_y = sigma_x # 5e-6
    # for Hermite-Gauss, the H and V mode index (start from 0)
    hm = 3
    hn = 1


    if mode_wavefront_before_lens == 'convergent spherical':
        # no need to propagate nor define lens
        wf3 = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)
        wf3.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)

    elif mode_wavefront_before_lens == 'divergent spherical with lens':
        # define wavefront at zero distance upstream the lens and apply lens

        focal_length = propagation_distance / 2.

        wf2 = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        wf2.set_spherical_wave(complex_amplitude=1.0,radius=propagation_distance)
        wf3 = apply_lens(wf2,focal_length)


    elif mode_wavefront_before_lens == 'plane with lens':
        # define wavefront at zero distance upstream the lens and apply lens
        focal_length = propagation_distance

        wf2 = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)


        wf2.set_plane_wave_from_complex_amplitude(1.0+0j)

        wf3 = apply_lens(wf2,focal_length)


    elif mode_wavefront_before_lens == 'Gaussian with lens':
        # define wavefront at source point, propagate to the lens and apply lens

        wf1 = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        X = wf1.get_mesh_x()
        Y = wf1.get_mesh_y()

        intensity = numpy.exp( - X**2/(2*sigma_x**2)) * numpy.exp( - Y**2/(2*sigma_y**2))


        wf1.set_complex_amplitude( numpy.sqrt(intensity) )

        # plot

        plot_image(wf1.get_intensity(),1e6*wf1.get_coordinate_x(),1e6*wf1.get_coordinate_y(),
                   xtitle="X um",ytitle="Y um",title="Gaussian source",show=1)

        wf2, x2, h2 = propagation_in_vacuum(wf1,propagation_distance=propagation_distance)


        plot_image(wf2.get_intensity(),1e6*wf2.get_coordinate_x(),1e6*wf2.get_coordinate_y(),
                   xtitle="X um",ytitle="Y um",title="Before lens fft",show=1)


        focal_length = propagation_distance / 2

        wf3 = apply_lens(wf2,focal_length)

    elif mode_wavefront_before_lens == 'Hermite with lens':
        # define wavefront at source point, propagate to the lens and apply lens


        wf1 = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        X = wf1.get_mesh_x()
        Y = wf1.get_mesh_y()

        efield =     (hermite(hm)(numpy.sqrt(2)*X/sigma_x)*numpy.exp(-X**2/sigma_x**2))**2 \
                   * (hermite(hn)(numpy.sqrt(2)*Y/sigma_y)*numpy.exp(-Y**2/sigma_y**2))**2

        wf1.set_complex_amplitude( efield )

        # plot

        plot_image(wf1.get_intensity(),1e6*wf1.get_coordinate_x(),1e6*wf1.get_coordinate_y(),
                   xtitle="X um",ytitle="Y um",title="Hermite-Gauss source",show=1)


        wf2, x2, h2 = propagation_in_vacuum(wf1,propagation_distance=30.0,defocus_factor=1.0,propagation_steps=1)

        plot_image(wf2.get_intensity(),1e6*wf2.get_coordinate_x(),1e6*wf2.get_coordinate_y(),
                   xtitle="X um",ytitle="Y um",title="Before lens %s"%USE_PROPAGATOR,show=1)


        wf3 = apply_lens(wf2,focal_length=propagation_distance / 2)

    elif mode_wavefront_before_lens == 'Undulator with lens':

        beamline = {}
        # beamline['name'] = "ESRF_NEW_OB"
        # beamline['ElectronBeamDivergenceH'] = 5.2e-6    # these values are not used (zero emittance)
        # beamline['ElectronBeamDivergenceV'] = 1.4e-6    # these values are not used (zero emittance)
        # beamline['ElectronBeamSizeH'] = 27.2e-6         # these values are not used (zero emittance)
        # beamline['ElectronBeamSizeV'] = 3.4e-6          # these values are not used (zero emittance)
        # beamline['ElectronEnergySpread'] = 0.001        # these values are not used (zero emittance)
        beamline['ElectronCurrent'] = 0.2
        beamline['ElectronEnergy']  = 6.0
        beamline['Kv']              = 1.68  # 1.87
        beamline['NPeriods']        = 111   # 14
        beamline['PeriodID']        = 0.018 # 0.035
        beamline['distance']        =   propagation_distance
        # beamline['gapH']      = pixelsize_x*npixels_x
        # beamline['gapV']      = pixelsize_x*npixels_x

        gamma = beamline['ElectronEnergy'] / (codata_mee * 1e-3)
        print ("Gamma: %f \n"%(gamma))

        resonance_wavelength = (1 + beamline['Kv']**2 / 2.0) / 2 / gamma**2 * beamline["PeriodID"]
        resonance_energy = m2ev / resonance_wavelength


        print ("Resonance wavelength [A]: %g \n"%(1e10*resonance_wavelength))
        print ("Resonance energy [eV]: %g \n"%(resonance_energy))

        # red shift 100 eV
        resonance_energy = resonance_energy - 100


        myBeam = ElectronBeam(Electron_energy=beamline['ElectronEnergy'], I_current=beamline['ElectronCurrent'])
        myUndulator = MagneticStructureUndulatorPlane(K=beamline['Kv'], period_length=beamline['PeriodID'],
                            length=beamline['PeriodID']*beamline['NPeriods'])


        wf2 = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        XX = wf2.get_mesh_x()
        YY = wf2.get_mesh_y()
        X =  wf2.get_coordinate_x()
        Y =  wf2.get_coordinate_y()

        source = SourceUndulatorPlane(undulator=myUndulator,
                            electron_beam=myBeam, magnetic_field=None)
        omega = resonance_energy * codata.e / codata.hbar
        Nb_pts_trajectory = int(source.choose_nb_pts_trajectory(0.01,photon_frequency=omega))
        print("Number of trajectory points: ",Nb_pts_trajectory)

        traj_fact = TrajectoryFactory(Nb_pts=Nb_pts_trajectory,method=TRAJECTORY_METHOD_ODE,
                                      initial_condition=None)

        print("Number of trajectory points: ",traj_fact.Nb_pts)

        if (traj_fact.initial_condition == None):
            traj_fact.initial_condition = source.choose_initial_contidion_automatic()

        print("Number of trajectory points: ",traj_fact.Nb_pts,traj_fact.initial_condition)

        rad_fact = RadiationFactory(method=RADIATION_METHOD_NEAR_FIELD, photon_frequency=omega)


        #print('step 3')
        trajectory = traj_fact.create_from_source(source=source)


        #print('step 4')
        radiation = rad_fact.create_for_one_relativistic_electron(trajectory=trajectory, source=source,
                            XY_are_list=False,distance=beamline['distance'], X=X, Y=Y)

        efield = rad_fact.calculate_electrical_field(trajectory=trajectory,source=source,
                            distance=beamline['distance'],X_array=XX,Y_array=YY)

        tmp = efield.electrical_field()[:,:,0]


        wf2.set_photon_energy(resonance_energy)
        wf2.set_complex_amplitude( tmp )


        # plot

        plot_image(wf2.get_intensity(),1e6*wf2.get_coordinate_x(),1e6*wf2.get_coordinate_y(),
                   xtitle="X um",ytitle="Y um",title="UND source at lens plane",show=1)

        # apply lens

        focal_length = propagation_distance / 2

        wf3 = apply_lens(wf2,focal_length=focal_length)

    else:
        raise Exception("Unknown mode")


    plot_image(wf3.get_phase(),1e6*wf3.get_coordinate_x(),1e6*wf3.get_coordinate_y(),
               title="Phase just after the lens %s"%USE_PROPAGATOR,xtitle="X um",ytitle="Y um",show=1)

    wf4, x4, h4 = propagation_in_vacuum(wf3,propagation_distance=propagation_distance,defocus_factor=1.0,propagation_steps=1)

    plot_image(wf4.get_intensity(),1e6*wf4.get_coordinate_x(),1e6*wf4.get_coordinate_y(),
               title="Intensity at focal point %s"%USE_PROPAGATOR,xtitle="X um",ytitle="Y um",show=1)

    plot(1e4*x4,h4,xtitle='x [um]',ytitle='intensity',title='horizontal profile',show=True)


if __name__ == "__main__":

    mode_wavefront_before_lens = 'convergent spherical'
    # mode_wavefront_before_lens = 'divergent spherical with lens'
    # mode_wavefront_before_lens = 'plane with lens'
    # mode_wavefront_before_lens = 'Gaussian with lens'
    # mode_wavefront_before_lens = 'Hermite with lens'
    # mode_wavefront_before_lens = 'Undulator with lens'

    main(mode_wavefront_before_lens)