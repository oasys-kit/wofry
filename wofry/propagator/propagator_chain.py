###################################################################
# DO NOT TOUCH THIS CODE -- BEGIN
###################################################################
import threading

def synchronized_method(method):

    outer_lock = threading.Lock()
    lock_name = "__"+method.__name__+"_lock"+"__"

    def sync_method(self, *args, **kws):
        with outer_lock:
            if not hasattr(self, lock_name): setattr(self, lock_name, threading.Lock())
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)

    return sync_method

class Singleton:

    def __init__(self, decorated):
        self._decorated = decorated

    @synchronized_method
    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)

###################################################################
# DO NOT TOUCH THIS CODE -- END
###################################################################

from syned.beamline.beamline import Beamline
from wofry.propagator.wavefront import Wavefront

class PropagationHandlers:
    SRXRAYLIB = "SRXRAYLIB"
    SRW = "SRW"
    WISE = "WISE"

class PropagationParameters(object):
    def __init__(self,
                 wavefront = Wavefront(),
                 beamline = Beamline()):
        self._wavefront = wavefront
        self._beamline = beamline

    def get_wavefront(self):
        return self._wavefront

    def get_beamline(self):
        return self._beamline

class AbstractPropagator(object):

    def __init__(self):
        super().__init__()

    def get_handler_name(self):
        raise NotImplementedError("This method is abstract")

    def is_handler(self, handler_name):
        return handler_name == self.get_handler_name()

    def do_propagation(self, parameters=PropagationParameters()):
        raise NotImplementedError("This method is abstract" +
                                  "\n\naccepts " + PropagationParameters.__module__ + "." + PropagationParameters.__name__ +
                                  "\nreturns " + Wavefront.__module__ + "." + Wavefront.__name__)


@Singleton
class PropagatorsChain(object):
    def __init__(self):
       self.__propagators_chain = []

    def add_propagator(self, propagator=AbstractPropagator()):
        if propagator is None: raise ValueError("Given propagator is None")
        if not isinstance(propagator, AbstractPropagator): raise ValueError("Given propagator is not a compatibile object")

        for existing in self.__propagators_chain:
            if existing.is_handler(propagator.get_handler_name()):
                raise ValueError("Propagator already in the Chain")

        self.__propagators_chain.append(propagator)

    def do_propagation(self, propagation_parameters, handler_name):
        for propagator in self.__propagators_chain:
            if propagator.is_handler(handler_name):
                return propagator.do_propagation(parameters=propagation_parameters)

        return None

