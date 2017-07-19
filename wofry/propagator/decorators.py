
class WavefrontDecorator():

    def get_dimension(self):
        raise NotImplementedError()

    def toGenericWavefront(self):
        raise NotImplementedError()

    @classmethod
    def fromGenericWavefront(cls, wavefront):
        raise NotImplementedError()