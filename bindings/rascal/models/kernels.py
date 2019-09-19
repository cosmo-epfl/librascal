# from ..lib.Kernels import CosineKernel
from ..lib._rascal.Models.Kernels import Kernel as Kernelcpp
from ..neighbourlist import AtomsList
import json

class Kernel(object):
    """
    Computes the kernel from an representation

    Attributes
    ----------

    Methods
    -------
    """
    def __init__(self, representation, name='Cosine', target_type='structure', **kwargs):
        """
        Compute the kernel.

        Parameters
        ----------
        representation : the representation used to produce the features that
            should be used to compute the kernel
        name : name of the kernel
        target_type : 'structure' if the target property is for the whole
            structure and 'atom' if it refers to atomic properties

        kwargs : additional arguments for the kernel. depends on the kernel

        """
        hypers = dict(name=name,target_type=target_type)
        hypers.update(**kwargs)
        hypers_str = json.dumps(hypers)
        self._representation = representation._representation

        self._kernel = Kernelcpp(hypers_str)


    def __call__(self, X, Y=None):
        """
        Compute the kernel.

        Parameters
        ----------
        X : AtomList or ManagerCollection (C++ class)
            Container of atomic structures.

        Returns
        -------
        kernel_matrix: ndarray
        """
        if Y is None:
            Y = X
        if isinstance(X, AtomsList):
            X = X.managers
        if isinstance(Y, AtomsList):
            Y = Y.managers
        return self._kernel.compute(self._representation, X, Y)
