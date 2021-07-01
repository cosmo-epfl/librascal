from ..utils import BaseIO
from ..lib._rascal.models import (
    compute_numerical_kernel_gradients as _compute_numerical_kernel_gradients,
)
from ..lib._rascal.models.kernels import Kernel as Kernelcpp
from ..lib._rascal.models.kernels import SparseKernel as SparseKernelcpp
from ..neighbourlist import AtomsList
from ..utils import BaseIO
from .sparse_points import SparsePoints
import json


import numpy as np


class Kernel(BaseIO):

    """
    Initialize the kernel with the given representation and parameters

    Parameters
    ----------

    representation : Calculator
        Representation calculator associated with the kernel

    name : string
        Type of kernel, 'Cosine' (aka dot-product) is the default and
        (currently) only option.

    target_type : string
        Type of target (prediction) properties, must be either 'Atom' (the kernel
        is between atomic environments) or 'Structure' (the kernel is summed over
        atoms in a structure), which is the default

    kernel_type : string
        Type of kernel method, either 'Full' (computing exact covariance matrix)
        or 'Sparse' (computing GAP [2]_ like kernel for sparse kernel methods like
        Subset of Regressors)

    Notes
    -----
    In the following
    we refer to the training samples with 'N' and, in the case of
    sparse kernels [1]_, we refer to the pseudo points with 'M'. So a
    kernel between the training samples and the pseudo points is
    'KNM'. For more information on sparse kernels see
    :meth:`rascal.models.krr.train_gap_model`.

    .. [1] Joaquin Quiñonero-Candela, Carl Edward Rasmussen;
            A Unifying View of Sparse Approximate Gaussian Process Regression,
            6(Dec):1939--1959, 2005.
            http://www.jmlr.org/papers/v6/quinonero-candela05a.html

    .. [2] Ceriotti, M., Willatt, M. J., & Csányi, G. (2018).
    Machine Learning of Atomic-Scale Properties Based on Physical Principles.
    In Handbook of Materials Modeling (pp. 1–27). Springer, Cham.
    https://doi.org/10.1007/978-3-319-42913-7_68-1

    """

    def __init__(
        self,
        representation,
        name="Cosine",
        kernel_type="Full",
        target_type="Structure",
        **kwargs
    ):
        # This case cannot be handled by the c++ side because c++ cannot deduce the
        # type from arguments inside a json, so it has to be casted in the c++
        # side. Therefore zeta has to be checked here.
        if name == "Cosine" and "zeta" in kwargs:
            # should be positive integer
            zeta = kwargs["zeta"]
            if not (zeta > 0 and isinstance(zeta, int)):
                raise ValueError("The given zeta has to be a positive integer.")
        elif name == "GAP" and "zeta" in kwargs:
            # should be positive integer
            zeta = kwargs["zeta"]
            if not (zeta > 0 and isinstance(zeta, int)):
                raise ValueError("The given zeta has to be a positive integer.")
        else:
            raise RuntimeError("Kernel name must be one of: Cosine, GAP.")
        hypers = dict(name=name, target_type=target_type)
        hypers.update(**kwargs)
        self._rep = representation
        self._representation = representation._representation
        self.name = name
        self._kwargs = kwargs
        self.kernel_type = kernel_type
        self.target_type = target_type
        if "Sparse" in kernel_type:
            self._kernel = SparseKernelcpp(hypers)
        else:
            self._kernel = Kernelcpp(hypers)

    def _get_init_params(self):
        init_params = dict(
            representation=self._rep,
            name=self.name,
            kernel_type=self.kernel_type,
            target_type=self.target_type,
        )
        init_params.update(**self._kwargs)
        return init_params

    def _set_data(self, data):
        super()._set_data(data)

    def _get_data(self):
        return super()._get_data()

    # I think not needed
    #def _set_data(self, data):
    #    super()._set_data(data)
    #    self._kernel = self._kernel.from_dict(data["kernel"])

    #def _get_data(self):
    #    data = super()._get_data()
    #    data.update(kernel=self._kernel.to_dict())
    #    return data

    def __call__(self, X, Y=None, grad=(False, False), compute_neg_stress=False):
        """
        Compute the kernel.

        Parameters
        ----------
            X : AtomList or ManagerCollection (C++ class)
                Container of atomic structures.

            Y : AtomList, ManagerCollection or SparsePoints* (C++ class).

            grad : tuple specifying if the kernel should be computed using
                gradient
                of the representation w.r.t. the atomic positions, e.g. (True,
                False) corresponds to the gradient of the 1st argument of the
                kernel w.r.t. the atomic positions.

            compute_neg_stress : if gradients are computed and True then compute
                also the kernel associated with the stress in Voigt format.

        Returns
        -------
            kernel_matrix: ndarray
        """
        if isinstance(X, AtomsList):
            X = X.managers
        if Y is None and grad == (False, False):
            # compute a kernel between features and themselves
            if self.kernel_type == "Full":
                return self._kernel.compute(self._representation, X)
            elif self.kernel_type == "Sparse":
                if isinstance(X, SparsePoints):
                    X = X._sparse_points
                # compute the KMM matrix
                return self._kernel.compute(X)
        elif grad == (True, False) and "Sparse" in self.kernel_type:
            if isinstance(Y, SparsePoints):
                Y = Y._sparse_points
            # compute the block of the KNM matrix corresponding to forces
            return self._kernel.compute_derivative(
                self._representation, X, Y, compute_neg_stress
            )
        elif grad == (False, False):
            # compute the kernel between two sets of features
            if isinstance(Y, AtomsList):
                # to make predictions with the full covariance
                Y = Y.managers
            elif isinstance(Y, SparsePoints):
                # to make predictions with a sparse kernel method
                Y = Y._sparse_points
            return self._kernel.compute(self._representation, X, Y)
        else:
            raise NotImplementedError(
                "The configuration: {} is not implemented for kernel {} in {} mode.".format(
                    grad, self.name, self.kernel_type
                )
            )


def compute_numerical_kernel_gradients(
    kernel, calculator, managers, sparse_points, h_disp, compute_neg_stress=True
):
    """This function is used for testing the numerical kernel gradient"""
    return _compute_numerical_kernel_gradients(
        kernel._kernel,
        calculator._representation,
        managers.managers,
        sparse_points._sparse_points,
        h_disp,
        compute_neg_stress,
    )
