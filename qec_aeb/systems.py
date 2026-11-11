r"""
Coupled resonator--ancilla systems, built as scqubits
``HilbertSpace`` objects and packaged as the keyword arguments of a
``FlexibleSweep``.

In the identifiers, ``res`` is the storage resonator and ``qubit`` is the
ancilla. Energies and coupling strengths are in GHz.
"""

__all__ = [
    'ResonatorTransmon',
    'ResonatorFluxonium',
    'FluxoniumResonatorFluxonium',
]

import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

import numpy as np

from abc import ABC, abstractproperty
from typing import Dict, List, Tuple, Union, Any, Callable
import copy

class JointSystems(ABC):
    r"""
    Base class of a coupled system: a ``HilbertSpace``, its subsystems, and the
    functions that update it from named parameters.

    A subclass builds the subsystems and the ``HilbertSpace`` and defines
    ``dict``, the keyword arguments for a ``FlexibleSweep``. The object can be
    read like that dictionary (``keys``, ``values``, ``items``, indexing,
    iteration).

    Parameters
    ----------
    sim_para : Dict[str, Any]
        Simulation parameters (dimensions and cutoffs).
    para : Dict[str, Any]
        Parameters of the Hamiltonian.
    """
    def __init__(
        self, 
        sim_para: Dict[str, Any],
        para: Dict[str, Any] = {},
    ):
        self.sim_para = sim_para
        self.para = para

    def qubit_init_w_conv_check(
        self, 
        qubit_init_ret_evecs: Callable,
        init_cut: int,
        convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
        update: bool = True,
    ) -> None:
        r"""
        Choose the basis cutoff of an ancilla by a convergence check of its
        eigenvectors.

        The convergence measure is the largest magnitude among the last three
        basis components of the eigenvectors. The cutoff is multiplied by 1.5
        while the measure is above the range and divided by 1.5 while it is below
        the range, except right after an increase.

        Parameters
        ----------
        qubit_init_ret_evecs : Callable
            Function of the cutoff that builds the ancilla and returns its bare
            eigenvectors.
        init_cut : int
            Initial cutoff.
        convergence_range : Tuple[float, float] | None
            Accepted range of the convergence measure. If None, the initial
            cutoff is kept.
        update : bool
            If False, the initial cutoff is kept.
        """
    
        last_operation = 0
        current_cut = init_cut
        
        while True:
            bare_evecs = qubit_init_ret_evecs(current_cut)

            # convergence checker: for all eigenvectors, check the smallness for 
            # the last three numbers
            conv = np.max(np.abs(bare_evecs[-3:, :]))  
            # conv = np.max(np.abs(bare_evecs[-1][-3:]))

            # to avoid the back-and-forth oscillation, we need to check the
            # operational history. last_operation = 1 means the previous cut was large and 
            # was reduced. And last_operation = -1 means the last cut is smaller and was 
            # increased. 
            if convergence_range is None or not update:
                break
            elif conv > convergence_range[1]:
                last_operation = -1
                current_cut = int(current_cut * 1.5)
            elif conv < convergence_range[0]:
                if last_operation == -1:
                    # When last_operation = -1 (meaning the conv was too BIG and the cut 
                    # number was reduced to a value that is too small) and the current conv is 
                    # too SMALL, we should not increase the cut number again, it might leads to 
                    # back-and-forth oscillation.
                    break
                last_operation = 1
                current_cut = int(current_cut / 1.5)
            else:
                break

    def keys(self):
        r"""
        Keys of ``dict``.
        """
        return self.dict.keys()

    def values(self):
        r"""
        Values of ``dict``.
        """
        return self.dict.values()
    
    def items(self):
        r"""
        Items of ``dict``.
        """
        return self.dict.items()
    
    def __getitem__(self, key):
        return self.dict[key]
    
    def __iter__(self):
        return self.dict.__iter__()

    @abstractproperty
    def dict(self) -> Dict[str, Any]:
        r"""
        Keyword arguments for a ``FlexibleSweep``.
        """
        ...


# ##############################################################################
# Systems built for FlexibleSweep
# Transmon -- Resonator 

class ResonatorTransmon(JointSystems):
    r"""
    Storage resonator coupled to a transmon ancilla.

    The resonator (``res``, an ``scq.Oscillator``) and the transmon (``qubit``)
    are coupled through the product of their charge operators
    (``n_operator``) with the strength ``g_GHz``. Parameters that ``para``
    does not hold take placeholder values, which a ``FlexibleSweep``
    overwrites.

    Parameters
    ----------
    sim_para : Dict[str, Any]
        Simulation parameters: ``"res_dim"`` and ``"qubit_dim"``, the
        truncated dimensions of the resonator and the transmon, and
        ``"qubit_ncut"``, the charge-basis cutoff of the transmon. With
        ``update_ncut``, ``"qubit_ncut"`` is replaced by the converged cutoff.
    para : Dict[str, Any]
        Parameters of the Hamiltonian: ``"E_osc_GHz"``, the resonator
        frequency; ``"EJ_GHz"`` and ``"EC_GHz"``, the Josephson and charging
        energies of the transmon; ``"ng"``, its offset charge; ``"g_GHz"``,
        the coupling strength.
    convergence_range : Tuple[float, float] | None
        Accepted range of the convergence measure of
        ``qubit_init_w_conv_check``.
    update_ncut : bool
        Whether the convergence check may change the cutoff.
    """
    def _qubit_init(self, cut):
        r"""
        Build the transmon with the charge-basis cutoff ``cut`` and return its bare
        eigenvectors.
        """
        self.qubit = scq.Transmon(
            EJ = self.para.get("EJ_GHz", 5),
            EC = self.para.get("EC_GHz", 0.2),
            ng = self.para.get("ng", 0.25),
            ncut = cut,
            truncated_dim = int(self.sim_para["qubit_dim"]),
            id_str = "qubit",
        )

        _, bare_evecs = self.qubit.eigensys(int(self.sim_para["qubit_dim"]))

        return bare_evecs
    
    def __init__(
        self,
        sim_para: Dict[str, Any],
        para: Dict[str, Any] = {},
        convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
        update_ncut: bool = True,
    ):
        r"""
        Build the subsystems and the ``HilbertSpace``; see the class docstring.
        """
        super().__init__(sim_para, para)

        # HilbertSpace
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        self.qubit_init_w_conv_check(
            self._qubit_init, 
            sim_para["qubit_ncut"], 
            convergence_range, 
            update_ncut
        )
        if update_ncut:
            sim_para["qubit_ncut"] = self.qubit.ncut

        self.hilbertspace = HilbertSpace([self.res, self.qubit])

        self.hilbertspace.add_interaction(
            g = para.get("g_GHz", 0.01),
            op1 = self.res.n_operator,
            op2 = self.qubit.n_operator,
            add_hc = False,
            id_str = "res-qubit"
        )

        # subsys_update_info
        self.subsys_update_info = {
            "E_osc_GHz": [self.res],
            "EJ_GHz": [self.qubit],
            "EC_GHz": [self.qubit],
            "ng": [self.qubit],
            "g_GHz": [],
        }

    def update_hilbertspace(
        self,
        hilbertspace: HilbertSpace, 
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, ng: float, g_GHz: float,
    ) -> None:
        r"""
        Set the parameters of a ``HilbertSpace`` of this system.

        Parameters
        ----------
        hilbertspace : HilbertSpace
            The Hilbert space, with the subsystems ``"res"`` and ``"qubit"``.
        E_osc_GHz : float
            Resonator frequency (GHz).
        EJ_GHz, EC_GHz : float
            Josephson and charging energies of the transmon (GHz).
        ng : float
            Offset charge of the transmon.
        g_GHz : float
            Coupling strength (GHz).
        """
        res: scq.Oscillator = hilbertspace.subsys_by_id_str("res")
        qubit: scq.Transmon = hilbertspace.subsys_by_id_str("qubit")
        interaction = hilbertspace.interaction_list[0]

        res.E_osc = E_osc_GHz
        qubit.EJ = EJ_GHz
        qubit.EC = EC_GHz
        qubit.ng = ng
        interaction.g_strength = g_GHz

    def update_hilbertspace_by_keyword(
        self,
        ps: ParameterSweep,
        **kwargs
    ) -> None:
        r"""
        Set the parameters of the ``HilbertSpace`` of a sweep from keywords.

        Parameters
        ----------
        ps : ParameterSweep
            The sweep whose ``hilbertspace`` is updated.
        **kwargs
            Values of ``E_osc_GHz``, ``EJ_GHz``, ``EC_GHz``, ``ng`` and ``g_GHz``. A missing parameter keeps the value of this object.
        """
        kwargs = copy.deepcopy(kwargs)
        self.update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz = kwargs.pop("E_osc_GHz", self.res.E_osc), 
            EJ_GHz = kwargs.pop("EJ_GHz", self.qubit.EJ),
            EC_GHz = kwargs.pop("EC_GHz", self.qubit.EC),
            ng = kwargs.pop("ng", self.qubit.ng),
            g_GHz = kwargs.pop("g_GHz", ps.hilbertspace.interaction_list[0].g_strength),
        )

    @property
    def dict(self) -> Dict[str, Any]:
        r"""
        Keyword arguments for a ``FlexibleSweep``.

        Returns
        -------
        Dict[str, Any]
            ``"sim_para"``, the simulation parameters; ``"hilbertspace"``, the
            ``HilbertSpace``; ``"qubit"`` and ``"res"``, the transmon and the resonator;
            ``"update_hilbertspace"`` and ``"update_hilbertspace_by_keyword"``,
            the two methods; ``"subsys_update_info"``, the subsystems that each
            parameter changes.
        """
        return {
            "sim_para": self.sim_para,
            "hilbertspace": self.hilbertspace,
            "qubit": self.qubit,
            "res": self.res,
            "update_hilbertspace": self.update_hilbertspace,
            "update_hilbertspace_by_keyword": self.update_hilbertspace_by_keyword,
            "subsys_update_info": self.subsys_update_info,
        }

class ResonatorFluxonium(JointSystems):
    r"""
    Storage resonator coupled to a fluxonium ancilla.

    The resonator (``res``, an ``scq.Oscillator``) and the fluxonium
    (``qubit``) are coupled through the product of their charge operators
    (``n_operator``) with the strength ``g_GHz``. Parameters that ``para``
    does not hold take placeholder values, which a ``FlexibleSweep``
    overwrites.

    Parameters
    ----------
    sim_para : Dict[str, Any]
        Simulation parameters: ``"res_dim"`` and ``"qubit_dim"``, the
        truncated dimensions of the resonator and the fluxonium, and
        ``"qubit_cutoff"``, the basis cutoff of the fluxonium. With
        ``update_cutoff``, ``"qubit_cutoff"`` is replaced by the converged
        cutoff.
    para : Dict[str, Any]
        Parameters of the Hamiltonian: ``"E_osc_GHz"``, the resonator
        frequency; ``"EJ_GHz"``, ``"EC_GHz"`` and ``"EL_GHz"``, the
        Josephson, charging and inductive energies of the fluxonium;
        ``"flux"``, its flux bias; ``"g_GHz"``, the coupling strength.
    convergence_range : Tuple[float, float] | None
        Accepted range of the convergence measure of
        ``qubit_init_w_conv_check``.
    update_cutoff : bool
        Whether the convergence check may change the cutoff.
    """
    def _qubit_init(self, cut):
        r"""
        Build the fluxonium with the basis cutoff ``cut`` and return its bare
        eigenvectors.
        """
        self.qubit = scq.Fluxonium(
            EJ = self.para.get("EJ_GHz", 5),
            EC = self.para.get("EC_GHz", 0.2),
            EL = self.para.get("EL_GHz", 0.5),
            flux = self.para.get("flux", 0.5),
            cutoff = cut,
            truncated_dim = int(self.sim_para["qubit_dim"]),
            id_str = "qubit",
        )

        _, bare_evecs = self.qubit.eigensys(int(self.sim_para["qubit_dim"]))

        return bare_evecs

    def __init__(
        self,
        sim_para: Dict[str, Any],
        para: Dict[str, Any] = {},
        convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
        update_cutoff: bool = True,
    ):
        r"""
        Build the subsystems and the ``HilbertSpace``; see the class docstring.
        """
        super().__init__(sim_para, para)
        
        # HilbertSpace
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        self.qubit_init_w_conv_check(
            self._qubit_init, 
            sim_para["qubit_cutoff"], 
            convergence_range, 
            update_cutoff
        )
        if update_cutoff:
            sim_para["qubit_cutoff"] = self.qubit.cutoff

        self.hilbertspace = HilbertSpace([self.res, self.qubit])

        self.hilbertspace.add_interaction(
            g = para.get("g_GHz", 0.01),
            op1 = self.res.n_operator,
            op2 = self.qubit.n_operator,
            add_hc = False,
            id_str = "res-qubit"
        )

        # subsys_update_info
        self.subsys_update_info = {
            "E_osc_GHz": [self.res],
            "EJ_GHz": [self.qubit],
            "EC_GHz": [self.qubit],
            "EL_GHz": [self.qubit],
            "flux": [self.qubit],
            "g_GHz": [],
        }

    def update_hilbertspace(
        self,
        hilbertspace: HilbertSpace,
        E_osc_GHz: float, EJ_GHz: float, EC_GHz: float, EL_GHz: float, flux: float,
        g_GHz: float,
    ) -> None:
        r"""
        Set the parameters of a ``HilbertSpace`` of this system.

        Parameters
        ----------
        hilbertspace : HilbertSpace
            The Hilbert space, with the subsystems ``"res"`` and ``"qubit"``.
        E_osc_GHz : float
            Resonator frequency (GHz).
        EJ_GHz, EC_GHz, EL_GHz : float
            Josephson, charging and inductive energies of the fluxonium (GHz).
        flux : float
            Flux bias of the fluxonium.
        g_GHz : float
            Coupling strength (GHz).
        """
        res: scq.Oscillator = hilbertspace.subsys_by_id_str("res")
        qubit: scq.Fluxonium = hilbertspace.subsys_by_id_str("qubit")
        interaction = hilbertspace.interaction_list[0]
        res.E_osc = E_osc_GHz
        qubit.EJ = EJ_GHz
        qubit.EC = EC_GHz
        qubit.EL = EL_GHz
        qubit.flux = flux
        interaction.g_strength = g_GHz

    def update_hilbertspace_by_keyword(
        self,
        ps: ParameterSweep,
        **kwargs,
    ) -> None:
        r"""
        Set the parameters of the ``HilbertSpace`` of a sweep from keywords.

        Parameters
        ----------
        ps : ParameterSweep
            The sweep whose ``hilbertspace`` is updated.
        **kwargs
            Values of ``E_osc_GHz``, ``EJ_GHz``, ``EC_GHz``, ``EL_GHz``, ``flux`` and ``g_GHz``. A missing parameter keeps the value of this object.
        """
        kwargs = copy.deepcopy(kwargs)
        self.update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz = kwargs.pop("E_osc_GHz", self.res.E_osc), 
            EJ_GHz = kwargs.pop("EJ_GHz", self.qubit.EJ),
            EC_GHz = kwargs.pop("EC_GHz", self.qubit.EC),
            EL_GHz = kwargs.pop("EL_GHz", self.qubit.EL),
            flux = kwargs.pop("flux", self.qubit.flux),
            g_GHz = kwargs.pop("g_GHz", self.hilbertspace.interaction_list[0].g_strength),
        )

    @property
    def dict(self) -> Dict[str, Any]:
        r"""
        Keyword arguments for a ``FlexibleSweep``.

        Returns
        -------
        Dict[str, Any]
            ``"sim_para"``, the simulation parameters; ``"hilbertspace"``, the
            ``HilbertSpace``; ``"qubit"`` and ``"res"``, the fluxonium and the resonator;
            ``"update_hilbertspace"`` and ``"update_hilbertspace_by_keyword"``,
            the two methods; ``"subsys_update_info"``, the subsystems that each
            parameter changes.
        """
        return {
            "sim_para": self.sim_para,
            "hilbertspace": self.hilbertspace,
            "qubit": self.qubit,
            "res": self.res,
            "update_hilbertspace": self.update_hilbertspace,
            "update_hilbertspace_by_keyword": self.update_hilbertspace_by_keyword,
            "subsys_update_info": self.subsys_update_info,
        }

class FluxoniumResonatorFluxonium(JointSystems):
    r"""
    Resonator coupled to two fluxoniums.

    Each fluxonium (``qubit1``, ``qubit2``) is coupled to the resonator
    (``res``, an ``scq.Oscillator``) through the product of their charge
    operators (``n_operator``), with the strengths ``g1_GHz`` and ``g2_GHz``.
    The subsystems are ordered as (``qubit1``, ``res``, ``qubit2``).
    Parameters that ``para`` does not hold take placeholder values, which a
    ``FlexibleSweep`` overwrites.

    Parameters
    ----------
    sim_para : Dict[str, Any]
        Simulation parameters: ``"res_dim"``, ``"qubit_dim1"`` and
        ``"qubit_dim2"``, the truncated dimensions; ``"qubit_cutoff1"`` and
        ``"qubit_cutoff2"``, the basis cutoffs of the fluxoniums. With
        ``update_cutoff``, the cutoffs are replaced by the converged ones.
    para : Dict[str, Any]
        Parameters of the Hamiltonian: ``"E_osc_GHz"``, the resonator
        frequency; ``"EJ1_GHz"``, ``"EC1_GHz"``, ``"EL1_GHz"`` and
        ``"flux1"``, the Josephson, charging and inductive energies and the
        flux bias of the first fluxonium; the same keys with ``2`` for the
        second one; ``"g1_GHz"`` and ``"g2_GHz"``, the coupling strengths.
    convergence_range : Tuple[float, float] | None
        Accepted range of the convergence measure of
        ``qubit_init_w_conv_check``.
    update_cutoff : bool
        Whether the convergence check may change the cutoffs.
    """
    def _qubit1_init(self, cut):
        r"""
        Build the first fluxonium with the basis cutoff ``cut`` and return its bare
        eigenvectors.
        """
        self.qubit1 = scq.Fluxonium(
            EJ = self.para.get("EJ1_GHz", 5),
            EC = self.para.get("EC1_GHz", 0.2),
            EL = self.para.get("EL1_GHz", 0.5),
            flux = self.para.get("flux1", 0.5),
            cutoff = cut,
            truncated_dim = int(self.sim_para["qubit_dim1"]),
            id_str = "qubit1",
        )

        _, bare_evecs = self.qubit1.eigensys(int(self.sim_para["qubit_dim1"]))

        return bare_evecs

    def _qubit2_init(self, cut):
        r"""
        Build the second fluxonium with the basis cutoff ``cut`` and return its
        bare eigenvectors.
        """
        self.qubit2 = scq.Fluxonium(
            EJ = self.para.get("EJ2_GHz", 5),
            EC = self.para.get("EC2_GHz", 0.2),
            EL = self.para.get("EL2_GHz", 0.5),
            flux = self.para.get("flux2", 0.5),
            cutoff = cut,
            truncated_dim = int(self.sim_para["qubit_dim2"]),
            id_str = "qubit2",
        )

        _, bare_evecs = self.qubit2.eigensys(int(self.sim_para["qubit_dim2"]))

        return bare_evecs
    
    def __init__(
        self,
        sim_para: Dict[str, Any],
        para: Dict[str, Any] = {},
        convergence_range: Tuple[float, float] | None = (1e-10, 1e-6), 
        update_cutoff: bool = True,
    ):
        r"""
        Build the subsystems and the ``HilbertSpace``; see the class docstring.
        """
        super().__init__(sim_para, para)

        # HilbertSpace
        self.res = scq.Oscillator(
            E_osc = para.get("E_osc_GHz", 5),
            truncated_dim = int(sim_para["res_dim"]),
            id_str = "res",
            l_osc = 1
        )

        self.qubit_init_w_conv_check(
            self._qubit1_init, 
            sim_para["qubit_cutoff1"], 
            convergence_range, 
            update_cutoff
        )
        if update_cutoff:
            sim_para["qubit_cutoff1"] = self.qubit1.cutoff

        self.qubit_init_w_conv_check(
            self._qubit2_init, 
            sim_para["qubit_cutoff2"], 
            convergence_range, 
            update_cutoff
        )
        if update_cutoff:
            sim_para["qubit_cutoff2"] = self.qubit2.cutoff

        self.hilbertspace = HilbertSpace([self.qubit1, self.res, self.qubit2])

        self.hilbertspace.add_interaction(
            g = para.get("g1_GHz", 0.01),
            op1 = self.qubit1.n_operator,
            op2 = self.res.n_operator,
            add_hc = False,
            id_str = "qubit1-res"
        )

        self.hilbertspace.add_interaction(
            g = para.get("g2_GHz", 0.01),
            op1 = self.res.n_operator,
            op2 = self.qubit2.n_operator,
            add_hc = False,
            id_str = "res-qubit2"
        )

        self.subsys_update_info = {
            "E_osc_GHz": [self.res],
            "EJ1_GHz": [self.qubit1],
            "EC1_GHz": [self.qubit1],
            "EL1_GHz": [self.qubit1],
            "flux1": [self.qubit1],
            "EJ2_GHz": [self.qubit2],
            "EC2_GHz": [self.qubit2],
            "EL2_GHz": [self.qubit2],
            "flux2": [self.qubit2],
            "g1_GHz": [],
            "g2_GHz": [],
        }

    def update_hilbertspace(
        self,
        hilbertspace: HilbertSpace,
        E_osc_GHz: float, 
        EJ1_GHz: float, EC1_GHz: float, EL1_GHz: float, flux1: float,
        EJ2_GHz: float, EC2_GHz: float, EL2_GHz: float, flux2: float,
        g1_GHz: float, g2_GHz: float,
    ) -> None:
        r"""
        Set the parameters of a ``HilbertSpace`` of this system.

        Parameters
        ----------
        hilbertspace : HilbertSpace
            The Hilbert space, with the subsystems ``"qubit1"``, ``"res"`` and
            ``"qubit2"``.
        E_osc_GHz : float
            Resonator frequency (GHz).
        EJ1_GHz, EC1_GHz, EL1_GHz : float
            Josephson, charging and inductive energies of the first fluxonium
            (GHz).
        flux1 : float
            Flux bias of the first fluxonium.
        EJ2_GHz, EC2_GHz, EL2_GHz, flux2 : float
            The same for the second fluxonium.
        g1_GHz, g2_GHz : float
            Coupling strengths of the two fluxoniums to the resonator (GHz).
        """
        res: scq.Oscillator = hilbertspace.subsys_by_id_str("res")
        qubit1: scq.Fluxonium = hilbertspace.subsys_by_id_str("qubit1")
        qubit2: scq.Fluxonium = hilbertspace.subsys_by_id_str("qubit2")
        interaction1 = hilbertspace.interaction_list[0]
        interaction2 = hilbertspace.interaction_list[1]

        res.E_osc = E_osc_GHz

        qubit1.EJ = EJ1_GHz
        qubit1.EC = EC1_GHz
        qubit1.EL = EL1_GHz
        qubit1.flux = flux1

        qubit2.EJ = EJ2_GHz
        qubit2.EC = EC2_GHz
        qubit2.EL = EL2_GHz
        qubit2.flux = flux2

        interaction1.g_strength = g1_GHz
        interaction2.g_strength = g2_GHz


    def update_hilbertspace_by_keyword(
        self,    
        ps: ParameterSweep,
        **kwargs
    ) -> None:
        r"""
        Set the parameters of the ``HilbertSpace`` of a sweep from keywords.

        Parameters
        ----------
        ps : ParameterSweep
            The sweep whose ``hilbertspace`` is updated.
        **kwargs
            Values of the parameters of ``update_hilbertspace``. A missing parameter keeps the value of this object.
        """
        kwargs = copy.deepcopy(kwargs)
        self.update_hilbertspace(
            ps.hilbertspace, 
            E_osc_GHz = kwargs.pop("E_osc_GHz", self.res.E_osc),
            EJ1_GHz = kwargs.pop("EJ1_GHz", self.qubit1.EJ),
            EC1_GHz = kwargs.pop("EC1_GHz", self.qubit1.EC),
            EL1_GHz = kwargs.pop("EL1_GHz", self.qubit1.EL),
            flux1 = kwargs.pop("flux1", self.qubit1.flux),
            EJ2_GHz = kwargs.pop("EJ2_GHz", self.qubit2.EJ),
            EC2_GHz = kwargs.pop("EC2_GHz", self.qubit2.EC),
            EL2_GHz = kwargs.pop("EL2_GHz", self.qubit2.EL),
            flux2 = kwargs.pop("flux2", self.qubit2.flux),
            g1_GHz = kwargs.pop("g1_GHz", self.hilbertspace.interaction_list[0].g_strength),
            g2_GHz = kwargs.pop("g2_GHz", self.hilbertspace.interaction_list[1].g_strength),
        )


    @property
    def dict(self) -> Dict[str, Any]:
        r"""
        Keyword arguments for a ``FlexibleSweep``.

        Returns
        -------
        Dict[str, Any]
            ``"sim_para"``, the simulation parameters; ``"hilbertspace"``, the
            ``HilbertSpace``; ``"qubit1"``, ``"res"`` and ``"qubit2"``, the subsystems;
            ``"update_hilbertspace"`` and ``"update_hilbertspace_by_keyword"``,
            the two methods; ``"subsys_update_info"``, the subsystems that each
            parameter changes.
        """
        return {
            "sim_para": self.sim_para,
            "hilbertspace": self.hilbertspace,
            "qubit1": self.qubit1,
            "res": self.res,
            "qubit2": self.qubit2,
            "update_hilbertspace": self.update_hilbertspace,
            "update_hilbertspace_by_keyword": self.update_hilbertspace_by_keyword,
            "subsys_update_info": self.subsys_update_info,
        }

