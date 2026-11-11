__all__ = [
    'EvolutionEdge',
    'PropagatorEdge',
    'MeasurementEdge',
    'CheckPointEdge',
]

import copy
import qutip as qt
import numpy as np
from warnings import warn
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, TYPE_CHECKING, Callable, Dict, Literal

from chencrafts.cqed.qt_helper import (
    superop_evolve,
    normalization_factor,
    dnorm,
)
from . import settings
from .utils import (
    effective_logical_process, 
    target_process_for_dnorm, 
    _subspace_types,
    process_block_dnorm,
)
from .node import TerminationError

if TYPE_CHECKING:
    from .node import (
        Node,
        MeasurementRecord,
    )


class EdgeBase(ABC):
    """
    Edge of an evolution graph: one step that takes the state of its
    initial node to the state of its final node.
    """
    name: str

    init_state: "Node"
    final_state: "Node"

    index: int
    
    def connect(self, init_state: "Node", final_state: "Node") -> None:
        """
        Attach the edge to the nodes that represent the states before and
        after the step.
        """
        self.init_state = init_state
        self.final_state = final_state
        # a tree node has one incoming edge; recording it lets the node
        # compose the dynamical maps of the steps that lead to it
        final_state.in_edge = self

    def assign_index(self, index: int) -> None:
        """Set the index of the edge in its graph."""
        self.index = index

    @abstractmethod
    def evolve(self):
        """
        Apply the step to the state of the initial node and store the
        result in the final node.
        """
        pass

    def to_nx(self) -> Tuple[int, int, Dict[str, Any]]:
        """
        The edge as a ``networkx`` edge: the indices of its two nodes and a
        dictionary of its name, type, and branching probability.
        """
        try:
            prob = self.branching_probability
        except AttributeError:
            prob = np.nan

        return (
            self.init_state.index,
            self.final_state.index,
            {
                "name": self.name,
                "type": type(self).__name__,
                "process": self,
                "branching_probability": prob,
            }
        )  

    @property
    def branching_probability(self) -> float:
        """
        Probability of the state of the final node divided by that of the
        state of the initial node.
        """
        return self.final_state.probability / self.init_state.probability     

    @property
    def terminated(self) -> bool:
        """
        Whether the final node of the edge is terminated.
        """
        return self.final_state.terminated

class EvolutionEdge(EdgeBase):
    r"""
    Edge that holds the dynamical map of one step (``real_map``) and the
    ideal operators of the step (``ideal_maps``).

    The dynamical map evolves the state of the initial node. Each ideal
    operator takes each error space of the initial node to one error space
    of the final node.
    """
    # Set by evolve(). The position is the error space after the step, and
    # the entry is the error space before the step from which an ideal
    # operator produced it.
    init_error_space_by_final: List[int]

    def __init__(
        self, 
        name: str,
        real_map: qt.Qobj | Callable[["MeasurementRecord"], qt.Qobj],
        ideal_maps: List[qt.Qobj] | Callable[["MeasurementRecord"], List[qt.Qobj]],
        to_ensemble: bool = False,
        error_accumulation: bool = False,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the edge.
        real_map : qt.Qobj | Callable[[MeasurementRecord], qt.Qobj]
            The dynamical map of the step, a superoperator on the physical
            space (qutip ``super`` representation), or a function that
            returns it for the measurement record of the initial node.
        ideal_maps : List[qt.Qobj] | Callable[[MeasurementRecord], List[qt.Qobj]]
            The ideal operators of the step, or a function that returns
            them for the measurement record of the initial node. An ideal
            operator maps a pair of ideal logical states before the step to
            the pair after it, which need not be normalized.
        to_ensemble : bool
            Must be False; True raises ``NotImplementedError``.
        error_accumulation : bool
            Whether the step appends a correctable error to the error
            record. It is True only for the idling of the resonator.
        """
        self.name = name
        self.real_map = real_map
        self.ideal_maps = ideal_maps
        self.error_accumulation = error_accumulation

        self.to_ensemble = to_ensemble
        if self.to_ensemble:
            raise NotImplementedError("to_ensemble = True is deprecated.")

    def physical_map(self) -> qt.Qobj:
        r"""
        Dynamical map of this step for the measurement record of the
        initial node.

        Returns
        -------
        qt.Qobj
            Superoperator on the physical space (qutip ``super``
            representation).
        """
        if isinstance(self.real_map, qt.Qobj):
            return self.real_map
        return self.real_map(self.init_state.meas_record)

    def evolve(self) -> None:
        """
        Apply the step to the state of the initial node and store the
        result in the final node.

        The dynamical map evolves the state and the encoded process. The
        ideal operators evolve the ideal logical states; a pair whose norm
        falls below the thresholds of ``settings``, or whose two states
        have become one, is dropped. If no pair is left, the final node is
        terminated. ``init_error_space_by_final`` records, for each error
        space after the step, the error space before the step that produced
        it.
        """
        try:
            self.init_state
            self.final_state
        except AttributeError:
            raise AttributeError("The initial state and the final state are not connected.")
        
        try:
            self.init_state.state
            self.init_state.prob_amp_01
            self.init_state._raw_ideal_logical_states
            self.init_state.encoded_process
        except AttributeError:
            raise AttributeError("The initial state are not evolved.")

        # evolve the state using the real map
        map_superop = self.physical_map()
        final_state = superop_evolve(
            map_superop, self.init_state.state
        )

        # only the process on encoded initial states is propagated, a
        # (d^2 x d^2) times (d^2 x 4k^2) product
        final_process = map_superop * self.init_state.encoded_process

        # evolve the ideal states using the ideal maps
        if isinstance(self.ideal_maps, list):
            map_op_list = self.ideal_maps
        else:
            map_op_list = self.ideal_maps(self.init_state.meas_record)

        new_ideal_logical_states = []
        # Same order as the final error spaces kept below.
        init_error_space_by_final: List[int] = []
        for map_op in map_op_list:
            for in_idx, (logical_0, logical_1) in enumerate(
                self.init_state.ideal_logical_states
            ):
                new_logical_0 = map_op * logical_0
                new_logical_1 = map_op * logical_1

                norm_0 = normalization_factor(new_logical_0)
                norm_1 = normalization_factor(new_logical_1)

                threshold_0 = np.sqrt(settings.IDEAL_STATE_THRESHOLD_0)
                if norm_0 < threshold_0 or norm_1 < threshold_0:
                    # when a syndrome measurement is done, it's likely that the 
                    # number of ideal states will be reduced and the state is not 
                    # normalized anymore. Only add the state to the list if it's 
                    # not zero norm.
                    continue

                threshold_1 = np.sqrt(settings.IDEAL_STATE_THRESHOLD_1)
                if norm_0 < threshold_1 or norm_1 < threshold_1:
                    # it's possible that the ideal evolution will give a state
                    # that has some small components. (For exmaple, the parity
                    # mapping with chi_prime can never be perfect, but we 
                    # still want to keep track of the chi_prime during evolution)
                    if settings.ISSUE_WARNING:
                        warn("Non-negligible small components are found in the ideal "
                             "states, for simplicity, they are ignored.\n")
                    continue

                if (
                    1 - np.abs((new_logical_0 / norm_0).overlap(new_logical_1 / norm_1))
                    < settings.IDEAL_STATE_OVERLAP_THRESHOLD
                ):
                    # the two codewords have become one state, e.g. after
                    # more photon losses than the truncated resonator holds;
                    # they no longer encode a qubit, and orthonormalizing
                    # them would give zero vectors
                    if settings.ISSUE_WARNING:
                        warn("A pair of ideal codewords has become one state; "
                             "it is dropped.\n")
                    continue

                new_ideal_logical_states.append(
                    [new_logical_0, new_logical_1]
                )
                init_error_space_by_final.append(in_idx)
        
        # no pair of ideal logical states is left, usually because
        # 1. the state is in its steady state: no photon is left to lose
        # 2. the measurement record is one that no correctable error can
        #    produce (a failure such as leakage)
        if len(new_ideal_logical_states) == 0:
            if settings.ISSUE_WARNING:
                warn("Can't find ideal logical states. Use the previous ideal logical states.\n")
            new_ideal_logical_states = copy.deepcopy(self.init_state.ideal_logical_states)
            # The fallback keeps one final error space per initial error space.
            init_error_space_by_final = list(range(len(new_ideal_logical_states)))
            self.final_state.terminated = True

        new_ideal_logical_state_array = np.empty(
            (len(new_ideal_logical_states), 2), dtype=object
        )
        new_ideal_logical_state_array[:] = new_ideal_logical_states
        self.init_error_space_by_final = init_error_space_by_final

        if not self.to_ensemble:
            self.final_state.accept(
                meas_record = copy.deepcopy(self.init_state.meas_record), 
                state = final_state, 
                prob_amp_01 = copy.deepcopy(self.init_state._prob_amp_01),
                raw_ideal_logical_states = new_ideal_logical_state_array,
                encoded_process = final_process,
                init_encoders = copy.deepcopy(self.init_state.init_encoders),
            )
        else:
            # deprecated now !!!!
            
            # self.final_state.join(
            #     meas_record = copy.copy(self.init_state.meas_record), 
            #     state = final_state, 
            #     prob_amp_01 = copy.copy(self.init_state._prob_amp_01),
            #     ideal_logical_states = new_ideal_logical_state_array,
            # )
            pass

    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def effective_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
    ) -> np.ndarray:
        r"""
        Step-wise effective logical maps of the edge.
        
        For error space :math:`a` before the step and error space :math:`b`
        after it, the map encodes the logical qubit into error space
        :math:`a`, applies the dynamical map of the step, and decodes from
        error space :math:`b`.

        Parameters
        ----------
        repr : {"super", "choi", "chi", "orth_chi"}
            Representation of each map: the qutip ``super``, ``choi``, or
            ``chi`` representation, or the process matrix of
            ``chencrafts.cqed.proc_repr.to_orth_chi``.

        Returns
        -------
        np.ndarray
            Object array of maps on the logical qubit; entry ``[b, a]``
            belongs to the transition from error space :math:`a` to error
            space :math:`b`.
        """
        init_encoders = self.init_state.ideal_encoders()
        final_encoders = self.final_state.ideal_encoders()
        
        if isinstance(self.real_map, qt.Qobj):
            map_superop = self.real_map
        else:
            map_superop = self.real_map(self.init_state.meas_record) 
            
        return effective_logical_process(
            process = map_superop,
            init_encoders = init_encoders,
            final_encoders = final_encoders,
            repr = repr,
        )
        
    def fidelity_by_process(
        self, type: Literal["avg", "etg"] = "etg",
    ) -> np.ndarray:
        r"""
        Process fidelity to the identity map of each step-wise effective
        logical map of the edge, the step-wise success probability

        .. math::
            g = \mathrm{tr}(S) / d^2 ,

        where :math:`S` is the superoperator matrix of the step-wise
        effective logical map and :math:`d=2` is the dimension of the
        logical qubit. :math:`g` is linear in the map and is not divided
        by the trace that the map retains. It is evaluated as
        ``qutip.process_fidelity`` of the map without a target.

        Parameters
        ----------
        type : {"avg", "etg"}
            ``"etg"`` (the default) returns :math:`g`. ``"avg"`` (average
            fidelity) raises ``NotImplementedError``, because the
            conversion from :math:`g` is not defined for a map that is
            not trace preserving.

        Returns
        -------
        np.ndarray
            Array of floats; entry ``[b, a]`` is :math:`g` from error
            space :math:`a` before the step to error space :math:`b`
            after it.
        """
        realized_process = self.effective_logical_process("super")
        
        fidelity = np.zeros(realized_process.shape)
        for idx, proc in np.ndenumerate(realized_process):
            # without a target, qutip returns tr(S)/d^2, which is linear in
            # the map; with the identity map as a target it takes matrix
            # square roots of the Choi matrices, which are accurate to
            # about 1e-8 only
            process_fidelity = qt.process_fidelity(proc)
            
            if type == "avg":
                raise NotImplementedError(
                    "Average fidelity is not implemented. As the conversion "
                    "from process fidelity isn't clear when the process isn't "
                    "CPTP."
                )
                # wrong when the process is not TP
                # fidelity[idx] = proc_fid_2_ave_fid(process_fidelity, 2)
            elif type == "etg":
                fidelity[idx] = process_fidelity
            else:
                raise ValueError("The type of fidelity should be either 'avg' or 'etg'.")

        return fidelity
        
    def process_dnorm(
        self,
        solver: str | None = None,
        **dnorm_kwargs,
    ) -> np.ndarray:
        r"""
        Diamond norm of the difference between each step-wise effective
        logical map of the edge and a scaled identity map.

        For a step-wise effective logical map
        :math:`\tilde{\mathcal E}`, the entry is

        .. math::
            \|\tilde{\mathcal E} - p\,\mathcal I\|_\diamond ,

        where :math:`\mathcal I` is the identity map and
        :math:`p=\mathrm{Tr}(J)/2`, with :math:`J` the Choi matrix of
        :math:`\tilde{\mathcal E}`, is the probability of the transition
        between the two error spaces for the maximally mixed logical
        state.

        Parameters
        ----------
        solver : str | None
            Solver of the semidefinite program of the diamond norm
            (``"SCS"``, ``"MOSEK"``, ``"CVXOPT"``), passed to
            :func:`chencrafts.cqed.qt_helper.dnorm`. If None (default)
            and no further argument is given, each entry is
            ``Qobj.dnorm`` of the map, with the default solver of qutip.
        **dnorm_kwargs
            Further arguments of :func:`chencrafts.cqed.qt_helper.dnorm`
            (``sparse``, ``zero_atol``, ...). If any is given, that
            function evaluates each entry, with its default solver if
            ``solver`` is None.

        Returns
        -------
        np.ndarray
            Array of shape (number of error spaces after the step, number
            of error spaces before the step).
        """
        processes = self.effective_logical_process("super")
        dnorms = np.zeros(processes.shape)

        for idx, process in np.ndenumerate(processes):
            # p times the identity map, with p the probability that the
            # map retains for the maximally mixed logical state
            target = target_process_for_dnorm(process)

            if solver is None and not dnorm_kwargs:
                dnorms[idx] = process.dnorm(target)
            else:
                # the options are copied because the solver is added to
                # them for every entry
                kwargs = dict(dnorm_kwargs)
                if solver is not None:
                    kwargs["solver"] = solver
                dnorms[idx] = dnorm(process, target=target, **kwargs)
        
        return dnorms
    
    def process_fidelity(self) -> np.ndarray:
        r"""
        Process fidelity to the identity map of each step-wise effective
        logical map of the edge, :math:`\chi_{II}/4`, with :math:`\chi` the
        process matrix of the map as ``qutip.to_chi`` normalizes it.

        Returns
        -------
        np.ndarray
            Entry ``[b, a]`` belongs to the transition from error space
            :math:`a` before the step to error space :math:`b` after it.
        """
        processes = self.effective_logical_process(repr="chi") / 4
        fidelities = np.zeros(processes.shape)
        
        for idx, process in np.ndenumerate(processes):
            fidelities[idx] = process[0, 0].real
        
        return fidelities
    
    def process_choi_trace(self) -> np.ndarray:
        r"""
        Trace of the Choi matrix of each step-wise effective logical map
        of the edge, which is 2 for a trace-preserving map on the logical
        qubit.

        Returns
        -------
        np.ndarray
            Entry ``[b, a]`` belongs to the transition from error space
            :math:`a` before the step to error space :math:`b` after it.
        """
        processes = self.effective_logical_process(repr = "choi")
        traces = np.zeros(processes.shape, dtype=float)
        for idx, process in np.ndenumerate(processes):
            traces[idx] = process.tr().real
            
        return traces
    
    def process_block_dnorm(
        self,
        init_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LC", "LD"], 
        final_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LC", "LD"],
        solver: str | None = None,
        **dnorm_kwargs,
    ):
        r"""
        Diamond norm of a projected dynamical map of the step that this
        edge represents.

        The projected dynamical map is
        :math:`\mathcal P_{\rm final}\,\mathcal E\,\mathcal P_{\rm init}`,
        with :math:`\mathcal E` the dynamical map of the step for the
        measurement record of the initial node. The projector map
        :math:`\mathcal P_{\rm init}` keeps one component of an operator
        before the step, defined by the error spaces of the initial
        node, and :math:`\mathcal P_{\rm final}` keeps one component
        after the step, defined by the error spaces of the final node.

        Parameters
        ----------
        init_subspace, final_subspace : tuple | Literal["LC", "LD"]
            The component kept before the step and the component kept
            after it, in the notation of
            ``utils.component_projector_map``: a tuple of two entries,
            each the integer index of an error space, ``"L"`` (joint
            error space), or ``"p"`` (leakage subspace); ``"LD"`` (the
            components within the individual error spaces); or ``"LC"``
            (the coherence between different error spaces).
        solver : str | None
            Solver of the semidefinite program of the diamond norm,
            passed to ``chencrafts.cqed.qt_helper.dnorm``. If None
            (default) and no further argument is given, the norm is
            ``Qobj.dnorm()`` of the projected dynamical map.
        **dnorm_kwargs
            Further arguments of ``chencrafts.cqed.qt_helper.dnorm``
            (``cp_atol``, ``cp_only``, ``sparse``, ...). If any is given,
            that function evaluates the norm.

        Returns
        -------
        float
            The diamond norm of the projected dynamical map, or one of
            the other values that ``chencrafts.cqed.qt_helper.dnorm``
            returns for the options given (for example ``nan`` with
            ``cp_only``).
        """
        init_encoders = self.init_state.ideal_encoders()
        final_encoders = self.final_state.ideal_encoders()
        
        if isinstance(self.real_map, qt.Qobj):
            map_superop = self.real_map
        else:
            map_superop = self.real_map(self.init_state.meas_record) 
            
        return process_block_dnorm(
            process = map_superop,
            init_subspace = init_subspace,
            final_subspace = final_subspace,
            init_encoders = init_encoders,
            final_encoders = final_encoders,
            solver = solver,
            **dnorm_kwargs,
        )
        

class PropagatorEdge(EvolutionEdge):
    """
    Edge of a step without a measurement.
    """
    pass


class MeasurementEdge(EvolutionEdge):
    """
    Edge of one outcome of a measurement; the outcome is appended to the
    measurement record of the final node.
    """
    def __init__(
        self, 
        name: str,
        outcome: float,
        real_map: qt.Qobj | Callable[["MeasurementRecord"], qt.Qobj],
        ideal_map: List[qt.Qobj] | Callable[["MeasurementRecord"], List[qt.Qobj]],
        to_ensemble: bool = False,
    ):
        """
        Parameters
        ----------
        name : str
            Name of the edge.
        outcome : float
            The measurement outcome of this edge.
        real_map : qt.Qobj | Callable[[MeasurementRecord], qt.Qobj]
            The dynamical map of the measurement with this outcome, a
            superoperator on the physical space, or a function that
            returns it for the measurement record of the initial node.
        ideal_map : List[qt.Qobj] | Callable[[MeasurementRecord], List[qt.Qobj]]
            The ideal operators of the measurement with this outcome.
        to_ensemble : bool
            Must be False.
        """
        super().__init__(name, real_map, ideal_map, to_ensemble)

        self.outcome = outcome

    def evolve(self) -> None:
        """
        Apply the step to the state of the initial node, store the result
        in the final node, and append the outcome to its measurement
        record.
        """
        super().evolve()
        init_record = copy.copy(self.init_state.meas_record)
        self.final_state.meas_record = init_record + [self.outcome]

    def __str__(self) -> str:
        return super().__str__() + f" ({self.outcome})"


Edge = PropagatorEdge | MeasurementEdge


class CheckPointEdge(EdgeBase):
    """
    Edge that projects the state onto the space of the ideal states of the
    initial node (``success`` True) or onto its complement (``success``
    False).
    """
    def __init__(
        self, 
        name: str,
        success: bool,
    ):
        self.name = name
        self.success = success

    def evolve(self) -> None:
        """
        Apply the projection to the state of the initial node and store
        the result in the final node. The probability of the complement
        approximates the failure probability to first order.
        """
        try:
            self.init_state
            self.final_state
        except AttributeError:
            raise AttributeError("The initial state and the final state are not connected.")
        
        try:
            self.init_state.state
            self.init_state.prob_amp_01
            self.init_state._raw_ideal_logical_states
        except AttributeError:
            raise AttributeError("The initial state are not evolved.")

        map_superop = self.physical_map()

        # only the process on encoded initial states is propagated
        process = map_superop * self.init_state.encoded_process
        final_state = superop_evolve(map_superop, self.init_state.state)

        self.final_state.accept(
            meas_record = copy.copy(self.init_state.meas_record),
            state = final_state,
            prob_amp_01 = copy.copy(self.init_state._prob_amp_01),
            raw_ideal_logical_states = copy.deepcopy(self.init_state._raw_ideal_logical_states),
            encoded_process = process,
            init_encoders = copy.deepcopy(self.init_state.init_encoders),
        )

    def physical_map(self) -> qt.Qobj:
        r"""
        Projection map of this check point: :math:`\hat P(\cdot)\hat P`,
        with :math:`\hat P` the projector onto the ideal states of the
        initial node, if ``success``; otherwise
        :math:`(1-\hat P)(\cdot)(1-\hat P)`.

        Returns
        -------
        qt.Qobj
            Superoperator on the physical space (qutip ``super``
            representation).
        """
        projector = self.init_state.ideal_projector
        if self.success:
            map_op = projector
        else:
            eye_op = qt.qeye_like(projector)
            map_op = eye_op - projector
        return qt.sprepost(map_op, map_op.dag())

    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()