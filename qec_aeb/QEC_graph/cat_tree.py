__all__ = [
    'FullCatTreeBuilder',
    'GATE_NU_HALF',
]

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable, overload, Literal
from warnings import warn
from copy import copy
import itertools
from tqdm.notebook import tqdm
from abc import ABC, abstractmethod

import numpy as np
import qutip as qt
import scqubits as scq

from chencrafts.cqed import FlexibleSweep, superop_evolve
from chencrafts.cqed.qt_helper import trans_by_kets

import qec_aeb.cat_ideal as cat_ideal
import qec_aeb.cat_recipe as cat_recipe
import qec_aeb.cat_real as cat_real
import chencrafts.settings as global_settings
import qec_aeb.QEC_graph.settings as graph_settings
from qec_aeb.batched_sweeps_recipe import pulse_shape_numbers

# Envelope shape number nu_theta of the simulated Gaussian pi/2 pulse
# (pulse_shape_numbers), independent of the duration. The ideal operator of
# the pi/2 pulse and the angle of the parity-mapping idling use this value
# in place of the square-pulse value 2/pi, so that each ideal operator
# contains the first-order tilt and phase that the dispersive shift produces
# during the simulated pulse.
GATE_NU_HALF = pulse_shape_numbers(np.pi / 2)[1]


from .node import StateNode, StateEnsemble, MeasurementRecord
from .edge import (
    PropagatorEdge, MeasurementEdge, Edge, CheckPointEdge)
from .graph import EvolutionGraph, EvolutionTree


class ComposedStepMaps:
    r"""
    Dynamical map and ideal operators of consecutive steps without a
    measurement, composed into those of one step.

    Each step gives its dynamical map and its ideal operators either
    directly or as a function of the measurement record before the step
    (as ``EvolutionEdge`` takes them). The record does not change within
    the composed steps, because none of them measures, so every step is
    evaluated at the record before the first one. Objects of this class
    hold no lambda, so a tree that holds them can be pickled when the
    builder can.
    """

    def __init__(
        self,
        real_maps: List[qt.Qobj | Callable[[MeasurementRecord], qt.Qobj]],
        ideal_maps: List[List[qt.Qobj] | Callable[[MeasurementRecord], List[qt.Qobj]]],
    ):
        r"""
        Parameters
        ----------
        real_maps : List[qt.Qobj | Callable[[MeasurementRecord], qt.Qobj]]
            Dynamical map of each step (superoperator), or the function
            that returns it for a measurement record, in the order of the
            protocol.
        ideal_maps : List[List[qt.Qobj] | Callable[[MeasurementRecord], List[qt.Qobj]]]
            Ideal operators of each step, or the function that returns them
            for a measurement record, in the same order.
        """
        self.step_real_maps = list(real_maps)
        self.step_ideal_maps = list(ideal_maps)
        # composed maps by measurement record: a tree of several cycles
        # evaluates the same record at many edges, and the maps are never
        # modified in place
        self._real_cache: Dict[Tuple, qt.Qobj] = {}
        self._ideal_cache: Dict[Tuple, List[qt.Qobj]] = {}

    @staticmethod
    def _at_record(entry, meas_record: MeasurementRecord):
        r"""
        A step's entry for a measurement record.

        Parameters
        ----------
        entry : qt.Qobj | List[qt.Qobj] | Callable
            Dynamical map or list of ideal operators of a step, or the
            function that returns it for a measurement record.
        meas_record : MeasurementRecord
            Measurement record before the composed steps.

        Returns
        -------
        qt.Qobj | List[qt.Qobj]
            The entry itself, or the function evaluated at ``meas_record``.
        """
        # a Qobj is callable, so it is tested before the call
        if isinstance(entry, (qt.Qobj, list)):
            return entry
        return entry(meas_record)

    def real_map(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        r"""
        Composed dynamical map, :math:`\mathcal E_k\cdots\mathcal E_2\mathcal E_1`
        for the steps :math:`1,\dots,k`.

        Parameters
        ----------
        meas_record : MeasurementRecord
            Measurement record before the composed steps.

        Returns
        -------
        qt.Qobj
            Superoperator on the physical space.
        """
        key = tuple(meas_record)
        if key not in self._real_cache:
            # a later step acts after an earlier one, so it multiplies from
            # the left
            composed = None
            for entry in self.step_real_maps:
                step_map = self._at_record(entry, meas_record)
                composed = step_map if composed is None else step_map * composed
            self._real_cache[key] = composed
        return self._real_cache[key]

    def ideal_maps(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        r"""
        Composed ideal operators, :math:`\hat U_k\cdots\hat U_2\hat U_1`
        for every choice of one ideal operator :math:`\hat U_n` of each step
        :math:`n`.

        Parameters
        ----------
        meas_record : MeasurementRecord
            Measurement record before the composed steps.

        Returns
        -------
        List[qt.Qobj]
            The products, ordered with the operator of the last step varying
            slowest, so that the error spaces after the merged step come in
            the order that the steps applied one after another would give
            (``EvolutionEdge.evolve`` puts the ideal operator in the outer
            loop).
        """
        key = tuple(meas_record)
        if key not in self._ideal_cache:
            step_lists = [
                self._at_record(entry, meas_record)
                for entry in self.step_ideal_maps
            ]
            products = []
            # last step outermost, matching the order of the error spaces
            for choice in itertools.product(*reversed(step_lists)):
                product = choice[0]
                for operator in choice[1:]:
                    product = product * operator
                products.append(product)
            self._ideal_cache[key] = products
        return self._ideal_cache[key]


class CatTreeBuilder(ABC):
    """
    Builder of the tree of the QEC cycles of a cat code in a resonator with
    an ancilla (identifier ``qubit``).

    Each method named in ``STEP_METHOD_NAMES`` adds the edges of one step
    to a tree. An edge holds the dynamical map of the step (attributes
    ``*_real``) and its ideal operators (attributes ``*_ideal``). The class
    attributes below are switches; ``SWITCH_NAMES`` lists them.
    """

    graph = EvolutionGraph()

    # if True, the ideal operator replaces the dynamical map of the step
    idling_is_ideal: bool = False
    gate_1_is_ideal: bool = True
    parity_mapping_is_ideal: bool = False
    gate_2_is_ideal: bool = True
    qubit_measurement_is_ideal: bool = False
    feedback_latency_is_ideal: bool = False
    qubit_reset_is_ideal: bool = True
    
    # if True, the ideal operators of the pulses and of the parity-mapping
    # idling contain the phases that the dispersive shift, the self-Kerr,
    # and chi' produce during the step
    use_improved_gate: bool = True
    use_improved_parity_mapping: bool = True

    # ancilla operator that the drive couples to (None: default of the
    # circuit type, see cat_real.qubit_gate); it must match the drive
    # operator assumed by the error budget
    drive_operator: str | None = None

    # ancilla levels kept while the ancilla pulses are integrated (None:
    # the qubit_dim of the tree); the extra levels are dropped afterwards, so
    # population driven into them counts as leakage
    gate_simulated_qubit_dim: int | None = None

    # ancilla frame changes (virtual Z gates) of the parity mapping: after
    # and before each pi/2 pulse by minus the phase of its Stark shift, and
    # at the end of the parity-mapping idling by minus the cat-averaged chi'
    # phase. They need the sweep keys gate_frame_phase and pm_frame_phase of
    # batched_sweep_cat_budget_ingredients. The ideal operators do not change.
    use_frame_update: bool = False

    # the chi' part of the frame changes alone: the frame change at the end of
    # the parity-mapping idling. Setting it to False keeps the frame changes
    # of the pulses, which remove the phase of the Stark shift, and drops
    # only that one. It has no effect when use_frame_update is False.
    use_chi_prime_frame_update: bool = True

    # decoherence of the ancilla during the ancilla pulses (the pi/2 pulses
    # and the reset). If False, each pulse is the closed-system unitary of
    # the dressed model. If True (default), each pulse is a dynamical map
    # that also contains the jumps of the idling that act on the ancilla
    # alone, which do not change the photon number. The loss, gain, and
    # dephasing of the resonator during a pulse are not included.
    gate_qubit_decoherence: bool = True

    # names of the switches above; a builder sets every switch that its
    # ``sim_para`` holds, and the others keep their class default
    SWITCH_NAMES: Tuple[str, ...] = (
        "idling_is_ideal",
        "gate_1_is_ideal",
        "parity_mapping_is_ideal",
        "gate_2_is_ideal",
        "qubit_measurement_is_ideal",
        "feedback_latency_is_ideal",
        "qubit_reset_is_ideal",
        "use_improved_gate",
        "use_improved_parity_mapping",
        "drive_operator",
        "gate_simulated_qubit_dim",
        "use_frame_update",
        "use_chi_prime_frame_update",
        "gate_qubit_decoherence",
    )

    # step table #######################################################
    # two-letter prefix of the name of an edge, by the name of the method
    # that adds the edge. An edge is named "<prefix> (<cycle>.<step>)", with
    # the index of the QEC cycle and the running index of the step.
    EDGE_PREFIXES: Dict[str, str] = {
        "idle": "ID",
        "qubit_gate_1": "G1",
        "parity_mapping": "PM",
        "qubit_gate_2": "G2",
        "qubit_measurement": "MS",
        "feedback_latency": "FL",
        "qubit_reset": "RS",
        "check_point": "CP",
    }
    # names of the methods that add the edges of one QEC cycle, in the order
    # of the protocol; each builder class defines its own tuple
    STEP_METHOD_NAMES: Tuple[str, ...] = ()

    def _edge_name(
        self,
        method_name: str,
        step: int | str,
    ) -> str:
        r"""
        Name ``"<prefix> (<step>)"`` of the edge that the method
        ``method_name`` (a key of ``EDGE_PREFIXES``) adds for the step
        labelled ``step``, for example ``"ID (0.0)"``.
        """
        return f"{self._step_prefix(method_name)} ({step})"

    # merged steps #####################################################
    # A run of consecutive steps without a measurement can be treated as one
    # step (merge_steps). The merged step is named by the names of its
    # methods joined by this separator, e.g.
    # "qubit_gate_1+parity_mapping+qubit_gate_2", and its edges by their
    # prefixes joined the same way, e.g. "G1+PM+G2 (0.1)".
    MERGED_STEP_SEPARATOR: str = "+"

    def _step_prefix(
        self,
        method_name: str,
    ) -> str:
        r"""
        Prefix of the edge names of a step.

        Parameters
        ----------
        method_name : str
            A key of ``EDGE_PREFIXES``, or the name of a merged step
            (``merge_steps``): such keys joined by ``MERGED_STEP_SEPARATOR``.

        Returns
        -------
        str
            The prefix of the step, or, for a merged step, the prefixes of
            its steps joined by ``MERGED_STEP_SEPARATOR``.
        """
        return self.MERGED_STEP_SEPARATOR.join(
            self.EDGE_PREFIXES[name]
            for name in method_name.split(self.MERGED_STEP_SEPARATOR)
        )

    def _step_method(
        self,
        method_name: str,
    ) -> Callable[[int | str, EvolutionTree, StateEnsemble], StateEnsemble]:
        r"""
        Method that adds the edges of one step to a tree.

        Parameters
        ----------
        method_name : str
            An entry of ``STEP_METHOD_NAMES``: the name of a method of the
            builder, or the name of a merged step (``merge_steps``).

        Returns
        -------
        Callable
            Called as ``method(step, graph, init_nodes)``, like the methods
            of ``STEP_METHOD_NAMES``; it returns the nodes after the step.
        """
        # a single step is a method of the builder
        if self.MERGED_STEP_SEPARATOR not in method_name:
            return getattr(self, method_name)
        # a merged step adds one edge per node for all of its steps
        method_names = tuple(method_name.split(self.MERGED_STEP_SEPARATOR))
        return lambda step, graph, init_nodes: self.merged_step(
            method_names, step, graph, init_nodes,
        )

    def merge_steps(
        self,
        method_names: List[str] | Tuple[str, ...],
    ) -> str:
        r"""
        Treat consecutive steps of the QEC cycle as one step in the trees
        that this builder generates afterwards.

        The merged step has, for every node, a single edge. Its dynamical
        map is the composition of the dynamical maps of the steps, and its
        ideal operators are the compositions of their ideal operators, one
        for each choice of an ideal operator of every step. The error
        spaces after the merged step are therefore those after its last
        step, orthonormalized once, after the merged step
        (``StateNode.ORTHOGONALIZE_METHOD``), as after any step. No step
        may contain a measurement, because a merged step does not branch.

        The builder's ``STEP_METHOD_NAMES``, ``STEP_EDGE_PREFIXES``, and
        ``steps_per_cycle`` are replaced by instance attributes in which the
        steps appear as one; the class attributes and the dynamical maps
        are not changed.

        Parameters
        ----------
        method_names : List[str] | Tuple[str, ...]
            At least two consecutive entries of ``STEP_METHOD_NAMES``, in
            the order of the protocol, e.g. ``["qubit_gate_1",
            "parity_mapping", "qubit_gate_2"]``.

        Returns
        -------
        str
            The name of the merged step in ``STEP_METHOD_NAMES``: the names
            joined by ``MERGED_STEP_SEPARATOR``.

        Raises
        ------
        ValueError
            If fewer than two names are given, or they are not consecutive
            entries of ``STEP_METHOD_NAMES``.
        """
        method_names = tuple(method_names)
        sequence = tuple(self.STEP_METHOD_NAMES)
        if len(method_names) < 2:
            raise ValueError("A merged step needs at least two steps.")
        # the run must appear in the cycle as it is given, in order
        starts = [
            start for start in range(len(sequence) - len(method_names) + 1)
            if sequence[start:start + len(method_names)] == method_names
        ]
        if not starts:
            raise ValueError(
                f"{method_names} are not consecutive steps of {sequence}."
            )
        start = starts[0]
        merged_name = self.MERGED_STEP_SEPARATOR.join(method_names)
        merged_sequence = (
            sequence[:start] + (merged_name,)
            + sequence[start + len(method_names):]
        )
        # instance attributes shadow the step table of the class, so other
        # builders of the same class keep their steps
        self.STEP_METHOD_NAMES = merged_sequence
        self.STEP_EDGE_PREFIXES = tuple(
            self._step_prefix(name) for name in merged_sequence
        )
        self.steps_per_cycle = len(merged_sequence)
        return merged_name

    def _step_edge_template(
        self,
        method_name: str,
    ) -> "PropagatorEdge":
        r"""
        The edge that a step without a measurement adds to a node.

        The step's method is applied to a single node of a scratch tree, so
        the edge holds the same dynamical map and ideal operators (or the
        same functions of the measurement record) as in a real tree.

        Parameters
        ----------
        method_name : str
            A method of the builder that adds the edges of one step.

        Returns
        -------
        PropagatorEdge
            The edge, connected to the scratch nodes only.

        Raises
        ------
        ValueError
            If the step adds more than one edge to a node or an edge that
            is not a ``PropagatorEdge`` (a measurement or a check point).
        """
        scratch_graph = EvolutionTree()
        placeholder = StateNode()
        scratch_graph.add_node(placeholder)
        getattr(self, method_name)(
            "template", scratch_graph, StateEnsemble([placeholder]),
        )
        edges = placeholder.out_edges
        # a measurement branches and a check point projects; neither can be
        # composed into a single dynamical map
        if len(edges) != 1 or type(edges[0]) is not PropagatorEdge:
            raise ValueError(
                f"The step {method_name} is not a single step without a "
                "measurement and cannot be merged."
            )
        return edges[0]

    def merged_step(
        self,
        method_names: Tuple[str, ...],
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        r"""
        Add the edge of a merged step (``merge_steps``) to every node that
        is not terminated.

        Parameters
        ----------
        method_names : Tuple[str, ...]
            Methods of the builder that add the edges of the steps, in the
            order of the protocol; none of them may contain a measurement.
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step.
        """
        templates = [self._step_edge_template(name) for name in method_names]
        composed = ComposedStepMaps(
            [edge.real_map for edge in templates],
            [edge.ideal_maps for edge in templates],
        )
        edge = PropagatorEdge(
            self._edge_name(self.MERGED_STEP_SEPARATOR.join(method_names), step),
            composed.real_map,
            composed.ideal_maps,
            # the merged step appends a correctable error to the error
            # record if one of its steps does
            error_accumulation=any(edge.error_accumulation for edge in templates),
        )
        return self._add_leaves(graph, init_nodes.active_nodes(), edge)

    def _step_method_sequence(
        self,
        with_check_point: bool = False,
    ) -> List[str]:
        r"""
        Names of the methods that add the edges of one QEC cycle, in the
        order of the protocol: ``STEP_METHOD_NAMES``, with ``"check_point"``
        after each entry if ``with_check_point``.
        """
        edge_method_names = list(self.STEP_METHOD_NAMES)
        if with_check_point:
            edge_w_check_point = []
            for method_name in edge_method_names:
                edge_w_check_point.append(method_name)
                edge_w_check_point.append("check_point")
            edge_method_names = edge_w_check_point

        return edge_method_names

    def _read_switches(
        self,
        sim_para: Dict[str, Any],
    ) -> None:
        r"""
        Set, as instance attributes, the switches of ``SWITCH_NAMES`` that
        ``sim_para`` holds; the other switches keep their class default.
        """
        for name in self.SWITCH_NAMES:
            if name in sim_para:
                setattr(self, name, sim_para[name])

    # utils ############################################################
    @staticmethod
    def _current_parity(meas_record: MeasurementRecord):
        """
        Photon-number parity (0 even, 1 odd) for a measurement record. Gate
        2 depends on the record, so that each outcome 1 marks a parity flip.
        """
        return sum(meas_record) % 2

    @staticmethod
    @overload
    def _kraus_to_super(
        qobj_list: List[qt.Qobj]
    ) -> qt.Qobj:
        ...

    @staticmethod
    @overload
    def _kraus_to_super(
        qobj_list: Callable[[MeasurementRecord], List[qt.Qobj]]
    ) -> Callable[[MeasurementRecord], qt.Qobj]:
        ...
        
    @staticmethod
    def _kraus_to_super(
        qobj_list: List[qt.Qobj] | Callable[[MeasurementRecord], List[qt.Qobj]]
    ) -> qt.Qobj | Callable[[MeasurementRecord], qt.Qobj]:
        """
        Superoperator of a list of Kraus operators, or, for a function of
        the measurement record that returns such a list, the function that
        returns the superoperator.
        """
        if isinstance(qobj_list, qt.Qobj):
            # a Qobj is callable, so it is rejected before the callable test
            raise TypeError("The input should be a list of Qobj or a function.")
        
        if isinstance(qobj_list, list):
            return sum([qt.to_super(qobj) for qobj in qobj_list])
        elif callable(qobj_list):
            return lambda meas_record: sum([qt.to_super(qobj) for qobj in qobj_list(meas_record)])
        else:
            raise TypeError("The input should be a list of Qobj or a function.")
        
    @staticmethod
    def _add_leaves(
        graph: EvolutionTree,
        nodes: StateEnsemble,
        edge: Edge,
    ) -> StateEnsemble:
        """
        Add a copy of ``edge`` and a new final node to every node of
        ``nodes`` that is not terminated; the new nodes are returned.
        """
        final_nodes = StateEnsemble()
        template = edge

        for node in nodes.active_nodes():
            # A shallow copy of the unconnected template: every copy shares
            # the maps held by the builder (they are never modified in
            # place), and gets its own connection to its nodes. A deep copy
            # would duplicate a d^2 x d^2 map per edge, and a deep copy of an
            # already connected edge would also copy the whole tree through
            # the nodes' in_edge links.
            edge = copy(template)
            f_node = StateNode()
            final_nodes.append(f_node)
            graph.add_node(f_node)
            graph.add_edge_connect(
                edge,
                node,
                f_node,
            )

        return final_nodes
        
    # idling ###########################################################
    _idling_real: qt.Qobj
    _idling_ideal: List[qt.Qobj]

    def idle(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
        correctable_single_photon_loss: bool = True,
    ) -> StateEnsemble:
        """
        Add the edge of the idling step to every node that is not
        terminated.

        If ``idling_is_ideal`` is True, the idling has zero duration, so
        that its dynamical map is the identity.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.
        correctable_single_photon_loss : bool
            If True (default), the loss of one photon is a correctable
            error: the edge has two ideal operators, the evolution without
            a jump and the evolution with one photon loss. If False, as for
            an evolution without QEC, only the first one.

        Returns
        -------
        StateEnsemble
            The nodes after the step.
        """
        if correctable_single_photon_loss:
            accepted_states_num = 2
        else:
            accepted_states_num = 1

        edge_idling = PropagatorEdge(
            self._edge_name("idle", step),
            self._idling_real,
            self._idling_ideal[:accepted_states_num],
            # only the idling of the resonator appends a correctable error
            # to the error record
            error_accumulation=True,
        )

        return self._add_leaves(graph, init_nodes.active_nodes(), edge_idling)
    
    # qubit gate #######################################################
    _qubit_gate_p_ideal: List[qt.Qobj]
    _qubit_gate_m_ideal: List[qt.Qobj]
    _qubit_gate_p_real: qt.Qobj
    _qubit_gate_m_real: qt.Qobj

    @abstractmethod
    def _qubit_gate_2_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        pass
        
    @abstractmethod
    def _qubit_gate_2_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        pass

    def qubit_gate_1(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        r"""
        Add the edge of gate 1, the :math:`+\pi/2` ancilla pulse before the
        parity-mapping idling, to every node that is not terminated.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step.
        """
        edge_qubit_gate = PropagatorEdge(
            self._edge_name("qubit_gate_1", step),
            (
                self._qubit_gate_p_real 
                if not self.gate_1_is_ideal 
                else self._kraus_to_super(self._qubit_gate_p_ideal)
            ),
            self._qubit_gate_p_ideal,
        )
        return self._add_leaves(graph, init_nodes.active_nodes(), edge_qubit_gate)
    
    def qubit_gate_2(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        r"""
        Add the edge of gate 2, the :math:`\pm\pi/2` ancilla pulse after
        the parity-mapping idling, to every node that is not terminated.
        The sign of the rotation depends on the measurement record.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step.
        """
        edge_qubit_gate = PropagatorEdge(
            self._edge_name("qubit_gate_2", step),
            (
                self._qubit_gate_2_map_real 
                if not self.gate_2_is_ideal 
                else self._kraus_to_super(self._qubit_gate_2_map_ideal)
            ),
            self._qubit_gate_2_map_ideal,
        )

        return self._add_leaves(graph, init_nodes.active_nodes(), edge_qubit_gate)
    
    # parity mapping ###################################################
    _parity_mapping_ideal: List[qt.Qobj]
    _parity_mapping_real: qt.Qobj

    def parity_mapping(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        """
        Add the edge of the parity-mapping idling to every node that is
        not terminated.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step.
        """
        edge_parity_mapping = PropagatorEdge(
            name = self._edge_name("parity_mapping", step),
            real_map = (
                self._parity_mapping_real
                if not self.parity_mapping_is_ideal 
                else self._kraus_to_super(self._parity_mapping_ideal)
            ),
            ideal_maps = self._parity_mapping_ideal
        )

        return self._add_leaves(graph, init_nodes.active_nodes(), edge_parity_mapping)
    
    # qubit measurement ################################################
    # the dynamical maps and the ideal operators are labelled by the same
    # outcomes, because the later steps act on the reported outcome
    _accepted_measurement_outcome_pool: List[int]
    _measurement_outcome_pool: List[int]

    _qubit_projs_ideal: List[qt.Qobj]
    _qubit_projs_real: List[qt.Qobj]

    def qubit_measurement(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        """
        Add one edge of the ancilla measurement for each outcome to every
        node that is not terminated. The final node of an outcome that is
        not accepted is terminated.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step, terminated ones included.
        """
        all_final_nodes = StateEnsemble()

        for idx in range(len(self._qubit_projs_ideal)):
            # an outcome beyond the accepted ones terminates its node
            trashed = idx >= len(self._accepted_measurement_outcome_pool)
            edge = MeasurementEdge(
                name = self._edge_name("qubit_measurement", step),
                outcome = self._measurement_outcome_pool[idx],
                real_map = (
                    self._qubit_projs_real[idx] 
                    if not self.qubit_measurement_is_ideal 
                    else self._qubit_projs_ideal[idx]
                ),
                ideal_map = [self._qubit_projs_ideal[idx]],
            )              

            final_nodes = self._add_leaves(
                graph, init_nodes.active_nodes(), edge
            )

            for node in final_nodes.active_nodes():
                node.terminated = trashed

            all_final_nodes.nodes += final_nodes.nodes

        return all_final_nodes
    
    # qubit reset ######################################################
    @abstractmethod
    def _qubit_reset_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        pass
            
    @abstractmethod
    def _qubit_reset_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        pass

    def qubit_reset(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        r"""
        Add the edge of the ancilla reset, a :math:`\pi` pulse applied if
        the last outcome is 1, to every node that is not terminated.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step.
        """
        final_nodes = StateEnsemble()

        for node in init_nodes.active_nodes():
            edge_qubit_reset = PropagatorEdge(
                self._edge_name("qubit_reset", step),
                (
                    self._qubit_reset_map_real 
                    if not self.qubit_reset_is_ideal 
                    else self._kraus_to_super(self._qubit_reset_map_ideal)
                ),
                self._qubit_reset_map_ideal,
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_qubit_reset,
                node,
                final_nodes[-1],
            )

        return final_nodes
    
    # check point ######################################################
    def check_point(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        """
        Add two check-point edges to every node that is not terminated:
        the projection onto the space of the ideal states of the node, and
        the projection onto its complement, whose final node is terminated.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the check point, terminated ones included.
        """
        all_final_nodes = StateEnsemble()

        for success in [True, False]:
            edge_check_point = CheckPointEdge(
                name = self._edge_name("check_point", step),
                success = success,
            )
            final_nodes = self._add_leaves(
                graph, init_nodes.active_nodes(), edge_check_point
            )

            for node in final_nodes.active_nodes():
                node.terminated = not success

            all_final_nodes.nodes += final_nodes.nodes

        
        
        return all_final_nodes


class FullCatTreeBuilder(CatTreeBuilder):
    _res_mode_idx: int
    _qubit_mode_idx: int
    res_dim: int
    qubit_dim: int
    res_me_dim: int
    qubit_me_dim: int
    
    # mean photon number rounded to an integer; the current frame rotates
    # at the ancilla frequency for this photon number
    n_bar_ref: int  
    
    c_ops: List[qt.Qobj]
    # name of each collapse operator, in the order of c_ops (the keys of
    # cat_recipe.dressed_jump_ops); None if new_recipe is False, because
    # cat_real does not name its collapse operators
    c_op_names: List[str] | None
    partial_esys: Tuple[np.ndarray, np.ndarray]
    frame_hamiltonian: qt.Qobj

    _accepted_measurement_outcome_pool = [0, 1]

    # step table #######################################################
    # the seven steps of one QEC cycle: the names of the methods that add
    # their edges, in the order of the protocol (the order in which
    # generate_tree calls them), and the prefixes of the edge names in the
    # same order
    STEP_METHOD_NAMES = (
        "idle",
        "qubit_gate_1",
        "parity_mapping",
        "qubit_gate_2",
        "qubit_measurement",
        "feedback_latency",
        "qubit_reset",
    )
    STEP_EDGE_PREFIXES = tuple(
        CatTreeBuilder.EDGE_PREFIXES[name] for name in STEP_METHOD_NAMES
    )
    steps_per_cycle = len(STEP_METHOD_NAMES)

    def __init__(
        self,
        fsweep: FlexibleSweep,
        sim_para: Dict[str, Any],
        new_recipe: bool = False,
    ):
        r"""
        Builder of the tree (``EvolutionTree``) of the QEC cycles of a cat
        code in a resonator with an ancilla. Each edge of the tree holds the
        dynamical map and the ideal operators of one step.

        One cycle has seven steps: idling, :math:`X_{\pi/2}` rotation of the
        ancilla, parity-mapping idling, :math:`X_{\mp\pi/2}` rotation,
        ancilla measurement, feedback latency, and ancilla reset. Operators
        act on the resonator-ancilla space of dimension
        ``res_dim * qubit_dim``, with the tensor order of the sweep
        (``res_mode_idx``), and superoperators are in the qutip ``super``
        representation. Durations are in ns and Hamiltonians in rad/ns. The
        maps are written in the current frame (``frame_transform``).

        Parameters
        ----------
        fsweep: FlexibleSweep
            Single-point sweep that holds the dressed spectrum, the jump
            rates, and the pulse parameters.
        sim_para: Dict[str, Any]
            Simulation parameters: ``res_dim`` and ``qubit_dim`` (numbers of
            resonator and ancilla levels); optionally ``res_me_dim`` and
            ``qubit_me_dim`` (dimensions of the jump operators of the master
            equation), ``higher_order_decay``, and any of the switches of
            ``SWITCH_NAMES``. A switch that ``sim_para`` does not hold keeps
            its class default and can be set as an attribute afterwards.
        new_recipe: bool
            If True, the master equation uses the dressed jump operators of
            ``cat_recipe.cavity_ancilla_me_ingredients``; if False, the
            collapse operators of ``cat_real.cavity_ancilla_me_ingredients``.
        """
        self.fsweep = fsweep
        self.sim_para = sim_para
        self.new_recipe = new_recipe

        self._read_switches(sim_para)

        self._find_sim_param()
        self._generate_cat_ingredients()
        
    def _find_sim_param(
        self,
    ):
        """
        Read the positions of the resonator and the ancilla in the tensor
        product from the sweep, and the dimensions from ``sim_para``.
        """
        assert len(self.fsweep.hilbertspace.subsystem_list) == 2

        if type(self.fsweep.hilbertspace.subsystem_list[0]) == scq.Oscillator:
            self._res_mode_idx = 0
            self._qubit_mode_idx = 1
        elif type(self.fsweep.hilbertspace.subsystem_list[1]) == scq.Oscillator:
            self._res_mode_idx = 1
            self._qubit_mode_idx = 0
        else:
            raise ValueError("The Hilbert space does not contain a resonator.")
        
        self.res_dim = self.sim_para["res_dim"]
        self.qubit_dim = self.sim_para["qubit_dim"]

        # dimensions of the jump operators of the master equation, if given
        try:
            self.res_me_dim = self.sim_para["res_me_dim"]
            self.qubit_me_dim = self.sim_para["qubit_me_dim"]
        except KeyError:
            pass

        # which qubit gate is used
        # qubit = self.fsweep.hilbertspace.subsystem_list[self._qubit_mode_idx]
        # if type(qubit) == scq.Transmon:
        #     self._gate_axis = "x"
        # elif type(qubit) == scq.Fluxonium:
        #     self._gate_axis = "x"
        # else:
        #     raise ValueError("Unknown qubit type.")
        self._gate_axis = "x"
        
    def _generate_cat_ingredients(
        self,
    ):
        """
        Build the static Hamiltonian (rad/ns) and the collapse operators of
        the idling master equation in the dressed basis and the current
        frame, and the frame Hamiltonian (rad/ns).
        """
        if not self.new_recipe:
            (
                self.static_hamiltonian, self.c_ops, self.partial_esys,
                self.frame_hamiltonian
            ) = cat_real.cavity_ancilla_me_ingredients(
                hilbertspace=self.fsweep.hilbertspace,
                res_mode_idx=self._res_mode_idx, 
                qubit_mode_idx=self._qubit_mode_idx,
                res_truncated_dim=self.res_dim, 
                qubit_truncated_dim=self.qubit_dim,
                collapse_parameters={
                    "res_decay": self.fsweep["kappa_s"],
                    "res_excite": self.fsweep["kappa_s"] * self.fsweep["n_th_s"],
                    "res_dephase": 0,
                    "qubit_decay": [
                        [0, self.fsweep["Gamma_up"]], 
                        [self.fsweep["Gamma_down"], 0]],
                    "qubit_dephase": [0, self.fsweep["Gamma_phi"]]
                },
                in_rot_frame=True,
            )
            # cat_real does not name its collapse operators
            self.c_op_names = None
        else:
            self.n_bar = np.abs(self.fsweep["disp"])**2
            self.n_bar_ref = int(np.round(self.n_bar))  
            
            (
                self.static_hamiltonian, self.c_ops, self.partial_esys,
                self.frame_hamiltonian, self.c_op_names
            ) = cat_recipe.cavity_ancilla_me_ingredients(
                fsweep = self.fsweep, 
                res_mode_idx=self._res_mode_idx,
                qubit_mode_idx=self._qubit_mode_idx, 
                res_dim=self.res_dim,
                qubit_dim=self.qubit_dim, 
                res_me_dim=self.res_me_dim,
                qubit_me_dim=self.qubit_me_dim,
                in_rot_frame=True,
                res_n_ref=self.n_bar_ref,
                # False: drop a†|i><j| and ΔN=±2 correlated jumps
                higher_order_decay=self.sim_para.get(
                    "higher_order_decay", True
                ),
                return_jump_names=True,
            )
        
        self.full_esys = (self.fsweep["evals"], self.fsweep["evecs"])
        self.full_dressed_indices = self.fsweep["dressed_indices"]

        # change unit from GHz to rad / ns
        self.static_hamiltonian = self.static_hamiltonian * np.pi * 2
        self.frame_hamiltonian = self.frame_hamiltonian * np.pi * 2
        
    # overall properties ################################################
    @property
    def _total_simulation_time(self) -> float:
        """Duration of one QEC cycle (ns), the sum of the step durations."""
        return (
            self._idling_time
            + self.parity_mapping_time
            + self._qubit_gate_1_time
            + self._qubit_gate_2_time
            + self.qubit_measurement_time
            + self._feedback_latency_time
            + self._qubit_reset_time
        )

    # idling ###########################################################
    def idling_real_by_time(self, time: float) -> qt.Qobj:
        r"""
        Dynamical map :math:`e^{\mathcal L t}` of an idling of a given
        duration, with every jump of the master equation, in the current
        frame.

        Parameters
        ----------
        time : float
            Duration :math:`t` (ns).

        Returns
        -------
        qt.Qobj
            Superoperator on the resonator-ancilla space (qutip ``super``
            representation).
        """
        return cat_real.idling_propagator(
            self.static_hamiltonian,
            self.c_ops,
            float(time),    
            # a float, because a Qobj times a zero-dimensional array is a
            # scipy sparse matrix, not a Qobj
        )
    
    def idling_ideal_by_time(self, time: float) -> List[qt.Qobj]:
        r"""
        Ideal operators of an idling of a given duration, in the current
        frame (``cat_ideal.idling_maps``).

        Parameters
        ----------
        time: float
            Duration (ns).

        Returns
        -------
        List[qt.Qobj]
            Two operators: the evolution without a jump, and the evolution
            with one photon loss.
        """
        return cat_ideal.idling_maps(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            static_hamiltonian=self.static_hamiltonian,
            time=time,
            decay_rate=self.fsweep["jump_a"] if self.new_recipe else self.fsweep["kappa_s"],
            self_Kerr=self.fsweep["K_s"],
        )
    
    def build_idling_process(
        self,
    ) -> None:
        r"""
        Build the dynamical map and the ideal operators of the idling step
        (``step_maps`` and ``ideal_operators``, key ``"idle"``), and the
        identity operator that the reset uses when no pulse is applied.
        """
        self._idling_real = self.idling_real_by_time(self._idling_time)
        self._idling_ideal = self.idling_ideal_by_time(self._idling_time)

        self._identity = cat_ideal.identity(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )

    @property
    def _idling_time(self) -> float:
        """Duration of the idling step (ns); zero if ``idling_is_ideal``."""
        return float(self.fsweep["T_W"] * (1 - self.idling_is_ideal))

    # qubit gate #######################################################
    def frame_transform(
        self,
        propagator: qt.Qobj,
        total_time: float,
        type: Literal[
            "rot2current", 
            "current2rot", 
            "lab2current",
            "current2lab",
            "lab2rot",
            "rot2lab",
        ],
        init_time: float = 0,
    ) -> qt.Qobj:
        r"""
        Transform a propagator, or a dynamical map, from one frame to another.

        Three frames are used:

        - ``lab``: the lab frame, with the Hamiltonian
          ``frame_hamiltonian + static_hamiltonian``;
        - ``current``: the frame in which the linear part of the natural
          frequencies is removed, with the Hamiltonian
          ``static_hamiltonian``;
        - ``rot``: the rotating frame, in which all natural frequencies
          vanish.

        With :math:`\hat H_{\rm t}` the Hamiltonian of the initial frame
        minus that of the final frame and
        :math:`\hat U_{\rm t}(t)=e^{-i\hat H_{\rm t}t}`, a propagator
        :math:`\hat U` from the time :math:`t_0` to :math:`t_0+T` becomes
        :math:`\hat U_{\rm t}^\dagger(t_0+T)\,\hat U\,\hat U_{\rm t}(t_0)`,
        and a dynamical map is transformed with the superoperators of the
        same two unitaries.

        Parameters
        ----------
        propagator : qt.Qobj
            The unitary or the superoperator to transform.
        total_time : float
            The duration :math:`T` of the evolution that it describes (ns).
        type : str
            The initial and final frame, as ``"<initial>2<final>"``.
        init_time : float
            The time :math:`t_0` at which the evolution starts (ns).

        Returns
        -------
        qt.Qobj
            The propagator in the final frame, a unitary or a dynamical map
            like the one given.
        """
        init, final = type.split("2")

        if init == "rot":
            init_ham = 0
        elif init == "current":
            init_ham = self.static_hamiltonian
        elif init == "lab":
            init_ham = self.frame_hamiltonian + self.static_hamiltonian

        if final == "rot":
            final_ham = 0
        elif final == "current":
            final_ham = self.static_hamiltonian
        elif final == "lab":
            final_ham = self.frame_hamiltonian + self.static_hamiltonian

        trans_ham = init_ham - final_ham

        prop_free = lambda t: (-1j * trans_ham * t).expm()
        if propagator.issuper:
            return (
                qt.to_super(prop_free(total_time + init_time).dag())
                * propagator
                * qt.to_super(prop_free(init_time))
            )
        return (
            prop_free(total_time + init_time).dag()
            * propagator
            * prop_free(init_time)
        )

    @property
    def _gate_collapse_operators(self) -> List[qt.Qobj] | None:
        r"""
        Collapse operators with which the ancilla pulses are integrated:
        the values of ``gate_collapse_operators``, or None if
        ``gate_qubit_decoherence`` is False.
        """
        if not self.gate_qubit_decoherence:
            return None
        return list(self.gate_collapse_operators.values())

    @property
    def gate_collapse_operators(self) -> Dict[str, qt.Qobj]:
        r"""
        Collapse operators :math:`\sqrt{\gamma}\hat L` of the ancilla
        pulses, by name, for either value of ``gate_qubit_decoherence``.

        They are the jump operators of the idling master equation that act
        on the ancilla alone: relaxation ``jump_01``, excitation
        ``jump_10``, dephasing ``jump_11``, leakage to level 2
        ``jump_21``, ... (the ``jump_ij`` keys of
        ``cat_recipe.dressed_jump_ops``). They do not change the photon
        number of the resonator. A jump operator with a vanishing rate is
        left out. If ``gate_qubit_decoherence`` is True, the pulses are
        integrated with these operators, in this order.

        Returns
        -------
        Dict[str, qt.Qobj]
            Name to operator (ns^-1/2), written on the full Hilbert space
            of the sweep with the dressed eigenvectors, the form that
            ``cat_real.qubit_gate`` takes.
        """
        if not self.new_recipe:
            raise ValueError("gate_qubit_decoherence needs new_recipe=True.")
        jumps = cat_recipe.dressed_jump_ops(
            self.fsweep, self._res_mode_idx, self.res_dim, self.qubit_dim,
            self.qubit_me_dim,
            higher_order_decay=self.sim_para.get("higher_order_decay", True),
            op_families=["jump_ij"],
        )
        # an operator in the dressed basis, written on the full space
        isometry = trans_by_kets(self.partial_esys[1])
        return {
            name: isometry * qt.Qobj(op.full()) * isometry.dag()
            for name, op in jumps.items() if np.any(op.full())
        }

    def gate_map_in_current_frame(
        self,
        lab_map: qt.Qobj,
        total_time: float,
        frame_change: qt.Qobj | None = None,
        as_operator: bool = False,
    ) -> qt.Qobj:
        r"""
        Dynamical map of an ancilla pulse in the current frame, with the same
        ancilla frame change before and after the pulse.

        Parameters
        ----------
        lab_map : qt.Qobj
            The lab-frame unitary or superoperator of the pulse, an entry of
            ``lab_frame_pulses``.
        total_time : float
            The pulse duration (ns).
        frame_change : qt.Qobj | None
            The unitary of the ancilla frame change that is applied before
            and after the pulse (``ancilla_frame_change``). If None, no frame
            change.
        as_operator : bool
            If True, return the operator of the pulse instead of its
            superoperator. Only a closed-system pulse (``lab_map`` is a
            unitary) has one; a superoperator raises a ValueError.

        Returns
        -------
        qt.Qobj
            The superoperator of the pulse (qutip ``super``
            representation); its operator if ``as_operator``.
        """
        current = self.frame_transform(
            lab_map, total_time=total_time, type="lab2current",
        )
        if frame_change is None:
            frame_change = cat_ideal.identity(
                res_dim=self.res_dim, qubit_dim=self.qubit_dim,
                res_mode_idx=self._res_mode_idx, superop=False,
            )
        if as_operator:
            # a dynamical map with jumps is not a single operator
            if lab_map.issuper:
                raise ValueError(
                    "as_operator needs the unitary of a closed-system pulse "
                    "(gate_qubit_decoherence = False), not a superoperator."
                )
            return frame_change * current * frame_change
        if lab_map.issuper:
            return qt.to_super(frame_change) * current * qt.to_super(frame_change)
        return self._kraus_to_super([frame_change * current * frame_change])

    def ancilla_frame_change(
        self,
        angle: float,
    ) -> qt.Qobj:
        r"""
        Ancilla frame change (virtual Z gate)
        :math:`\hat Z_\Phi=e^{-i\Phi\hat n_{\mathrm q}}`, with
        :math:`\hat n_{\mathrm q}` the ancilla level index.

        Advancing the ancilla frame by :math:`\Phi` shifts the phase of
        every later drive pulse; in the current frame it acts as this phase
        gate on every photon-number subspace.

        Parameters
        ----------
        angle : float
            :math:`\Phi` (rad).

        Returns
        -------
        qt.Qobj
            The unitary on the resonator-ancilla space; the identity if
            ``use_frame_update`` is False.
        """
        if not self.use_frame_update:
            angle = 0.0
        return cat_ideal.qubit_frame_change(
            self.res_dim, self.qubit_dim, self._res_mode_idx, angle
        )

    @property
    def gate_frame_phase(self) -> float:
        r"""
        Phase :math:`\varphi_{\mathrm S}` of the ancilla phase gate that the
        Stark shift leaves on each side of the rotation of a :math:`\pi/2`
        pulse (sweep key ``gate_frame_phase``). The protocol removes it with
        an ancilla frame change by :math:`-\varphi_{\mathrm S}` before and
        after each :math:`\pi/2` pulse.

        Returns
        -------
        float
            The phase (rad); zero if ``use_frame_update`` is False.
        """
        if not self.use_frame_update:
            return 0.0
        return float(self.fsweep["gate_frame_phase"])

    @property
    def pm_frame_phase(self) -> float:
        r"""
        Ancilla phase :math:`\chi_1't_{\mathrm{pm}}\langle d^2\rangle_+` that
        the qubit-state-dependent resonator non-linearity :math:`\chi_1'`
        produces during the parity-mapping idling of duration
        :math:`t_{\mathrm{pm}}` (sweep key ``pm_frame_phase``). Here
        :math:`\langle d^2\rangle_+` is the even-parity average of
        :math:`d^2` after the idling step (the first entry of the sweep key
        ``cat_mean_d2``). The protocol removes the phase with an ancilla
        frame change by its negative at the end of the parity-mapping
        idling.

        Returns
        -------
        float
            The phase (rad); zero if ``use_frame_update`` or
            ``use_chi_prime_frame_update`` is False.
        """
        if not (self.use_frame_update and self.use_chi_prime_frame_update):
            return 0.0
        return float(self.fsweep["pm_frame_phase"])

    def build_qubit_gate_process(
        self,
        num_cpus: int = 8,
    ) -> None:
        r"""
        Build the dynamical maps and the ideal operators of the
        :math:`\pm\pi/2` ancilla pulses (``step_maps``, ``ideal_operators``,
        and ``lab_frame_pulses``, keys ``"qubit_gate_p"`` and
        ``"qubit_gate_m"``).

        Parameters
        ----------
        num_cpus: int
            Number of worker processes that integrate the pulse.
        """
        self._gate_p_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            eigensys = self.full_esys,
            dressed_indices = self.full_dressed_indices,
            rotation_angle = np.pi / 2,
            gate_params = self.fsweep,
            num_cpus = num_cpus,
            drive_operator = self.drive_operator,
            simulated_qubit_dim = self.gate_simulated_qubit_dim,
            collapse_operators = self._gate_collapse_operators,
        )
        # frame changes by minus the phase of the Stark shift on both sides
        # of the pulse (identity when use_frame_update is off)
        stark_frame = self.ancilla_frame_change(-self.gate_frame_phase)
        self._qubit_gate_p_real = self.gate_map_in_current_frame(
            self._gate_p_lab_real,
            total_time = self.fsweep["tau_p_eff"],
            frame_change = stark_frame,
        )
        self._gate_m_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            eigensys = self.full_esys,
            dressed_indices = self.full_dressed_indices,
            rotation_angle = - np.pi / 2,
            gate_params = self.fsweep,
            num_cpus = num_cpus,
            drive_operator = self.drive_operator,
            simulated_qubit_dim = self.gate_simulated_qubit_dim,
            collapse_operators = self._gate_collapse_operators,
        )
        self._qubit_gate_m_real = self.gate_map_in_current_frame(
            self._gate_m_lab_real,
            total_time = self.fsweep["tau_p_eff"],
            frame_change = stark_frame,
        )
        
        # ideal ancilla rotations by +pi/2 (p, plus) and -pi/2 (m, minus)
        assert self._qubit_gate_1_rabi_freq == self._qubit_gate_2_rabi_freq
        if self.use_improved_gate:
            # the ancilla rotation with the phases that the dispersive shift,
            # the self-Kerr, and chi_prime produce during the pulse
            self._qubit_gate_p_ideal = [self.finite_duration_qubit_rotation(np.pi / 2)]
            self._qubit_gate_m_ideal = [self.finite_duration_qubit_rotation(-np.pi / 2)]
        else:
            # the ancilla rotations alone, without these phases
            self._qubit_gate_p_ideal = [cat_ideal.qubit_rot_propagator(
                res_dim=self.res_dim, qubit_dim=self.qubit_dim,
                res_mode_idx=self._res_mode_idx,
                angle=np.pi/2, axis=self._gate_axis, superop=False,
            )]
            self._qubit_gate_m_ideal = [cat_ideal.qubit_rot_propagator(
                res_dim=self.res_dim, qubit_dim=self.qubit_dim,
                res_mode_idx=self._res_mode_idx,
                angle=-np.pi/2, axis=self._gate_axis, superop=False,
            )]

    def _qubit_gate_2_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        r"""
        Dynamical map of gate 2 for a measurement record: the
        :math:`-\pi/2` pulse for even parity and the :math:`+\pi/2` pulse
        for odd parity. This choice leaves the ancilla in :math:`|0\rangle`
        while the parity is unchanged, which minimizes the probability that
        the ancilla is excited.
        """
        if self._current_parity(meas_record) == 0:
            # even parity, use the opposite gate
            return self._qubit_gate_m_real
        else:
            # odd parity, use the same gate
            return self._qubit_gate_p_real
        
    def _qubit_gate_2_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        r"""
        Ideal operators of gate 2 for a measurement record: those of the
        :math:`-\pi/2` pulse for even parity and of the :math:`+\pi/2`
        pulse for odd parity (see ``_qubit_gate_2_map_real``).
        """
        if self._current_parity(meas_record) == 0:
            # even parity, use the opposite gate
            return self._qubit_gate_m_ideal
        else:
            # odd parity, use the same gate
            return self._qubit_gate_p_ideal

    @property
    def _qubit_gate_1_time(self) -> float:
        """
        Duration of gate 1 (ns), ``tau_p_eff`` of the sweep, also if
        ``gate_1_is_ideal``.
        """
        return float(self.fsweep["tau_p_eff"])
    
    @property
    def _qubit_gate_2_time(self) -> float:
        """
        Duration of gate 2 (ns), ``tau_p_eff`` of the sweep, also if
        ``gate_2_is_ideal``.
        """
        return float(self.fsweep["tau_p_eff"])
    
    @property
    def _qubit_gate_1_rabi_freq(self) -> float:
        r"""
        Fictitious frequency :math:`f` (1/ns) such that
        :math:`\chi_1/(2f)` equals the angle
        :math:`\chi_1 t_g \nu` per photon of the controlled rotation of the
        resonator during gate 1, with :math:`t_g` the gate duration and
        :math:`\nu` the envelope shape number ``GATE_NU_HALF``.
        """
        return 1 / (2 * self._qubit_gate_1_time * GATE_NU_HALF)
    
    @property
    def _qubit_gate_2_rabi_freq(self) -> float:
        """The same frequency for gate 2; see ``_qubit_gate_1_rabi_freq``."""
        return 1 / (2 * self._qubit_gate_2_time * GATE_NU_HALF)

    # parity mapping ###################################################
    def build_parity_mapping_process(
        self,
    ) -> None:
        r"""
        Build the dynamical map and the ideal operator of the parity-mapping
        idling (``step_maps`` and ``ideal_operators``, key
        ``"parity_mapping"``).
        """
        # the frame change at the end of the idling removes the cat-averaged
        # ancilla phase of chi' (identity when use_frame_update is off)
        self._parity_mapping_real = (
            qt.to_super(self.ancilla_frame_change(-self.pm_frame_phase))
            * self.idling_real_by_time(self.parity_mapping_time)
        )

        if self.use_improved_parity_mapping:
            # the controlled rotation of the resonator by the parity-mapping
            # angle, with the phases that the self-Kerr and chi_prime produce
            # during the idling
            self._parity_mapping_ideal = [self.finite_duration_parity_mapping()]
        else:
            self._parity_mapping_ideal = [
                self._parity_mapping_ideal_n_ref_frame(
                    angle=np.pi,
                    pm_qubit_state=1,
                )
            ]

    @property
    def average_pm_interaction(self) -> float:
        r"""
        Dispersive shift per photon at the mean photon number :math:`\bar n`,
        :math:`\chi_1=\chi+(2\bar n-1)\chi_1'`, with :math:`\chi` and :math:`\chi_1'`
        the sweep keys ``chi_sa`` and ``chi_prime_sa``. It is the rate of
        the controlled rotation of the resonator during the parity mapping.

        Returns
        -------
        float
            :math:`\chi_1` (rad/ns).
        """
        # the ancilla frequency shifts by chi * n + chi_prime * n * (n - 1)
        # for n photons; this is its derivative with respect to n at n_bar
        return float(
            self.fsweep["chi_sa"] 
            + (2 * self.n_bar - 1) * self.fsweep["chi_prime_sa"]
        )
        
        # # another option:
        # # take the dispersive shift difference between n = n_bar and n = n_bar - 0
        # return float(
        #     self.fsweep["chi_sa"] * n_bar
        #     + n_bar * (n_bar - 1) * self.fsweep["chi_prime_sa"]
        # ) / n_bar   # divide by n_bar to get the frequency shift per photon

    @property
    def parity_mapping_angle(self) -> float:
        r"""
        Angle per photon of the controlled rotation of the resonator during
        the parity-mapping idling: :math:`\pm\pi` minus the angles that the two
        :math:`\pi/2` pulses contribute.

        Returns
        -------
        float
            The angle (rad), with the sign of ``average_pm_interaction``.
        """
        if self.average_pm_interaction >= 0:
            full_angle = np.pi
        else:
            full_angle = -np.pi
            
        # angle that the two pi/2 pulses already contribute
        partial_angle = (
            self.average_pm_interaction / 2 / self._qubit_gate_1_rabi_freq
            + self.average_pm_interaction / 2 / self._qubit_gate_2_rabi_freq
        )
        return full_angle - partial_angle

    @property
    def parity_mapping_time(self) -> float:
        r"""
        Duration of the parity-mapping idling,
        ``parity_mapping_angle / average_pm_interaction``, limited to the
        duration ``T_W`` of the idling step. It is the same if
        ``parity_mapping_is_ideal``.

        Returns
        -------
        float
            The duration (ns).
        """
        t = float(self.parity_mapping_angle / self.average_pm_interaction)

        if t > self.fsweep["T_W"]:
            # happens for a dispersive shift that is unreasonably small
            if graph_settings.ISSUE_WARNING:
                warn("The parity mapping time is longer than the waiting time. Set the time to be the waiting time.\n")
            t = float(self.fsweep["T_W"])

        return t
    
    def _parity_mapping_ideal_n_ref_frame(
        self,
        angle: float,
        pm_qubit_state: int = 1,
    ) -> qt.Qobj:
        r"""
        Unitary of the controlled rotation of the resonator by ``angle``
        (rad) per photon, in the current frame,
        :math:`\exp[-i\,\theta\,(\hat a^\dagger\hat a-n_{\rm ref})\,|q\rangle\langle q|]`,
        with :math:`\theta` = ``angle``, :math:`q` = ``pm_qubit_state`` the
        ancilla level that acquires the phase, and :math:`n_{\rm ref}` =
        ``n_bar_ref``. The phase vanishes for :math:`n_{\rm ref}` photons,
        because the current frame rotates at the ancilla frequency for that
        photon number.
        """
        return cat_ideal.parity_mapping_n_ref_frame(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=angle,
            n_ref=self.n_bar_ref,
            qubit_state=pm_qubit_state,
        )
    
    def _Kerr_rotation_n_ref_frame(
        self,
        angle: float,
    ) -> qt.Qobj:
        r"""
        Unitary
        :math:`\exp[-i\,\theta\,(\hat a^\dagger\hat a-n_{\rm ref})^2]` of a
        resonator phase that is quadratic in the photon number counted from
        :math:`n_{\rm ref}` = ``n_bar_ref``, with :math:`\theta` = ``angle``
        (rad).
        """
        return cat_ideal.res_Kerr_rotation_n_ref_frame(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=angle,
            n_ref=self.n_bar_ref,
        )
        
    # def parity_mapping(
    #     self,
    #     step: int | str,
    #     graph: EvolutionTree,
    #     init_nodes: StateEnsemble,
    # ) -> StateEnsemble:
    #     """
    #     Overwrite the base class parity mapping - as when the parity_mapping_is_ideal 
    #     is true, the self._parity_mapping_ideal is not TP and can't serve as the 
    #     ideal map for actual process.
    #     """
    #     edge_parity_mapping = PropagatorEdge(
    #         name = f"PM ({step})",
    #         real_map = (
    #             self._parity_mapping_real
    #             if not self.parity_mapping_is_ideal 
    #             else self._kraus_to_super(self._parity_mapping_constructed)
    #         ),
    #         ideal_maps = (
    #             self._parity_mapping_ideal
    #             if not self.parity_mapping_is_ideal
    #             else self._parity_mapping_constructed
    #         )
    #     )

    #     return self._add_leaves(graph, init_nodes.active_nodes(), edge_parity_mapping)

    # qubit measurement ################################################
    def build_qubit_measurement_process(
        self,
    ) -> None:
        r"""
        Build the dynamical maps and the ideal operators of the ancilla
        measurement (``step_maps`` and ``ideal_operators``, key
        ``"qubit_measurement"``).
        """
        self._qubit_projs_constructed = cat_ideal.qubit_projectors(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )
        # the ideal operator of an outcome is the projector after the
        # no-jump evolution over the measurement duration
        idling_prop = self.idling_ideal_by_time(self.qubit_measurement_time)[0]
        self._qubit_projs_ideal = [
            proj * idling_prop for proj in self._qubit_projs_constructed
        ]

        # dynamical maps of the measurement, with the assignment errors M_ge
        # and M_eg of the sweep
        self._qubit_projs_real = self.measurement_maps()

        if len(self._qubit_projs_ideal) != len(self._accepted_measurement_outcome_pool):
            # outcomes beyond the accepted ones are labelled by their
            # ancilla level
            self._measurement_outcome_pool = (
                self._accepted_measurement_outcome_pool
                + list(range(len(self._accepted_measurement_outcome_pool), len(self._qubit_projs_ideal)))
            )
            if len(self._measurement_outcome_pool) != len(set(self._measurement_outcome_pool)):
                raise ValueError("The measurement outcome pool should not contain duplicate values.")
            else:
                if graph_settings.ISSUE_WARNING:
                    warn("The number of accepted measurement outcomes is not equal to"
                        f"defined projectors, use {self._measurement_outcome_pool} instead.\n")
        else:
            self._measurement_outcome_pool = self._accepted_measurement_outcome_pool

        assert len(self._qubit_projs_ideal) == len(self._qubit_projs_real)

    @property
    def qubit_measurement_time(self) -> float:
        r"""
        Duration of the ancilla measurement, ``tau_m`` of the sweep; zero if
        ``qubit_measurement_is_ideal``.

        Returns
        -------
        float
            The duration (ns).
        """
        return float(self.fsweep["tau_m"] * (1 - self.qubit_measurement_is_ideal))

    @property
    def measurement_confusion_matrix(self) -> np.ndarray:
        r"""
        Confusion matrix of the ancilla measurement, built from the
        assignment-error probabilities ``M_ge`` and ``M_eg`` of the sweep
        (``cat_real.assignment_confusion_matrix``).

        Returns
        -------
        np.ndarray
            The ``qubit_dim`` x ``qubit_dim`` matrix whose element [i, j] is
            the probability of reporting j when the ancilla is in level i.
        """
        return cat_real.assignment_confusion_matrix(
            self.qubit_dim, self.fsweep["M_ge"], self.fsweep["M_eg"],
        )

    def measurement_maps(
        self,
        confusion_matrix: np.ndarray | Literal["sweep"] | None = "sweep",
        time: float | None = None,
    ) -> List[qt.Qobj]:
        r"""
        Dynamical maps of the ancilla measurement with assignment errors, one
        per reported outcome.

        The system idles for the measurement duration :math:`t`, with every jump
        of the idling master equation, and the ancilla is then projected on
        its levels. The element :math:`C_{ij}` of the confusion matrix is the
        probability of reporting the outcome j when the ancilla is projected
        on level i. The map of the reported outcome j is

        .. math::
            \mathcal M_j = \sum_i C_{ij}\,\mathcal P_i\,e^{\mathcal L t},

        with :math:`\mathcal P_i\hat\rho=\hat P_i\hat\rho\hat P_i` the
        projection of the ancilla on level i and :math:`\mathcal L` the
        Liouvillian of the idling. The maps are in the current frame.

        Parameters
        ----------
        confusion_matrix: np.ndarray | "sweep" | None
            The ``qubit_dim`` x ``qubit_dim`` confusion matrix :math:`C_{ij}`.
            ``"sweep"`` (default): ``measurement_confusion_matrix``, from
            ``M_ge`` and ``M_eg`` of the sweep. None: no assignment error.
        time: float | None
            Duration :math:`t` of the idling before the projection (ns). If None,
            the measurement duration of the builder (``tau_m`` of the sweep,
            or zero if ``qubit_measurement_is_ideal``).

        Returns
        -------
        List[qt.Qobj]
            ``qubit_dim`` superoperators, ordered by the reported outcome.
        """
        if isinstance(confusion_matrix, str):
            if confusion_matrix != "sweep":
                raise ValueError(
                    "confusion_matrix is a matrix, None, or 'sweep'."
                )
            confusion_matrix = self.measurement_confusion_matrix
        if time is None:
            time = self.qubit_measurement_time

        real_projs = cat_real.qubit_projectors(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            confusion_matrix=confusion_matrix,
            ensamble_average=True,  # one map per reported outcome, so that the list matches the list of ideal operators
        )
        # the system idles for the measurement duration before the projection
        idling_prop = self.idling_real_by_time(time)
        return [
            proj * idling_prop for proj in real_projs
        ]

    def qubit_measurement(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        """
        Add one edge of the ancilla measurement for each outcome to every
        node that is not terminated. The final node of an outcome other
        than 0 and 1 is terminated. If ``qubit_measurement_is_ideal``, the
        dynamical map and the ideal operator of an outcome are the
        projector of the ancilla level alone.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step, terminated ones included.
        """
        all_final_nodes = StateEnsemble()

        for idx in range(len(self._qubit_projs_ideal)):
            # an outcome beyond the accepted ones terminates its node
            trashed = idx >= len(self._accepted_measurement_outcome_pool)

            edge_qubit_measurement = MeasurementEdge(
                name = self._edge_name("qubit_measurement", step),
                outcome = self._measurement_outcome_pool[idx],
                real_map = (
                    self._qubit_projs_real[idx] 
                    if not self.qubit_measurement_is_ideal 
                    else qt.to_super(self._qubit_projs_constructed[idx])
                ),
                ideal_map = (
                    [self._qubit_projs_ideal[idx]]
                    if not self.qubit_measurement_is_ideal 
                    else [self._qubit_projs_constructed[idx]]
                ),
            )                    

            final_nodes = self._add_leaves(
                graph, init_nodes.active_nodes(), edge_qubit_measurement
            )

            for node in final_nodes.active_nodes():
                node.terminated = trashed

            all_final_nodes.nodes += final_nodes.nodes
            

        return all_final_nodes

    # feedback latency #################################################
    _feedback_latency_real: qt.Qobj
    _feedback_latency_ideal: List[qt.Qobj]

    def build_feedback_latency_process(
        self,
    ):
        r"""
        Build the dynamical map and the ideal operator of the feedback
        latency, the idling between the end of the measurement and the
        conditional reset (``step_maps`` and ``ideal_operators``, key
        ``"feedback_latency"``).

        The outcome is already recorded, so the step has a single edge.
        The dynamical map is the idling with every jump for the duration
        ``tau_FD``. The ideal operator is the evolution without a jump over
        the same duration: a photon loss during the latency is not a
        correctable error, because the parity was recorded before it.
        """
        self._feedback_latency_real = self.idling_real_by_time(
            self._feedback_latency_time)
        self._feedback_latency_ideal = [
            self.idling_ideal_by_time(self._feedback_latency_time)[0]
        ]

    @property
    def _feedback_latency_time(self) -> float:
        r"""
        Duration of the feedback latency (ns): ``tau_FD`` of the sweep;
        zero, with a warning, if the sweep has no such key, and zero if
        ``feedback_latency_is_ideal``.
        """
        try:
            tau_FD = self.fsweep["tau_FD"]
        except KeyError:
            # a forgotten key would otherwise remove the step silently
            warn(
                "The sweep has no 'tau_FD': the feedback-latency step is built "
                "with zero duration. Add tau_FD (ns) to the parameters, or set "
                "it to 0 explicitly for a cycle without latency.",
                UserWarning, stacklevel=2,
            )
            return 0.0
        return float(tau_FD * (1 - self.feedback_latency_is_ideal))

    def feedback_latency(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        r"""
        Add the edge of the feedback latency to every node that is not
        terminated.

        Parameters
        ----------
        step : int | str
            Label of the step in the edge name.
        graph : EvolutionTree
            Tree that receives the edges and the new nodes.
        init_nodes : StateEnsemble
            Nodes that represent the states before the step.

        Returns
        -------
        StateEnsemble
            The nodes after the step.
        """
        edge_feedback_latency = PropagatorEdge(
            self._edge_name("feedback_latency", step),
            self._feedback_latency_real,
            self._feedback_latency_ideal,
        )
        return self._add_leaves(
            graph, init_nodes.active_nodes(), edge_feedback_latency)

    # qubit reset ######################################################
    def build_qubit_reset_process(
        self,
        num_cpus: int = 8,
    ) -> None:
        r"""
        Build the dynamical map and the ideal operator of the reset
        :math:`\pi` pulse (``step_maps``, ``ideal_operators``, and
        ``lab_frame_pulses``, key ``"qubit_reset"``).

        Parameters
        ----------
        num_cpus : int
            Number of worker processes that integrate the pulse.
        """
        self._reset_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            eigensys = self.full_esys,
            dressed_indices = self.full_dressed_indices,
            rotation_angle = np.pi,
            gate_params = self.fsweep,
            num_cpus = num_cpus,
            drive_operator = self.drive_operator,
            simulated_qubit_dim = self.gate_simulated_qubit_dim,
            collapse_operators = self._gate_collapse_operators,
        )
        # the reset has no frame change
        self._qubit_reset_real = self.gate_map_in_current_frame(
            self._reset_lab_real, total_time = self._qubit_reset_time,
        )
        
        if self.use_improved_gate:
            # the pi rotation of the ancilla with the phases that the
            # dispersive shift, the self-Kerr, and chi_prime produce during
            # the pulse
            self._qubit_reset_ideal = [self.finite_duration_qubit_reset()]
        else:
            # the pi rotation of the ancilla alone, without these phases
            self._qubit_reset_ideal = [cat_ideal.qubit_rot_propagator(
                res_dim=self.res_dim, qubit_dim=self.qubit_dim,
                res_mode_idx=self._res_mode_idx,
                angle=np.pi, axis=self._gate_axis, superop=False,
            )]

    def _qubit_reset_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        r"""
        Dynamical map of the reset for a measurement record: the
        :math:`\pi` pulse if the last outcome is 1, else the identity.
        """
        if meas_record[-1] == 1:
            return self._qubit_reset_real
        elif meas_record[-1] == 0:
            return qt.to_super(self._identity)
        else:
            # an outcome that is not accepted; its node is terminated
            return qt.to_super(self._identity)
            
    def _qubit_reset_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        r"""
        Ideal operators of the reset for a measurement record: that of the
        :math:`\pi` pulse if the last outcome is 1, else the identity.
        """
        if meas_record[-1] == 1:
            return self._qubit_reset_ideal
        elif meas_record[-1] == 0:
            return [self._identity]
        else:
            # an outcome that is not accepted; its node is terminated
            return [self._identity]
        
    @property
    def _qubit_reset_time(self) -> float:
        """
        Duration of the reset pulse (ns), twice ``tau_p_eff`` of the sweep;
        zero if ``qubit_reset_is_ideal``.
        """
        return float(self.fsweep["tau_p_eff"] * 2 * (1 - self.qubit_reset_is_ideal))
        
    # parameters, dynamical maps, and ideal operators of the builder ####
    @property
    def res_mode_idx(self) -> int:
        r"""
        Position of the resonator in the tensor product, 0 or 1.
        """
        return self._res_mode_idx

    @property
    def qubit_mode_idx(self) -> int:
        r"""
        Position of the ancilla in the tensor product, 0 or 1.
        """
        return self._qubit_mode_idx

    @property
    def collapse_operators(self) -> Dict[str, qt.Qobj]:
        r"""
        Collapse operators :math:`\sqrt{\gamma}\hat L` of the idling master
        equation, by name.

        The names are the keys of ``cat_recipe.dressed_jump_ops``:
        ``jump_a`` (photon loss, :math:`\hat a`), ``jump_adag`` (photon gain,
        :math:`\hat a^\dagger`), ``jump_adag_a`` (resonator dephasing,
        :math:`\hat a^\dagger\hat a`), ``jump_ij`` (:math:`|i\rangle\langle j|`
        of the ancilla, e.g. ``jump_01`` for the ancilla relaxation and
        ``jump_21`` for the leakage from :math:`|1\rangle` to :math:`|2\rangle`),
        ``jump_a_ij`` (:math:`\hat a|i\rangle\langle j|`), and ``jump_adag_ij``
        (:math:`\hat a^\dagger|i\rangle\langle j|`). An operator with a
        vanishing rate is present as a zero matrix.

        Returns
        -------
        Dict[str, qt.Qobj]
            Name to operator on the ``res_dim * qubit_dim`` space, in the
            order of ``c_ops``, with :math:`\gamma` in 1/ns.
        """
        if not self.new_recipe:
            raise ValueError(
                "The collapse operators have names only with new_recipe=True."
            )
        names = getattr(self, "c_op_names", None)
        if names is None:
            # a builder that does not hold c_op_names: the function that
            # built c_ops gives the names in the same order
            names = list(cat_recipe.dressed_jump_ops(
                self.fsweep, self._res_mode_idx, self.res_dim, self.qubit_dim,
                self.qubit_me_dim,
                higher_order_decay=self.sim_para.get("higher_order_decay", True),
            ).keys())
        # a mismatch means the names do not belong to these operators
        if len(names) != len(self.c_ops):
            raise ValueError(
                f"{len(names)} names for {len(self.c_ops)} collapse operators: "
                "the builder was made with another set of jumps."
            )
        return dict(zip(names, self.c_ops))

    @property
    def step_durations(self) -> Dict[str, float]:
        r"""
        Duration of each step of one QEC cycle.

        Returns
        -------
        Dict[str, float]
            Step (the entries of ``STEP_METHOD_NAMES``) to duration (ns), in
            the order of the protocol. The idling, the measurement, the
            feedback latency, and the reset have zero duration if their
            ``*_is_ideal`` switch is True.
        """
        return {
            "idle": self._idling_time,
            "qubit_gate_1": self._qubit_gate_1_time,
            "parity_mapping": self.parity_mapping_time,
            "qubit_gate_2": self._qubit_gate_2_time,
            "qubit_measurement": self.qubit_measurement_time,
            "feedback_latency": self._feedback_latency_time,
            "qubit_reset": self._qubit_reset_time,
        }

    def finite_duration_qubit_rotation(
        self,
        angle: float,
        duration: float | None = None,
    ) -> qt.Qobj:
        r"""
        Ideal operator of a :math:`\pm\pi/2` ancilla pulse of a given duration,
        with the phases that the dispersive shift, the self-Kerr, and
        :math:`\chi_1'` produce during the pulse
        (``cat_ideal.finite_duration_qubit_rotation`` with the parameters
        of this builder), in the current frame.

        The tree uses it as the ideal operator of the pulse if
        ``use_improved_gate`` is True.

        Parameters
        ----------
        angle: float
            Rotation angle (rad), :math:`+\pi/2` or :math:`-\pi/2`.
        duration: float | None
            Pulse duration (ns). If None, the duration of the builder's
            :math:`\pi/2` pulses.

        Returns
        -------
        qt.Qobj
            Operator on the resonator-ancilla space.
        """
        if duration is None:
            duration = self._qubit_gate_1_time
        return cat_ideal.finite_duration_qubit_rotation(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=angle,
            duration=duration,
            chi_1=self.average_pm_interaction,
            K_s=self.fsweep["K_s"],
            chi_prime=self.fsweep["chi_prime_sa"],
            n_ref=self.n_bar_ref,
            nu=GATE_NU_HALF,
            axis=self._gate_axis,
        )

    def finite_duration_qubit_reset(
        self,
        duration: float | None = None,
    ) -> qt.Qobj:
        r"""
        Ideal operator of the reset :math:`\pi` pulse of a given duration, with
        the phases that the dispersive shift, the self-Kerr, and
        :math:`\chi_1'` produce during the pulse
        (``cat_ideal.finite_duration_qubit_reset`` with the parameters of
        this builder), in the current frame.

        The tree uses it as the ideal operator of the reset if
        ``use_improved_gate`` is True.

        Parameters
        ----------
        duration: float | None
            Pulse duration (ns). If None, the duration of the builder's reset
            pulse, which is zero if ``qubit_reset_is_ideal``.

        Returns
        -------
        qt.Qobj
            Operator on the resonator-ancilla space.
        """
        if duration is None:
            duration = self._qubit_reset_time
        return cat_ideal.finite_duration_qubit_reset(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            duration=duration,
            chi_1=self.average_pm_interaction,
            K_s=self.fsweep["K_s"],
            chi_prime=self.fsweep["chi_prime_sa"],
            n_ref=self.n_bar_ref,
            axis=self._gate_axis,
        )

    def finite_duration_parity_mapping(
        self,
        angle: float | None = None,
        duration: float | None = None,
    ) -> qt.Qobj:
        r"""
        Ideal operator of the parity-mapping idling: the controlled
        rotation of the resonator with the phases that the self-Kerr and
        :math:`\chi_1'` produce during the idling
        (``cat_ideal.finite_duration_parity_mapping`` with the parameters of
        this builder), in the current frame.

        The tree uses it as the ideal operator of the step if
        ``use_improved_parity_mapping`` is True.

        Parameters
        ----------
        angle: float | None
            Angle of the controlled rotation of the resonator per photon
            (rad). If None, ``parity_mapping_angle``.
        duration: float | None
            Idling duration (ns). If None, ``parity_mapping_time``. For an
            idling of another duration t, pass
            ``angle = average_pm_interaction * t`` as well.

        Returns
        -------
        qt.Qobj
            Unitary on the resonator-ancilla space.
        """
        if angle is None:
            angle = self.parity_mapping_angle
        if duration is None:
            duration = self.parity_mapping_time
        return cat_ideal.finite_duration_parity_mapping(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=angle,
            duration=duration,
            K_s=self.fsweep["K_s"],
            chi_prime=self.fsweep["chi_prime_sa"],
            n_ref=self.n_bar_ref,
            qubit_state=1,
        )

    def _built_attributes(
        self,
        attribute_names: Dict[str, str],
    ) -> Dict[str, Any]:
        r"""
        Dictionary of key to ``getattr(self, name)`` for the pairs
        ``key: name`` of ``attribute_names`` whose attribute exists; the key
        of a step that has not been built is absent.
        """
        return {
            key: getattr(self, name)
            for key, name in attribute_names.items() if hasattr(self, name)
        }

    @property
    def step_maps(self) -> Dict[str, qt.Qobj | List[qt.Qobj]]:
        r"""
        Dynamical maps of the steps of one QEC cycle, in the current frame.

        Returns
        -------
        Dict[str, qt.Qobj | List[qt.Qobj]]
            Superoperators by step; a step that has not been built is absent.

            - ``"idle"``: the idling.
            - ``"qubit_gate_p"``, ``"qubit_gate_m"``: the :math:`+\pi/2` and
              :math:`-\pi/2` pulses, with the frame changes of the protocol.
              Gate 1 is the :math:`+\pi/2` pulse. Gate 2 is the :math:`-\pi/2`
              pulse if the measurement record holds an even number of
              outcomes 1 (parity flips), else the :math:`+\pi/2` pulse.
            - ``"parity_mapping"``: the parity-mapping idling, with its frame
              change.
            - ``"qubit_measurement"``: list, one map per reported outcome
              (``measurement_maps``).
            - ``"feedback_latency"``: the idling of the feedback latency.
            - ``"qubit_reset"``: the reset :math:`\pi` pulse, applied when the
              last outcome is 1.
        """
        return self._built_attributes({
            "idle": "_idling_real",
            "qubit_gate_p": "_qubit_gate_p_real",
            "qubit_gate_m": "_qubit_gate_m_real",
            "parity_mapping": "_parity_mapping_real",
            "qubit_measurement": "_qubit_projs_real",
            "feedback_latency": "_feedback_latency_real",
            "qubit_reset": "_qubit_reset_real",
        })

    @property
    def ideal_operators(self) -> Dict[str, List[qt.Qobj]]:
        r"""
        Ideal operators of the steps of one QEC cycle, in the current frame.

        Returns
        -------
        Dict[str, List[qt.Qobj]]
            Lists of operators by step, with the keys of ``step_maps``; a step
            that has not been built is absent. ``"idle"`` holds the evolution
            without a jump and the evolution with one photon loss;
            ``"qubit_measurement"`` holds one operator per outcome; every
            other list has one operator.
        """
        return self._built_attributes({
            "idle": "_idling_ideal",
            "qubit_gate_p": "_qubit_gate_p_ideal",
            "qubit_gate_m": "_qubit_gate_m_ideal",
            "parity_mapping": "_parity_mapping_ideal",
            "qubit_measurement": "_qubit_projs_ideal",
            "feedback_latency": "_feedback_latency_ideal",
            "qubit_reset": "_qubit_reset_ideal",
        })

    @property
    def lab_frame_pulses(self) -> Dict[str, qt.Qobj]:
        r"""
        Unitaries or dynamical maps of the ancilla pulses in the lab frame,
        as ``cat_real.qubit_gate`` returns them, without frame changes.

        Each entry is a unitary if ``gate_qubit_decoherence`` was False when
        the pulse was built, else a superoperator.
        ``gate_map_in_current_frame`` transforms an entry to the current
        frame.

        Returns
        -------
        Dict[str, qt.Qobj]
            ``"qubit_gate_p"`` (:math:`+\pi/2` pulse), ``"qubit_gate_m"``
            (:math:`-\pi/2` pulse), and ``"qubit_reset"`` (reset :math:`\pi` pulse).
            A pulse that has not been built is absent.
        """
        return self._built_attributes({
            "qubit_gate_p": "_gate_p_lab_real",
            "qubit_gate_m": "_gate_m_lab_real",
            "qubit_reset": "_reset_lab_real",
        })

    # overall generation ###############################################
    def build_all_processes(
        self,
        num_cpus: int = 8,
    ) -> None:
        """
        Build the dynamical maps and the ideal operators of all steps of
        one QEC cycle.

        Parameters
        ----------
        num_cpus : int
            Number of worker processes that integrate the ancilla pulses.
        """
        
        builds = [
            self.build_idling_process,
            self.build_qubit_gate_process,
            self.build_parity_mapping_process,
            self.build_qubit_measurement_process,
            self.build_feedback_latency_process,
            self.build_qubit_reset_process,
        ]

        for build in tqdm(builds, disable=global_settings.PROGRESSBAR_DISABLED):
            try:
                build(num_cpus=num_cpus)
            except TypeError as e:
                if "num_cpus" in str(e):
                    # this step takes no num_cpus argument
                    build()
                else:
                    raise e

    def generate_tree(
        self,
        init_prob_amp_01: Tuple[float, float],
        logical_0: qt.Qobj,
        logical_1: qt.Qobj,
        QEC_rounds: int = 1,
        with_check_point: bool = False,
    ) -> EvolutionTree:
        r"""
        Tree of a number of QEC cycles that starts from an encoded state.
        The steps must have been built (``build_all_processes``); the
        states are evolved afterwards by ``EvolutionTree.traverse``.

        Parameters
        ----------
        init_prob_amp_01 : Tuple[float, float]
            Amplitudes :math:`(c_0, c_1)` of the encoded state
            :math:`c_0|0_L\rangle+c_1|1_L\rangle`.
        logical_0, logical_1 : qt.Qobj
            Kets of the logical states :math:`|0_L\rangle` and
            :math:`|1_L\rangle` on the resonator-ancilla space.
        QEC_rounds : int
            Number of QEC cycles.
        with_check_point : bool
            If True, a check point follows every step.

        Returns
        -------
        EvolutionTree
            The tree; its edges are named ``"<prefix> (<cycle>.<step>)"``,
            with the prefixes of ``EDGE_PREFIXES``, the index of the cycle,
            and the running index of the step.
        """
        edge_method_names = self._step_method_sequence(with_check_point)

        graph = EvolutionTree()

        init_state_node = StateNode.initial_note(
            init_prob_amp_01, logical_0, logical_1,
        )
        graph.add_node(init_state_node)

        # each step adds its edges to the nodes of the previous step
        current_ensemble = StateEnsemble([init_state_node])
        step_counter = 0
        for round in range(QEC_rounds):
            for method_name in edge_method_names:
                current_ensemble = self._step_method(method_name)(
                    f"{round}.{step_counter}", graph, current_ensemble,
                )
                step_counter += 1

        return graph

    def generate_wo_QEC(
        self,
        init_prob_amp_01: Tuple[float, float],
        logical_states: np.ndarray[qt.Qobj],
        QEC_rounds: int = 1,
        with_check_point: bool = False,
    ) -> EvolutionTree:
        r"""
        Tree of an evolution without QEC: each cycle is the idling step
        alone, and a photon loss is not a correctable error.

        Parameters
        ----------
        init_prob_amp_01 : Tuple[float, float]
            Amplitudes :math:`(c_0, c_1)` of the encoded state
            :math:`c_0|0_L\rangle+c_1|1_L\rangle`.
        logical_states : np.ndarray[qt.Qobj]
            Array of shape ``(k, 2)``; each row is a pair of logical states
            that the initial node holds as one error space. The initial
            state is encoded with the first row.
        QEC_rounds : int
            Number of idling steps.
        with_check_point : bool
            If True, a check point follows every step.

        Returns
        -------
        EvolutionTree
            The tree.
        """
        graph = EvolutionTree()

        logical_0, logical_1 = logical_states[0]
        init_state_node = StateNode.initial_note(
            init_prob_amp_01, logical_0, logical_1,
        )
        init_state_node._raw_ideal_logical_states = logical_states
        graph.add_node(init_state_node)

        current_ensemble = StateEnsemble([init_state_node])

        step_counter = 0
        for round in range(QEC_rounds):
            current_ensemble = self.idle(
                f"{round}.{step_counter}", graph, current_ensemble,
                correctable_single_photon_loss=False,
            )
            step_counter += 1

            if with_check_point:
                current_ensemble = self.check_point(
                    f"{round}.{step_counter}", graph, current_ensemble,
                )
                step_counter += 1

        return graph
