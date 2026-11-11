r"""Search ranges of the four free parameters of the cat-code optimization.

Defines the lower and upper bounds of the resonator frequency, coupling,
ancilla Josephson energy, and error-accumulation idling duration used by
``cycle_target.py``. Units: circuit energies and frequencies in GHz, durations
in ns.
"""

# free parameter -> [lower bound, upper bound]
FREE_PARAMETER_RANGES = {
    # GHz, resonator frequency
    "E_osc_GHz": [4, 10],
    # GHz, coupling between the resonator and the ancilla
    "g_GHz": [0.050, 0.300],
    # GHz, Josephson energy; the charging energy stays at its reference value
    "EJ_GHz": [15, 60],
    # ns, idle duration
    "T_W": [2e3, 50e3],
}

# what each free parameter is, in the words of the manuscript, with the unit
# of its range; stored next to the ranges in the result files
FREE_PARAMETER_MEANING = {
    "E_osc_GHz": "resonator frequency (GHz)",
    "g_GHz": "coupling between the resonator and the ancilla (GHz)",
    "EJ_GHz": "Josephson energy of the ancilla (GHz)",
    "T_W": "idle duration (ns)",
}
