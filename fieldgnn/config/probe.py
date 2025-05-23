import numpy as np
from typing import Literal

PROBE_ID_MAP = {"CO2": 0, "CH4": 1, "N2": 2, "H2": 3, "CO": 4, "H2O": 5}

ProbeType = Literal["CO2", "CH4", "N2", "H2", "CO", "H2O"]
SUPPORT_PROBES = list(PROBE_ID_MAP.keys())

PROBE_ID_MAP = {probe: idx for idx, probe in enumerate(SUPPORT_PROBES)}

PROBE_STR_MAP = {idx: probe for idx, probe in enumerate(SUPPORT_PROBES)}

# These are the energy of an isolate probe.
# The unit can be eV (NOT SURE), but the unit is the same as the PBE energy in qMOF dataset.
# These reuslts are calculated using MACE itself, the same as adsorption energy.
# See calc_probe_energy.py for implementation detail and configuration (bond length, etc.)
PROBE_ENERGY = {
    "CO2": -22.792044918639867,
    # "CO2": -21.954653597365485,
    # "CO2": -22.283242978465257,
    "N2": -16.414844604929698,
    "CH4": -23.766712242396007,
    "H2": -6.555498491824343,
    "CO": -14.376489673500195,
    "H2O": -14.039490859467241,  # Added H2O energy
    # "H2O": -14.12903462612029,
}

PROBE_BOND_LENGTH = {
    "CO2": {"C-O": 1.16},  # CO₂: C-O bond length in Å
    "N2": {"N-N": 1.10},  # N₂: N-N bond length in Å
    "CH4": {"C-H": 1.09},  # CH₄: C-H bond length in Å
    "H2": {"H-H": 0.74},  # H₂: H-H bond length in Å
    "CO": {"C-O": 1.13},  # CO: C-O bond length in Å
    "H2O": {"O-H": 0.96, "H-H": 1.51},  # H₂O: O-H bond length and H-H distance in Å
}

PROBE_ATOMIC_NUMBER = {
    "CO2": [6, 8, 8],  # C, O, O
    "N2": [7, 7],  # N, N
    "CH4": [6, 1, 1, 1, 1],  # C, H, H, H, H
    "H2": [1, 1],  # H, H
    "CO": [6, 8],  # C, O
    "H2O": [8, 1, 1],  # O, H, H
}


class PROBE_CLASS:
    def __init__(self):
        self.bond_lengths = PROBE_BOND_LENGTH
        self.atomic_numbers = PROBE_ATOMIC_NUMBER
        # Define offsets for each probe type
        self.offsets = {
            "CO2": self._get_co2_offsets(),
            "N2": self._get_n2_offsets(),
            "CH4": self._get_ch4_offsets(),
            "H2": self._get_h2_offsets(),
            "CO": self._get_co_offsets(),
            "H2O": self._get_h2o_offsets(),
        }
        self.support_probes = list(self.bond_lengths.keys())

    def _get_co2_offsets(self):
        bond_length = self.bond_lengths["CO2"]["C-O"]
        return np.array(
            [
                [0, 0, 0],  # Carbon
                [bond_length, 0, 0],  # Oxygen 1
                [-bond_length, 0, 0],  # Oxygen 2
            ]
        )

    def _get_n2_offsets(self):
        bond_length = self.bond_lengths["N2"]["N-N"]
        return np.array(
            [
                [bond_length / 2, 0, 0],  # Nitrogen 1
                [-bond_length / 2, 0, 0],  # Nitrogen 2
            ]
        )

    def _get_ch4_offsets(self):
        bond_length = self.bond_lengths["CH4"]["C-H"]
        a = 1.0 / np.sqrt(3.0)
        offsets = np.array([
            [ a,  a,  a],
            [ a, -a, -a],
            [-a,  a, -a],
            [-a, -a,  a]
        ])
        return np.vstack([
            [0, 0, 0],  # 碳原子
            bond_length * offsets  # 氢原子
        ])

    def _get_h2_offsets(self):
        bond_length = self.bond_lengths["H2"]["H-H"]
        return np.array(
            [
                [bond_length / 2, 0, 0],  # Hydrogen 1
                [-bond_length / 2, 0, 0],  # Hydrogen 2
            ]
        )

    def _get_co_offsets(self):
        bond_length = self.bond_lengths["CO"]["C-O"]
        return np.array(
            [
                [0, 0, 0],  # Carbon
                [bond_length, 0, 0],  # Oxygen
            ]
        )

    def _get_h2o_offsets(self):
        """Returns coordinates for water molecule in XY plane with O at origin"""
        oh_length = self.bond_lengths["H2O"]["O-H"]
        hh_length = self.bond_lengths["H2O"]["H-H"]

        # Calculate angle between O-H bonds (104.5° for water)
        theta = np.radians(104.5)

        # Position O at origin, H atoms in XY plane
        return np.array(
            [
                [0, 0, 0],  # Oxygen
                [
                    oh_length * np.sin(theta / 2),
                    oh_length * np.cos(theta / 2),
                    0,
                ],  # Hydrogen 1
                [
                    -oh_length * np.sin(theta / 2),
                    oh_length * np.cos(theta / 2),
                    0,
                ],  # Hydrogen 2
            ]
        )


probe_class = PROBE_CLASS()
PROBE_OFFSET = probe_class.offsets
