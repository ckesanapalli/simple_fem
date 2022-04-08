# Test cases
# 1. Read the abaqus .mtx files and convert to numpy array
# 2. Create a element and node files and read and compare
# 3. Create a global matrix

# import logging
from pathlib import Path

import numpy as np

import truss_modal.truss as trm


def test_truss_global():
    abaqus_files = Path("abaqus_results")
    elem_db_files = abaqus_files.glob("*_elem_db.csv")
    node_db_files = abaqus_files.glob("*_node_db.csv")

    for elem_db_file, node_db_file in zip(elem_db_files, node_db_files):
        prefix = elem_db_file.stem[:-8]
        space_frame = trm.Truss.from_file(node_db_file, elem_db_file)

        truss_global = space_frame.global_mats()

        # Verification of global matrices
        abaqus_dir = Path("abaqus_results")

        abaqus_mass = trm.abaqus2np(abaqus_dir / f"{prefix}_MASS1.mtx")
        abaqus_stiff = trm.abaqus2np(abaqus_dir / f"{prefix}_STIF1.mtx")
        abaqus_damp = trm.abaqus2np(abaqus_dir / f"{prefix}_DMPV1.mtx")

        rtol = 1e-14
        atol = 1e-6
        assert np.allclose(truss_global.mass, abaqus_mass, rtol=rtol, atol=atol)
        assert np.allclose(truss_global.stiffness, abaqus_stiff, rtol=rtol, atol=atol)
        assert np.allclose(truss_global.damping, abaqus_damp, rtol=rtol, atol=atol)

        # Verification of eigen analysis
        mode_range = slice(6, 30)
        abaqus_freq = np.genfromtxt(abaqus_dir / f"{prefix}_freq.csv")[mode_range]
        mode_data = trm.modal_analysis(
            truss_global.stiffness,
            truss_global.mass,
            mode_range=mode_range,
            damp_mat=truss_global.damping,
            get_data=True,
        )

        assert np.allclose(mode_data.frequency, abaqus_freq, rtol=1e-3, atol=1e-3)
