"""Computes the Truss global matrices and modal analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sLA
from attrs import field, frozen, make_class, validators

__all__ = ["abaqus2np", "modal_analysis", "Truss"]


def abaqus2np(abaqus_mtx_path: Path, debug: bool = True):
    """
    Converts the Abaqus .MTX file to 2D numpy.nd arrays

    Parameters
    ----------
    abaqus_mtx_path : str
        path of Abaqus .mtx file.
    debug : bool, optional
        Prints node data if debug is True. The default is True.
    Returns
    -------
    matrix : numpy.ndarray
        2D matrix of the .mtx file.
    """
    dtypes = [int, int, int, int, float]
    dtype_dict = dict(zip(range(5), dtypes))
    mtx_data = pd.read_csv(
        abaqus_mtx_path, sep=",", header=None, dtype=dtype_dict
    ).to_numpy(dtype=object)
    dof_per_node_count = mtx_data[:, [1, 3]].max()
    node_count = mtx_data[:, [0, 2]].max()
    total_dof_count = node_count * dof_per_node_count

    rows = (dof_per_node_count * (mtx_data[:, 0] - 1) + (mtx_data[:, 1] - 1)).astype(
        int
    )
    cols = (dof_per_node_count * (mtx_data[:, 2] - 1) + (mtx_data[:, 3] - 1)).astype(
        int
    )
    matrix = np.zeros([total_dof_count, total_dof_count], dtype=float)

    matrix[rows, cols] = mtx_data[:, 4]
    matrix[cols, rows] = matrix[rows, cols]
    return matrix


def modal_analysis(
    stiff_mat: np.ndarray,
    mass_mat: np.ndarray,
    mode_range: slice = slice(6, 20),
    damp_mat: np.ndarray = None,
    save_prefix: str = None,
    get_data: bool = False,
):
    """
    Perform modal analysis for given mass and stiffness matrix.

    Parameters
    ----------
    stiff_mat : numpy.ndarray
        Stiffness Matrix.
    mass_mat : numpy.ndarray
        Mass Matrix.

    save_prefix : str, optional
        provide prefix of the data files to output mode data. The default is None.

    Returns
    -------
    mode_freq : numpy.ndarray
        Mode frequencies.
    mode_shapes : numpy.ndarray
        Mode Shapes. Here mode shapes are normalized with it's maximum value.
    modal_stiff : numpy.ndarray
        Modal Stiffness.
    modal_mass : numpy.ndarray
        Modal Mass.

    """
    eigen_values, eigen_vectors = sLA.eigh(stiff_mat, b=mass_mat)
    eigen_values = eigen_values[mode_range]
    eigen_vectors = eigen_vectors[:, mode_range]

    mode_freq = np.sqrt(np.abs(eigen_values))
    mode_shapes = eigen_vectors / np.max(np.abs(eigen_vectors[2::3, :]), axis=0)

    modal_stiff = np.abs(mode_shapes.conj().T @ stiff_mat @ mode_shapes).diagonal()
    modal_mass = np.abs(mode_shapes.conj().T @ mass_mat @ mode_shapes).diagonal()

    if damp_mat is not None:
        modal_damp_2d = np.abs(mode_shapes.conj().T @ damp_mat @ mode_shapes)
        modal_damp = modal_damp_2d.diagonal()

    if save_prefix is not None:
        np.savetxt(f"{save_prefix}_normalized_mass.csv", modal_mass, delimiter=",")
        np.savetxt(
            f"{save_prefix}_normalized_stiffness.csv", modal_stiff, delimiter=","
        )
        np.savetxt(f"{save_prefix}_modal_frequencies.csv", mode_freq, delimiter=",")
        np.savetxt(f"{save_prefix}_mode_shapes.csv", mode_shapes, delimiter=",")

        if damp_mat is not None:
            np.savetxt(
                f"{save_prefix}_normalized_damping.csv", modal_damp, delimiter=","
            )
            np.savetxt(
                f"{save_prefix}_normalized_damping_2d.csv", modal_damp_2d, delimiter=","
            )

    if get_data:
        if damp_mat is None:
            ModeData = make_class(
                "ModeData", ["frequency", "shapes", "stiffness", "mass"], frozen=True
            )
            return ModeData(mode_freq, mode_shapes, modal_stiff, modal_mass)
        else:
            ModeData = make_class(
                "ModeData",
                ["frequency", "shapes", "stiffness", "mass", "damping"],
                frozen=True,
            )
            return ModeData(mode_freq, mode_shapes, modal_stiff, modal_mass, modal_damp)


@frozen
class Truss:
    """
    Class containing node and element data.

    Attributes
    ----------
    node_db : pandas.DataFrame
        DataFrame containing node coordinates and point interia (optional).
    elem_db : pandas.DataFrame
        DataFrame containing element node indices, density, modulus and
        damping (optional).

    Methods
    -------
    from_file(node_db_file, elem_db_file)
        Read node and element files and created a Truss object.
    assemble_mats(nodes_size, elem_nodes_ids, trans_mats)
        Assemble the element matrices to global matrices.
    global_mats(mass_type={"full", "lump"})
        Returns all global matrices of the Truss.
    modal_analysis(mode_range=slice(0,30), save_prefix=None, get_data=True)
        Returns the modal data.

    """

    node_db = field(validator=validators.instance_of(pd.DataFrame))
    elem_db = field(validator=validators.instance_of(pd.DataFrame))
    trans_mats = field(init=False)
    disp_count = field(init=False)
    elem_nodes_ids = field(init=False)

    _dtypes = {
        "node_1": int,
        "node_2": int,
        "x": float,
        "y": float,
        "z": float,
        "dashpot": float,
        "damping": float,
        "modulus": float,
        "cs_area": float,
    }

    def __attrs_post_init__(self):
        """Update elem_db with truss bar vectors and assigns unit_vector matrix."""
        self.elem_db.loc[:, ["lx", "ly", "lz"]] = (
            self.node_db.loc[self.elem_db.node_2, ["x", "y", "z"]].to_numpy()
            - self.node_db.loc[self.elem_db.node_1, ["x", "y", "z"]].to_numpy()
        )

        self.elem_db.loc[:, ["length"]] = np.linalg.norm(
            self.elem_db.loc[:, ["lx", "ly", "lz"]], axis=1
        )

        self.elem_db.loc[:, ["nx", "ny", "nz"]] = (
            self.elem_db.loc[:, ["lx", "ly", "lz"]].to_numpy()
            / self.elem_db.loc[:, ["length"]].to_numpy()
        )

        trans_mats = np.einsum(
            "ij,ik->ijk",
            self.elem_db.loc[:, ["nx", "ny", "nz"]],
            self.elem_db.loc[:, ["nx", "ny", "nz"]],
        )

        disp_count = self.node_db.shape[0] * 3
        elem_nodes_ids = self.elem_db.loc[:, ["node_1", "node_2"]].to_numpy()

        object.__setattr__(self, "trans_mats", trans_mats)
        object.__setattr__(self, "disp_count", disp_count)
        object.__setattr__(self, "elem_nodes_ids", elem_nodes_ids)

    @classmethod
    def from_file(cls, node_db_file: Path, elem_db_file: Path):
        """
        Read the node and element data from the files.

        Parameters
        ----------
        node_db_file : str or pathlib.Path
            Path to the node data file.
        elem_db_file : str or pathlib.Path
            Path to the element data file.

        Returns
        -------
        Truss

        """
        node_db = pd.read_csv(node_db_file, sep=",", dtype=cls._dtypes)
        elem_db = pd.read_csv(elem_db_file, sep=",", dtype=cls._dtypes)
        return cls(node_db, elem_db)

    def _old_assembler(self, trans_mats: np.ndarray, elem_mats: np.ndarray):
        """
        Return Assembler function for the local element matrices.

        Parameters
        ----------
        trans_mats : np.ndarray
            Unit vector of the truss bars.
        elem_mats : np.ndarray
            Local matrix of all elements.

        Returns
        -------
        np.ndarray
            Global matrix of all elements.

        """
        global_mat = np.zeros([self.disp_count, self.disp_count])
        trans_elem_mats = [np.einsum("nij,nlm->nijlm", elem_mats, trans_mats)[0]]

        for elem_nodes_idx, trans_elem_mat in zip(
            [self.elem_nodes_ids[0]], trans_elem_mats
        ):
            start_idx, end_idx = elem_nodes_idx
            start_slice = slice(3 * start_idx, 3 * start_idx + 3)
            end_slice = slice(3 * end_idx, 3 * end_idx + 3)

            global_mat[start_slice, start_slice] += trans_elem_mat[0, 0]
            global_mat[start_slice, end_slice] += trans_elem_mat[0, 1]
            global_mat[end_slice, start_slice] += trans_elem_mat[1, 0]
            global_mat[end_slice, end_slice] += trans_elem_mat[1, 1]
        return global_mat

    def assembler(self, trans_mats: np.ndarray, elem_mats: np.ndarray):
        """
        Return Assembler function for the local element matrices.

        Parameters
        ----------
        trans_mats : np.ndarray
            Unit vector of the truss bars.
        elem_mats : np.ndarray
            Local matrix of all elements.

        Returns
        -------
        np.ndarray
            Global matrix of all elements.

        """
        global_mat = np.zeros([self.disp_count, self.disp_count])
        trans_elem_mats = np.einsum("nij,nlm->nijlm", elem_mats, trans_mats)

        start_slices = 3 * np.array(3 * [3 * [self.elem_nodes_ids[:, 0]]]).T
        start_slices[:, :, 1] += 1
        start_slices[:, :, 2] += 2

        end_slices = 3 * np.array(3 * [3 * [self.elem_nodes_ids[:, 1]]]).T
        end_slices[:, :, 1] += 1
        end_slices[:, :, 2] += 2

        np.add.at(
            global_mat,
            (start_slices, start_slices.transpose(0, 2, 1)),
            trans_elem_mats[:, 0, 0],
        )
        np.add.at(
            global_mat,
            (start_slices, end_slices.transpose(0, 2, 1)),
            trans_elem_mats[:, 0, 1],
        )
        np.add.at(
            global_mat,
            (end_slices, start_slices.transpose(0, 2, 1)),
            trans_elem_mats[:, 1, 0],
        )
        np.add.at(
            global_mat,
            (end_slices, end_slices.transpose(0, 2, 1)),
            trans_elem_mats[:, 1, 1],
        )

        return global_mat

    def global_mats(self, mass_type: str = "lump"):
        """
        Return global matrices.

        Parameters
        ----------
        mass_type : {'full', 'lump'}, optional
            Type of the global matrices. The default is "lump".

        Raises
        ------
        ValueError
            When mass_type is not in {'full', 'lump'}.

        Returns
        -------
        TrussGlobal
            Object containing global matrices of the Truss.

        """
        elem_mass_mats = np.einsum(
            "jk,i->ijk",
            [[2, 1], [1, 2]],
            self.elem_db.cs_area * self.elem_db.density * self.elem_db.length / 6,
        )

        elem_lump_mats = np.einsum(
            "jk,i->ijk",
            [[1, 0], [0, 1]],
            self.elem_db.cs_area * self.elem_db.density * self.elem_db.length / 2,
        )

        elem_stiff_mats = np.einsum(
            "jk,i->ijk",
            [[1, -1], [-1, 1]],
            self.elem_db.cs_area * self.elem_db.modulus / self.elem_db.length,
        )

        global_stiff_mat = self.assembler(self.trans_mats, elem_stiff_mats)

        if mass_type == "full":
            global_mass_mat = self.assembler(self.trans_mats, elem_mass_mats)
        elif mass_type == "lump":
            eye_mats = np.array([np.eye(3)] * self.elem_db.shape[0])
            global_mass_mat = self.assembler(eye_mats, elem_lump_mats)
        else:
            raise ValueError("mass_type = {mass_type} is not in {'full', 'lump'}")

        # Adding point interia at nodes to global mass
        if "point_interia" in self.node_db:
            node_ids = self.node_db.index
            global_mass_mat[
                3 * node_ids, 3 * node_ids
            ] += self.node_db.point_interia.to_numpy()
            global_mass_mat[
                3 * node_ids + 1, 3 * node_ids + 1
            ] += self.node_db.point_interia.to_numpy()
            global_mass_mat[
                3 * node_ids + 2, 3 * node_ids + 2
            ] += self.node_db.point_interia.to_numpy()

        if "dashpot" in self.elem_db:
            elem_damp_mats = np.einsum(
                "jk,i->ijk", [[1, -1], [-1, 1]], self.elem_db.dashpot
            )
            global_damp_mat = self.assembler(self.trans_mats, elem_damp_mats)

            TrussGlobal = make_class(
                "TrussGlobal", ["mass", "stiffness", "damping"], frozen=True
            )
            return TrussGlobal(global_mass_mat, global_stiff_mat, global_damp_mat)
        else:
            TrussGlobal = make_class("TrussGlobal", ["mass", "stiffness"], frozen=True)
            return TrussGlobal(global_mass_mat, global_stiff_mat)
