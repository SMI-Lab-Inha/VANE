# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Modal analysis — eigensolver, MAC, participation factors, mode identification."""

from vane.modal.averaging import (
    AzimuthConsistency,
    AzimuthStates,
    HarmonicModel,
    azimuth_consistency,
    azimuth_tangent,
    fit_harmonics,
    interval_weights,
    locate_azimuth_state,
    pin_azimuth,
    weight_kernel,
)
from vane.modal.averaging_error_estimate import (
    AveragingErrorEstimate,
    AzimuthCountTable,
    BranchErrorEstimate,
    alias_free_azimuth_count,
    azimuth_count_table,
    commensurate_time_step,
    estimate_averaging_error,
)
from vane.modal.biorthogonal import (
    BiorthogonalParticipation,
    biorthogonal_participation,
    classify_state_subsystem,
)
from vane.modal.calibration import (
    AzimuthConvergence,
    azimuth_density_sweep,
    recommend_azimuth_count,
)
from vane.modal.confidence_calibration import (
    ConfidenceCalibration,
    calibrate_confidence,
    confidence_threshold,
    recalibrate,
)
from vane.modal.eigensolver import ModalSolution, compute_modes, modes_from_mbc
from vane.modal.floquet import (
    HILL_UNIFORM_RTOL,
    FloquetSolution,
    averaging_error,
    floquet_modes,
    monodromy_exponents,
)
from vane.modal.identifier import (
    DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    IdentificationResult,
    ModeTrack,
    heavily_damped_modes,
    identify_modes,
    rotor_frame_damping_ratios,
)
from vane.modal.labeler import ModeLabel, label_mode, label_modes, label_solution
from vane.modal.mac import compute_mac, compute_macxp, match_modes
from vane.modal.participation import (
    ParticipationResult,
    compute_participation,
    participation_from_modes,
    participation_scale_factors,
)
from vane.modal.uncertainty import (
    AzimuthSpread,
    azimuth_spread,
    unified_mode_confidence,
)

__all__ = [
    "DEFAULT_HEAVILY_DAMPED_THRESHOLD",
    "HILL_UNIFORM_RTOL",
    "AveragingErrorEstimate",
    "AzimuthConsistency",
    "AzimuthConvergence",
    "AzimuthCountTable",
    "AzimuthSpread",
    "AzimuthStates",
    "BiorthogonalParticipation",
    "BranchErrorEstimate",
    "ConfidenceCalibration",
    "FloquetSolution",
    "HarmonicModel",
    "IdentificationResult",
    "ModalSolution",
    "ModeLabel",
    "ModeTrack",
    "ParticipationResult",
    "alias_free_azimuth_count",
    "averaging_error",
    "azimuth_consistency",
    "azimuth_count_table",
    "azimuth_density_sweep",
    "azimuth_spread",
    "azimuth_tangent",
    "biorthogonal_participation",
    "calibrate_confidence",
    "classify_state_subsystem",
    "commensurate_time_step",
    "compute_mac",
    "compute_macxp",
    "compute_modes",
    "compute_participation",
    "confidence_threshold",
    "estimate_averaging_error",
    "fit_harmonics",
    "floquet_modes",
    "heavily_damped_modes",
    "identify_modes",
    "interval_weights",
    "label_mode",
    "label_modes",
    "label_solution",
    "locate_azimuth_state",
    "match_modes",
    "modes_from_mbc",
    "monodromy_exponents",
    "participation_from_modes",
    "participation_scale_factors",
    "pin_azimuth",
    "recalibrate",
    "recommend_azimuth_count",
    "rotor_frame_damping_ratios",
    "unified_mode_confidence",
    "weight_kernel",
]
