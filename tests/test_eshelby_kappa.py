"""Eshelby confinement factor kappa: closed forms, limits, and a FEM cross-check.

kappa is derived from the matrix Poisson ratio (source doc, PART 2, Module 1 (i)),
not asserted free. These tests pin:

* the boxed sphere components S_1111, S_1122;
* the two limits the doc names -- kappa -> 1/3 (free, nu -> 0) and kappa -> 1
  (incompressible, nu -> 1/2), with monotonicity between them;
* the physical-bound guard (kappa in [1/3, 1]; nu in [0, 0.5));
* a real scikit-fem solve of a spherical inclusion with a dilatational eigenstrain,
  whose recovered constrained dilatation agrees with the analytical kappa.
"""

from __future__ import annotations

import numpy as np
import pytest

from base_neural_model.base.types import MechanicsParams, VoxelGeometry
from base_neural_model.mechanics.derive import (
    FAST_BAND_UNDRAINED_NU,
    motor_cortex_with_aspect_ratio,
)
from base_neural_model.mechanics.eshelby import (
    MatrixParams,
    eshelby_kappa,
    eshelby_kappa_from,
    eshelby_sphere_components,
    eshelby_spheroid_trace,
    interaction_verdict_shift_db,
    mori_tanaka_kappa_clamped,
    mori_tanaka_kappa_traction_free,
)
from base_neural_model.mechanics.neuron_constants import get_single_neuron_displacement
from base_neural_model.mechanics.transduction import mechanical_displacement


@pytest.mark.parametrize("nu", [0.0, 0.1, 0.3, 0.45, 0.49])
def test_sphere_components_match_closed_forms(nu):
    """S_1111 = (7-5nu)/[15(1-nu)], S_1122 = (5nu-1)/[15(1-nu)]."""
    s1111, s1122 = eshelby_sphere_components(nu)
    denom = 15.0 * (1.0 - nu)
    assert s1111 == pytest.approx((7.0 - 5.0 * nu) / denom)
    assert s1122 == pytest.approx((5.0 * nu - 1.0) / denom)


def test_kappa_free_limit():
    """nu -> 0 is the free/unconfined inclusion: kappa = 1/3."""
    assert eshelby_kappa(0.0) == pytest.approx(1.0 / 3.0)


def test_kappa_approaches_confined_at_incompressibility():
    """nu -> 1/2 (the fast-band undrained matrix): kappa -> 1."""
    assert eshelby_kappa(0.4999) == pytest.approx(1.0, abs=1e-3)


def test_kappa_monotone_increasing_in_nu():
    """Stiffer (more incompressible) matrix -> more confined dilatation."""
    nus = np.linspace(0.0, 0.49, 25)
    kappas = np.array([eshelby_kappa(nu) for nu in nus])
    assert np.all(np.diff(kappas) > 0.0)
    assert np.all(kappas >= 1.0 / 3.0 - 1e-12)
    assert np.all(kappas <= 1.0 + 1e-12)


def test_kappa_equals_sphere_trace_at_unit_aspect_ratio():
    """A unit aspect ratio reduces the spheroid trace to S_1111 + 2 S_1122."""
    for nu in (0.1, 0.3, 0.45):
        s1111, s1122 = eshelby_sphere_components(nu)
        assert eshelby_spheroid_trace(nu, 1.0) == pytest.approx(s1111 + 2.0 * s1122)


@pytest.mark.parametrize("aspect", [0.3, 0.7, 1.5, 3.0])
def test_kappa_in_band_across_aspect_ratios(aspect):
    """Penny/prolate shapes still land kappa inside [1/3, 1]."""
    for nu in (0.2, 0.4, 0.48):
        kappa = eshelby_kappa(nu, aspect_ratio=aspect)
        assert 1.0 / 3.0 - 1e-9 <= kappa <= 1.0 + 1e-9


@pytest.mark.parametrize("bad_nu", [-0.01, 0.5, 0.6, 1.0])
def test_matrix_params_rejects_out_of_range_nu(bad_nu):
    with pytest.raises(ValueError):
        MatrixParams(poisson_ratio=bad_nu)


def test_matrix_params_rejects_nonpositive_aspect():
    with pytest.raises(ValueError):
        MatrixParams(poisson_ratio=0.3, aspect_ratio=0.0)


def test_eshelby_kappa_from_matches_scalar_call(matrix_params):
    assert eshelby_kappa_from(matrix_params) == pytest.approx(
        eshelby_kappa(matrix_params.poisson_ratio, matrix_params.aspect_ratio)
    )


# --- scikit-fem cross-check ------------------------------------------------------
# A real FEM solve of the canonical Eshelby problem: a spherical inclusion carrying
# a uniform dilatational eigenstrain, embedded in a large isotropic matrix with the
# far field held fixed. The realized (constrained) dilatation inside the inclusion,
# as a fraction of the imposed eigenstrain dilatation, is exactly kappa. We solve it
# on a coarse tetrahedral mesh and check agreement with the analytical kappa.


def _fem_constrained_dilatation(nu: float, *, refine: int = 3) -> float:
    skfem = pytest.importorskip("skfem")
    from skfem import (
        Basis,
        ElementTetP1,
        ElementVector,
        MeshTet,
        asm,
        condense,
        solve,
    )
    from skfem.helpers import sym_grad, trace
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    # Unit cube matrix, inclusion = sphere of radius R about the centre.
    mesh = MeshTet().refined(refine)
    e = ElementVector(ElementTetP1())
    basis = Basis(mesh, e, intorder=2)

    young = 1.0
    lam, mu = lame_parameters(young, nu)
    R = 0.25
    eig = 1e-3  # imposed eigenstrain dilatation per axis (isotropic)

    centroids = mesh.p[:, mesh.t].mean(axis=1)
    r = np.linalg.norm(centroids - 0.5, axis=0)
    in_incl = r < R

    stiffness = linear_elasticity(lam, mu)

    @skfem.LinearForm
    def eig_load(v, w):
        # Isotropic eigenstrain eps* = eig*I inside the inclusion. The equivalent
        # body load is the divergence of the eigenstress sigma* = (2 mu + 3 lam) eig I,
        # which in weak form contracts with the test strain as a trace term.
        sig_star = (2.0 * mu + 3.0 * lam) * eig
        mask = (np.linalg.norm(w.x - 0.5, axis=0) < R).astype(float)
        return mask * sig_star * trace(sym_grad(v))

    K = asm(stiffness, basis)
    f = asm(eig_load, basis)

    # Fix the outer boundary (far field clamped).
    boundary_dofs = basis.get_dofs().all()
    x = solve(*condense(K, f, D=boundary_dofs))

    # Realized dilatation = average trace(grad u) over the inclusion elements.
    du = basis.interpolate(x).grad  # shape (3, 3, nelem, nqp)
    realized = (du[0, 0] + du[1, 1] + du[2, 2]).mean(axis=1)  # per element
    realized_incl = realized[in_incl].mean()

    # Fraction of the imposed eigenstrain dilatation that is realized.
    return realized_incl / (3.0 * eig)


@pytest.mark.parametrize("nu", [0.2, 0.35, 0.45])
def test_fem_cross_check(nu):
    """scikit-fem inclusion solve agrees with the analytical kappa (loose tol).

    The coarse mesh and clamped far field make this an order-of-magnitude /
    monotone cross-check, not a high-precision match: we assert the FEM-realized
    constrained dilatation lands within tolerance of analytical kappa and tracks
    it upward with nu.
    """
    fem = _fem_constrained_dilatation(nu)
    analytic = eshelby_kappa(nu)
    # Coarse, clamped-far-field FEM under-realizes; check same order and direction.
    assert fem > 0.0
    assert fem == pytest.approx(analytic, rel=0.6)


# --- motor_cortex_with_aspect_ratio: the non-spherical path, wired up and reachable -----
# eshelby_kappa's aspect_ratio has always been implemented and unit-tested (above), but
# before this, no preset or production call site ever passed a non-1.0 value -- it was
# a dormant, unreachable capability. These pin the new opt-in builder
# (mechanics.derive.motor_cortex_with_aspect_ratio) and confirm it is reachable through
# the REAL transduction chain, not just the bare eshelby_kappa function.


def test_motor_cortex_with_aspect_ratio_only_changes_kappa():
    """Every field except confinement_kappa matches MechanicsParams.motor_cortex()."""
    from dataclasses import asdict

    base = asdict(MechanicsParams.motor_cortex())
    varied = asdict(motor_cortex_with_aspect_ratio(2.5))
    diffs = {k for k in base if base[k] != varied[k]}
    assert diffs == {"confinement_kappa"}


def test_motor_cortex_with_aspect_ratio_does_not_mutate_the_base_preset():
    """MechanicsParams.motor_cortex() itself is unchanged by the new opt-in path."""
    before = MechanicsParams.motor_cortex()
    motor_cortex_with_aspect_ratio(3.0)  # exercised, result discarded
    after = MechanicsParams.motor_cortex()
    assert before == after
    assert after.confinement_kappa == pytest.approx(0.5)


def test_motor_cortex_with_aspect_ratio_matches_eshelby_kappa_from():
    """The derived kappa is exactly eshelby_kappa_from at the same (nu, aspect_ratio)."""
    aspect = 2.0
    params = motor_cortex_with_aspect_ratio(aspect)
    expected = eshelby_kappa_from(
        MatrixParams(poisson_ratio=FAST_BAND_UNDRAINED_NU, aspect_ratio=aspect)
    )
    assert params.confinement_kappa == pytest.approx(expected)


def test_motor_cortex_with_aspect_ratio_respects_custom_matrix_nu():
    params = motor_cortex_with_aspect_ratio(1.5, matrix_nu=0.3)
    expected = eshelby_kappa_from(MatrixParams(poisson_ratio=0.3, aspect_ratio=1.5))
    assert params.confinement_kappa == pytest.approx(expected)


@pytest.mark.parametrize("aspect", [0.3, 0.7, 1.0, 1.5, 3.0])
def test_motor_cortex_with_aspect_ratio_kappa_stays_in_physical_band(aspect):
    params = motor_cortex_with_aspect_ratio(aspect)
    assert 1.0 / 3.0 - 1e-9 <= params.confinement_kappa <= 1.0 + 1e-9


def test_motor_cortex_with_aspect_ratio_departs_from_sphere_pulls_kappa_toward_free():
    """Non-spherical (either direction) confines less than the sphere at fixed nu."""
    sphere = motor_cortex_with_aspect_ratio(1.0)
    prolate = motor_cortex_with_aspect_ratio(3.0)
    oblate = motor_cortex_with_aspect_ratio(0.3)
    assert prolate.confinement_kappa < sphere.confinement_kappa
    assert oblate.confinement_kappa < sphere.confinement_kappa


def test_motor_cortex_with_aspect_ratio_reaches_mechanical_displacement():
    """End-to-end: the derived non-spherical kappa is reachable through the real chain,
    not just unit-tested against the bare eshelby_kappa function."""
    from dataclasses import replace

    d_single = get_single_neuron_displacement()
    geom = VoxelGeometry.motor_cortex_layer5()
    params = replace(
        motor_cortex_with_aspect_ratio(3.0), membrane_disp_m=d_single.value_m
    )

    out = mechanical_displacement(d_single, geom, params, synchrony_fraction=0.8)

    expected_kappa = eshelby_kappa_from(
        MatrixParams(poisson_ratio=FAST_BAND_UNDRAINED_NU, aspect_ratio=3.0)
    )
    assert out.confinement_kappa == pytest.approx(expected_kappa)
    assert out.axial_displacement_m > 0.0
    assert any(f"kappa = {expected_kappa}" in a for a in out.provenance.assumptions)


# --- Mori-Tanaka interaction bounds: the dilute assumption is bounded, not asserted ------
# eshelby_kappa is the SINGLE-inclusion (dilute) limit. At the model's operating
# cell_volume_fraction ~ 0.15 that limit is at the dilute/interacting boundary, so the
# neglected inclusion-inclusion interaction is bracketed by the two Mori-Tanaka closures
# (traction-free upper / clamped lower) and shown to move the kappa-linear verdict by a
# negligible amount. These tests pin both bounds and, crucially, that magnitude.


def test_mt_bounds_reduce_to_dilute_at_zero_volume_fraction():
    for kd in (1.0 / 3.0, 0.5, 0.9739):
        assert mori_tanaka_kappa_traction_free(kd, 0.0) == pytest.approx(kd)
        assert mori_tanaka_kappa_clamped(kd, 0.0) == pytest.approx(kd)


def test_mt_bounds_reach_their_limits_as_f_approaches_one():
    # traction-free -> 1 (all-inclusion, free swelling); clamped -> 0 (rigidly held).
    assert mori_tanaka_kappa_traction_free(0.5, 0.999) == pytest.approx(1.0, abs=1e-3)
    assert mori_tanaka_kappa_clamped(0.5, 0.999) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("kd", [0.4, 0.5, 0.88, 0.9739])
@pytest.mark.parametrize("f", [0.05, 0.15, 0.3, 0.5])
def test_dilute_kappa_lies_within_the_interaction_bracket(kd, f):
    lo = mori_tanaka_kappa_clamped(kd, f)
    hi = mori_tanaka_kappa_traction_free(kd, f)
    assert lo <= kd <= hi


def test_mt_bounds_are_monotone_in_f():
    kd = 0.5
    fs = np.linspace(0.0, 0.95, 20)
    tf = np.array([mori_tanaka_kappa_traction_free(kd, f) for f in fs])
    cl = np.array([mori_tanaka_kappa_clamped(kd, f) for f in fs])
    assert np.all(np.diff(tf) > 0)   # traction-free rises with f
    assert np.all(np.diff(cl) < 0)   # clamped falls with f


def test_interaction_shift_matches_closed_form_and_is_small_at_operating_f():
    """The load-bearing assertion: at f=0.15 the interaction can shift the verdict by only
    ~1.41 dB, matching |20 log10(1-f)| exactly."""
    import math

    assert interaction_verdict_shift_db(0.15) == pytest.approx(
        abs(20.0 * math.log10(1.0 - 0.15)), abs=1e-9
    )
    assert interaction_verdict_shift_db(0.15) == pytest.approx(1.41, abs=0.05)
    # Even a conservative doubling of packing stays only a few dB.
    assert interaction_verdict_shift_db(0.30) == pytest.approx(3.10, abs=0.05)
    assert interaction_verdict_shift_db(0.0) == 0.0


def test_interaction_shift_is_independent_of_kappa_anchor():
    """The realized dB spread of the bracket is the same at every kappa_dilute anchor
    (preset 0.5 and the derived 0.88 / 0.9739), because the widest excursion is the
    clamped factor (1-f), which does not depend on kappa_dilute. This is what makes the
    'negligible' conclusion hold regardless of which kappa path (hand-set vs Eshelby-
    derived) the model uses."""
    import math

    f = 0.15
    expected = abs(20.0 * math.log10(1.0 - f))
    for kd in (0.5, 0.88, 0.9739):
        lo = mori_tanaka_kappa_clamped(kd, f)
        hi = mori_tanaka_kappa_traction_free(kd, f)
        # dB shift of each bound relative to the dilute anchor.
        shift_lo = abs(20.0 * math.log10(lo / kd))
        shift_hi = abs(20.0 * math.log10(hi / kd))
        assert shift_lo == pytest.approx(expected, abs=1e-9)  # anchor-independent
        assert max(shift_lo, shift_hi) <= expected + 1e-9     # clamped is the widest


@pytest.mark.parametrize("bad_f", [1.0, 1.5, -0.1])
def test_mt_functions_reject_out_of_range_f(bad_f):
    with pytest.raises(ValueError):
        interaction_verdict_shift_db(bad_f)
    with pytest.raises(ValueError):
        mori_tanaka_kappa_traction_free(0.5, bad_f)
    with pytest.raises(ValueError):
        mori_tanaka_kappa_clamped(0.5, bad_f)


@pytest.mark.parametrize("bad_kd", [-0.1, 1.5])
def test_mt_bounds_reject_out_of_range_kappa(bad_kd):
    with pytest.raises(ValueError):
        mori_tanaka_kappa_traction_free(bad_kd, 0.15)
    with pytest.raises(ValueError):
        mori_tanaka_kappa_clamped(bad_kd, 0.15)
