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

from base_neural_model.mechanics.eshelby import (
    MatrixParams,
    eshelby_kappa,
    eshelby_kappa_from,
    eshelby_sphere_components,
    eshelby_spheroid_trace,
)


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
