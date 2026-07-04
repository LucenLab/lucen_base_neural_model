r"""Directional (deviatoric) channel: orientation coherence of the per-cell strain.

The isotropic chain (``transduction.py``) reads only the per-cell **volume change** -
a scalar monopole that has no direction, which is exactly why it survives a random
population: a pure dilatation cannot cancel. But a firing neuron does not expand as a
perfect sphere. Its membrane motion has a **preferred axis** ``u`` (e.g. a pyramidal
cell elongated along the cortical column), so the per-cell eigenstrain carries a
**deviatoric** (shape-changing, directional) part on top of the hydrostatic part::

    eps*_ij = eps_iso * delta_ij  +  beta * eps_vol * (u_i u_j - 1/3 delta_ij)

* ``eps_vol = 3 dr / r`` is the fractional volume change (the monopole magnitude);
* ``eps_iso = eps_vol / 3`` is its hydrostatic (trace) part;
* ``beta in [0, 1]`` is the **anisotropy fraction**: 0 = perfectly spherical
  expansion (pure monopole, the current model), 1 = a fully directional expansion
  along ``u``.

Temporal synchrony ``s`` (do cells fire at the *same time*) and **directional
coherence** (do their axes ``u`` *point the same way*) are independent. The beam reads
the **axial** (``z``, the beam direction) component of the population-averaged strain.
The deviatoric part's axial component for one cell is

    beta * eps_vol * ( (u . z)^2 - 1/3 ) .

Averaged over the population's orientation distribution this depends only on
``<(u . z)^2>``. For an isotropic (randomly oriented) population
``<(u . z)^2> = 1/3`` and the **deviatoric term vanishes exactly** - recovering the
current model's "the directional sum cancels". For an aligned population it survives.

We summarize the orientation distribution by a single uniaxial **orientation order
parameter** ``Q in [0, 1]`` and the director's projection on the beam axis,
``mu = cos(theta_d) = (director . z)``, via the standard nematic relation

    <(u . z)^2> = 1/3 + (2/3) * Q * P2(mu) ,   P2(x) = (3 x^2 - 1) / 2 ,

so the **axial directional survival factor** is

    g(Q, mu) = <(u . z)^2> - 1/3 = (2/3) * Q * P2(mu) ,

which is 0 at ``Q = 0`` (random) for any director, ``+2/3`` at ``Q = 1, mu = 1``
(aligned to the beam), and ``-1/3`` at ``Q = 1, mu = 0`` (aligned across the beam:
cells expand sideways, contributing a *negative* axial deviatoric strain). The signed
result is physical and is carried through; the chain takes its axial magnitude.

This module computes that geometric factor and the deviatoric Eshelby response that
multiplies it. It is the directional complement to the scalar ``kappa`` trace in
``eshelby.py``; both are exact in their limits and reduce the model to its isotropic
form when ``beta = 0`` or ``Q = 0``.
"""

from __future__ import annotations

from base_neural_model.mechanics.eshelby import eshelby_sphere_components


def legendre_p2(x: float) -> float:
    """Second Legendre polynomial ``P2(x) = (3 x^2 - 1) / 2`` (the nematic kernel)."""
    return 0.5 * (3.0 * x * x - 1.0)


def axial_orientation_factor(order_parameter: float, director_projection: float) -> float:
    r"""Axial directional survival ``g = <(u.z)^2> - 1/3 = (2/3) Q P2(mu)``.

    The geometric factor that scales the deviatoric (directional) per-cell strain into
    its net **axial** component once averaged over the population orientation
    distribution. ``order_parameter`` is the uniaxial orientation order ``Q in [0, 1]``
    (0 = random, 1 = perfectly aligned along the director); ``director_projection`` is
    ``mu = (director . z)`` in ``[-1, 1]``, the cosine of the angle between the
    population's mean axis and the beam.

    Limits (the validation anchors):

    * ``Q = 0``: returns 0 for any director - random orientation kills the directional
      term, exactly as the isotropic model assumes.
    * ``Q = 1, mu = +-1`` (aligned to the beam): returns ``+2/3`` (max axial survival).
    * ``Q = 1, mu = 0`` (aligned across the beam): returns ``-1/3`` (cells expand
      sideways; the axial deviatoric strain is negative).
    """
    if not 0.0 <= order_parameter <= 1.0:
        raise ValueError(
            f"order_parameter must lie in [0, 1], got {order_parameter!r}"
        )
    if not -1.0 <= director_projection <= 1.0:
        raise ValueError(
            f"director_projection must lie in [-1, 1], got {director_projection!r}"
        )
    return (2.0 / 3.0) * order_parameter * legendre_p2(director_projection)


def deviatoric_eshelby_response(nu: float) -> float:
    r"""Constrained deviatoric response ``S_1111 - S_1122`` for a spherical inclusion.

    The Eshelby tensor maps a *deviatoric* (uniaxial, shape-changing) eigenstrain to
    its constrained value with a different eigenvalue than the dilatational trace. For
    the uniaxial deviatoric component ``(u u - 1/3 I)`` the constrained axial part
    scales with ``S_1111 - S_1122``, the directional analogue of the
    ``S_1111 + 2 S_1122`` trace that sets ``kappa``. From the boxed sphere
    components::

        S_1111 - S_1122 = [(7 - 5 nu) - (5 nu - 1)] / [15 (1 - nu)]
                        = (8 - 10 nu) / [15 (1 - nu)] ,

    the fraction of a deviatoric eigenstrain realized as constrained deviatoric
    strain. Unlike the *volumetric* trace (which is forced toward 1 as ``nu -> 1/2``),
    this deviatoric eigenvalue is only weakly nu-dependent: ``8/15 ~ 0.533`` at
    ``nu = 0`` and ``2/5 = 0.4`` at the incompressible limit ``nu -> 1/2``. So
    incompressibility does **not** suppress the directional channel - a shape change at
    constant volume is exactly what an incompressible matrix still transmits, while it
    confines the volume change. The directional and volumetric channels therefore
    respond to the matrix state differently, which is part of why the directional term
    is a genuinely new, non-redundant signal.

    **Dilute (single-inclusion) limit, and why interaction is bounded here too.** Like
    ``kappa`` (:func:`base_neural_model.mechanics.eshelby.eshelby_kappa`) this is the
    single-inclusion value; inclusion-inclusion interaction is neglected. It carries the
    SAME Mori-Tanaka interaction bound as the volume trace
    (:func:`base_neural_model.mechanics.eshelby.mori_tanaka_kappa_clamped`) --
    ``<= |20 log10(1 - f)|`` -- but **suppressed by the orientation order ``Q``**: the net
    deviatoric back-field one inclusion feels from its neighbours is the population average
    of their deviatoric eigenstrains, ``beta eps_vol <u u - I/3> = beta eps_vol (2/3) Q``,
    the SAME nematic order tensor :func:`axial_orientation_factor` uses. So the deviatoric
    interaction scales with ``f Q`` (only the coherently aligned fraction sources a net
    back-field; the random fraction cancels, exactly as the dilute directional term does at
    ``Q = 0``), never exceeding the ``f`` volume bound, and reaching it only in the
    unphysical ``Q = 1`` limit. Because the directional term is itself a fraction of the
    total axial signal, the total-axial verdict impact is ``<= ~0.4 dB`` at the motor
    operating point (``f ~ 0.15``, ``Q_eff`` up to ~0.9). **Positional correlation** (aligned
    cells clustered in columnar files, which the mean-field average ignores) does NOT break
    this: the Eshelby exterior strain falls as ``(a/r)^3`` and the cell spacing at
    ``f = 0.15`` is ``~3.03`` radii ``= (4 pi / 3 f)^(1/3)``, capping per-neighbour coupling
    at ``~3.5%``; even a perfectly-aligned axial column of cells sums to a local back-field
    below the mean-field ``f Q`` estimate, so mean-field envelopes the correlated case.
    A denser cortex (``f >~ 0.4``, spacing ``< ~2`` radii) is where correlation would begin to
    matter and FEM would be needed - the model does not operate there.
    """
    s1111, s1122 = eshelby_sphere_components(nu)
    return s1111 - s1122


def directional_axial_strain(
    volumetric_strain: float,
    *,
    anisotropy: float,
    order_parameter: float,
    director_projection: float,
    nu: float,
) -> float:
    r"""Net **axial** deviatoric (directional) strain from the orientation channel.

    Composes the per-cell deviatoric magnitude ``beta * eps_vol``, the constrained
    deviatoric Eshelby response ``S_1111 - S_1122``, and the population orientation
    factor ``g(Q, mu)``::

        eps_dir_axial = (S_1111 - S_1122) * beta * eps_vol * g(Q, mu) .

    ``volumetric_strain`` is the coherent volumetric strain ``eps_vol`` already carrying
    the temporal-synchrony and volume-fraction factors (so the directional channel
    shares the same temporal coherence as the monopole). Returns a signed strain; the
    transduction chain takes the magnitude of the total axial displacement.

    Reduces to 0 when ``anisotropy = 0`` (spherical expansion) or
    ``order_parameter = 0`` (random orientation) - the isotropic model.
    """
    if not 0.0 <= anisotropy <= 1.0:
        raise ValueError(f"anisotropy must lie in [0, 1], got {anisotropy!r}")
    g = axial_orientation_factor(order_parameter, director_projection)
    return deviatoric_eshelby_response(nu) * anisotropy * volumetric_strain * g


def orientation_provenance(
    *, anisotropy: float, order_parameter: float, director_projection: float, nu: float
) -> tuple[str, ...]:
    """Assumption strings the chain appends when the directional channel is active."""
    return (
        "per-cell eigenstrain has an isotropic (volume) part AND a deviatoric "
        "(directional) part beta*(u u - 1/3 I) along the cell axis u; temporal "
        "synchrony s and directional coherence are independent axes",
        f"anisotropy fraction beta = {anisotropy} (0 = spherical expansion = the "
        "isotropic monopole-only model; 1 = fully directional)",
        f"orientation order Q = {order_parameter}, director projection on beam "
        f"mu = {director_projection}; axial factor g = (2/3) Q P2(mu) "
        "(0 at Q=0 -> the directional sum cancels for a random population)",
        f"deviatoric Eshelby response S1111 - S1122 = "
        f"{deviatoric_eshelby_response(nu):.4g} at nu = {nu} (weakly nu-dependent: "
        "~0.53 at nu=0, 0.4 at incompressibility - the directional channel is NOT "
        "suppressed by the incompressible fast-band matrix, unlike the volume trace)",
        "dilute (single-inclusion) deviatoric response; inclusion-inclusion interaction "
        "carries the same Mori-Tanaka bound as kappa but suppressed by orientation order "
        "Q (deviatoric back-field ~ f*Q, only the aligned fraction contributes), <= ~0.4 dB "
        "total-axial impact at the motor operating point (f~0.15, Q_eff up to ~0.9); "
        "positional (columnar-file) correlation is enveloped by the mean-field estimate "
        "because the (a/r)^3 exterior falloff at ~3-radii spacing caps per-neighbour "
        "coupling at ~3.5% -- see deviatoric_eshelby_response docstring",
    )
