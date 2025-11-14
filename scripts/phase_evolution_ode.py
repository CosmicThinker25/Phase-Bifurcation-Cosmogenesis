"""
phase_evolution_ode.py
----------------------

Numerical integrator for the Siamese phase-difference equation:

    Δφ¨ + 3H(a) Δφ˙ + m_φ² Δφ = S_rot(a)

where H(a) is the effective Loop Quantum Cosmology Hubble rate,
and S_rot(a) encodes the rotational source arising from primordial
spin / orientation asymmetry.

This module provides:
- H_lqc(a, params):   effective LQC Hubble rate
- S_rot(a, y, params): rotational source term
- phase_ode(a, y, params): ODE system
- run_phase_evolution(): high-level solver returning Δφ(a)

No data or external files are required. The module can be imported
from other scripts (e.g. phase_sector_scan.py) or executed alone
for quick tests.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ================================================================
# 1. Loop Quantum Cosmology background (effective Hubble function)
# ================================================================

def H_lqc(a, params):
    """
    Effective Hubble rate H(a) used in the Δφ evolution.

    This is a placeholder / simplified form. Replace with your
    preferred LQC effective expression if needed.

    Parameters
    ----------
    a : float
        Scale factor.
    params : dict
        Dictionary containing cosmological parameters.

    Returns
    -------
    float : Hubble rate at scale factor a.
    """
    H0 = params.get("H0", 1.0)

    # Simple matter-like scaling H ∝ a^{-3/2}
    # (Replace with your actual LQC model when ready)
    return H0 * np.sqrt(a**(-3))


# ================================================================
# 2. Rotational source term S_rot(a)
# ================================================================

def S_rot(a, y, params):
    """
    Rotational source term S_rot(a) for the phase equation.

    Parameters
    ----------
    a : float
        Scale factor.
    y : array-like
        State vector (Δφ, Δφ˙). Included for consistency; often unused.
    params : dict
        Contains k_rot and q.

    Returns
    -------
    float
    """
    k_rot = params.get("k_rot", 0.0)
    q = params.get("q", 1.0)

    # Simple model: S_rot(a) ∝ a^{-q}
    return k_rot * a**(-q)


# ================================================================
# 3. Full ODE system
# ================================================================

def phase_ode(a, y, params):
    """
    System of first-order ODEs for Δφ(a).

    y = (Δφ, Δφ˙)

    Parameters
    ----------
    a : float
        Scale factor (independent variable).
    y : array-like
        State vector [Δφ, Δφ˙].
    params : dict
        Must contain m_phi; may contain H0, k_rot, q.

    Returns
    -------
    list : [d(Δφ)/da, d(Δφ˙)/da]
    """
    delta_phi, delta_phidot = y
    m_phi = params.get("m_phi", 1.0)

    H = H_lqc(a, params)
    source = S_rot(a, y, params)

    # Second-order ODE written as two first-order:
    dphi_da = delta_phidot
    dphidot_da = -3.0 * H * delta_phidot - m_phi**2 * delta_phi + source

    return [dphi_da, dphidot_da]


# ================================================================
# 4. High-level solver
# ================================================================

def run_phase_evolution(
    m_phi,
    k_rot,
    q,
    delta_phi_ini=0.0,
    delta_phidot_ini=0.0,
    a_ini=1e-3,
    a_max=10.0,
    n_steps=2000,
    H0=1.0,
    rtol=1e-7,
    atol=1e-9,
):
    """
    Integrates Δφ(a) from a_ini to a_max.

    Parameters
    ----------
    m_phi : float
    k_rot : float
    q : float
    delta_phi_ini : float
        Initial Δφ(a_ini).
    delta_phidot_ini : float
        Initial Δφ˙(a_ini).
    a_ini, a_max : float
        Integration interval.
    n_steps : int
        Number of output evaluation points.
    H0 : float
        Effective Hubble normalisation in H_lqc.
    rtol, atol : float
        Integration tolerances.

    Returns
    -------
    a_arr : ndarray
        Scale factor values.
    delta_phi_arr : ndarray
        Phase difference Δφ(a).
    delta_phidot_arr : ndarray
        Phase derivative Δφ˙(a).
    """

    params = {
        "m_phi": m_phi,
        "k_rot": k_rot,
        "q": q,
        "H0": H0,
    }

    y0 = [delta_phi_ini, delta_phidot_ini]
    a_span = (a_ini, a_max)
    a_eval = np.linspace(a_ini, a_max, n_steps)

    sol = solve_ivp(
        fun=lambda a, y: phase_ode(a, y, params),
        t_span=a_span,
        y0=y0,
        t_eval=a_eval,
        rtol=rtol,
        atol=atol,
    )

    return sol.t, sol.y[0], sol.y[1]


# ================================================================
# 5. Optional quick test (does not run when imported)
# ================================================================

if __name__ == "__main__":
    # Minimal sanity test to verify integration works
    import matplotlib.pyplot as plt

    a, dphi, dphidot = run_phase_evolution(
        m_phi=0.5,
        k_rot=0.2,
        q=1.0,
        delta_phi_ini=0.01,
        delta_phidot_ini=0.0,
    )

    plt.plot(a, dphi)
    plt.xlabel("Scale factor a")
    plt.ylabel(r"$\Delta\phi(a)$")
    plt.title("Test run of the Siamese Phase Evolution")
    plt.tight_layout()
    plt.show()
