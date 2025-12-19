import numpy as np

from synthe_py.tools.gibbs_solver import minimize_gibbs


def _toy_h_h2(mu_h=0.0, mu_h2=-4.0):
    """
    Construct a toy H/H2 system:
    species: H, H2
    elements: H
    mu0: configurable to bias molecular formation (more negative -> favored)
    """
    mu0 = np.array([mu_h, mu_h2], dtype=np.float64)
    stoich = np.array([[1.0], [2.0]], dtype=np.float64)  # rows: species, cols: elements
    elem_totals = np.array([1.0], dtype=np.float64)  # 1 unit of hydrogen atoms
    return mu0, stoich, elem_totals


def test_high_temperature_atomic_limit():
    # High T should favor atomic H
    T = 10000.0
    from synthe_py.tools.gibbs_solver import k_BOLTZ

    P = k_BOLTZ * T  # so ntot target ~1
    mu0, stoich, elem_totals = _toy_h_h2(mu_h=0.0, mu_h2=+2.0)
    n = minimize_gibbs(T, P, mu0, stoich, elem_totals, max_iter=300)
    n_h, n_h2 = n
    # Element conservation
    assert np.isclose(n_h + 2 * n_h2, elem_totals[0], rtol=1e-4, atol=1e-6)
    # Atomic dominance (with positive mu_h2, H2 disfavored)
    assert n_h > 0.7 * elem_totals[0]
    assert n_h2 < 0.3 * elem_totals[0]


def test_low_temperature_molecular_limit():
    # Low T should favor H2
    T = 2000.0
    from synthe_py.tools.gibbs_solver import k_BOLTZ

    P = k_BOLTZ * T  # so ntot target ~1
    mu0, stoich, elem_totals = _toy_h_h2(mu_h=0.0, mu_h2=-50.0)
    n = minimize_gibbs(T, P, mu0, stoich, elem_totals, max_iter=400)
    n_h, n_h2 = n
    # Element conservation
    assert np.isclose(n_h + 2 * n_h2, elem_totals[0], rtol=2e-3, atol=1e-5)
    # Molecular dominance
    assert n_h2 > 0.45 * elem_totals[0]  # at least ~90% of atoms in H2
    assert n_h < 0.2 * elem_totals[0]


def test_mid_temperature_mixed_state():
    # Intermediate T should give mixed populations
    T = 3500.0
    from synthe_py.tools.gibbs_solver import k_BOLTZ

    P = k_BOLTZ * T  # so ntot target ~1
    mu0, stoich, elem_totals = _toy_h_h2(mu_h=0.0, mu_h2=-3.0)
    n = minimize_gibbs(T, P, mu0, stoich, elem_totals, max_iter=400)
    n_h, n_h2 = n
    # Element conservation
    assert np.isclose(n_h + 2 * n_h2, elem_totals[0], rtol=1e-2, atol=1e-5)
    # Mixed-ish regime: both nonzero
    frac_h = n_h / elem_totals[0]
    frac_h2 = 2 * n_h2 / elem_totals[0]
    assert frac_h > 0.05
    assert frac_h2 > 0.05
