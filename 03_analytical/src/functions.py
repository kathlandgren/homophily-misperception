

def compute_p_oo(hoo,hos,C):
    """
    Compute the probability of opponent-opponent connection p_oo
    given the homophily parameters hoo and hos, and the growth factor C.
    """
    p_oo = (hoo * C) / (hoo * C + hos * (2 - C))
    return p_oo

def compute_p_ss(hss,hso,C):
    """
    Compute the probability of supporter-supporter connection p_ss
    given the homophily parameters hss and hso, and the growth factor C.
    """
    p_ss = (hss * (2 - C)) / (hss * (2 - C) + hso * C)
    return p_ss

def compute_p_os(hos,hoo,C):
    """
    Compute the probability of opponent-supporter connection p_os
    given the homophily parameters hos and hoo, and the growth factor C.
    """
    p_os = (hos * (2 - C)) / (hoo * C + hos * (2 - C))
    return p_os

def compute_p_so(hso,hss,C):
    """
    Compute the probability of supporter-opponent connection p_so
    given the homophily parameters hso and hss, and the growth factor C.
    """
    p_so = (hso * C) / (hss * (2 - C) + hso * C)
    return p_so

def compute_C(fo, fs, hoo, hos, hso, hss, tol=1e-12, max_iter=200):
    """
    Solve for C in [0, 2] from:
        C = f_o*(1 + (h_oo*C)/(h_oo*C + h_os*(2 - C))) \
            + f_s*((h_so*C)/(h_ss*(2 - C) + h_so*C))

    Assumes all h_* >= 0 and f_o, f_s >= 0 with f_o + f_s = 1.
    Uses bisection, guaranteed to converge for this bracket.
    """
    def rhs(C):
        den1 = hoo*C + hos*(2 - C)
        den2 = hss*(2 - C) + hso*C
        term1 = 0.0 if den1 == 0 else (hoo*C) / den1
        term2 = 0.0 if den2 == 0 else (hso*C) / den2
        return fo*(1 + term1) + fs*term2

    a, b = 0.0, 2.0
    fa, fb = rhs(a) - a, rhs(b) - b
    if fa == 0: return a
    if fb == 0: return b

    for _ in range(max_iter):
        m = 0.5*(a + b)
        fm = rhs(m) - m
        if abs(fm) <= tol or (b - a) <= tol:
            return m
        if fa * fm > 0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5*(a + b)

def compute_opponent_perception(No, Ns, p_os,p_so, p_oo):
    """
    Compute the perceived fraction of supporters by opponent group, on average.
    """
    total_connections = 2 * p_oo + (Ns/No)*p_so + p_os
    perceived_supporters = (Ns/No)*p_so+p_os
    perceived_fraction = perceived_supporters / total_connections
    return perceived_fraction

def compute_supporter_perception(No, Ns, p_os,p_so, p_ss):
    """
    Compute the perceived fraction of supporters by supporter group, on average.
    """
    total_connections = 2 * p_ss + (No/Ns)*p_os + p_so
    perceived_supporters = 2 * p_ss
    perceived_fraction = perceived_supporters / total_connections
    return perceived_fraction

def compute_misperception(fs, perceived_fraction):
    """
    Compute the misperception of supporters, defined as the difference between
    the actual fraction of supporters and the perceived fraction.
    """
    misperception = fs - perceived_fraction
    return misperception

def compute_overall_misperception(fs, beta_o, beta_s):
    """
    Compute the overall misperception in the network, defined as the weighted average
    of the misperceptions of both groups.
    """
    overall_misperception = fs * beta_s + (1 - fs) * beta_o
    return overall_misperception

def rescale_bayes(p, delta, gamma):
    """
    Rescale the perceived fraction using the Guay et al. Bayesian rescaling function
    """
    rescaled_p = delta**(1-gamma)*p**gamma / (delta**(1-gamma)*p**gamma + (1-p)**gamma)
    return rescaled_p
