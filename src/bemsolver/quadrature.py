import numpy as np

def triquad(N_in: int, v: np.ndarray):
    """
    Gaussian quadrature rule for a triangular domain using square->triangle collapse.
    Based on the matlab function triquad.m written by Greg von Winckel.

    Parameters
    ----------
    N_in : int
        Quadrature order N. The rule produces N^2 points.
    v : (3,2) array_like
        Triangle vertices [[x1,y1],
                           [x2,y2],
                           [x3,y3]]

    Returns
    -------
    X : (N,N) ndarray
        x-coordinates of quadrature nodes on the physical triangle.
    Y : (N,N) ndarray
        y-coordinates of quadrature nodes on the physical triangle.
    Wx : (N,) ndarray
        Quadrature weights in the "x" direction.
    Wy : (N,) ndarray
        Quadrature weights in the "y" direction.

    Notes
    -----
    To integrate a scalar function f(x,y) over the triangle:
        fvals = f(X, Y)          # shape (N,N)
        Q = Wx @ fvals @ Wy      # scalar ~ integral over the triangle

    This is a translation of Greg von Winckel's MATLAB triquad.m
    (Gaussian cubature on triangles via a collapsed square construction).
    """

    N = int(N_in)
    v = np.asarray(v, dtype=float)

    # --- First part: Gauss-Jacobi-ish rule in one direction ---
    n = np.arange(1, N + 1, dtype=float)          # 1:N
    nnk = 2 * n + 1
    A = np.concatenate(([1.0 / 3.0],
                        1.0 / (nnk * (nnk + 2.0))))

    n2 = np.arange(2, N + 1, dtype=float)         # 2:N
    nnk_sel = nnk[(n2 - 1).astype(int)]          # nnk(n) in MATLAB (1-based)
    B1 = 2.0 / 9.0
    nk = n2 + 1.0
    nnk2_sq = nnk_sel * nnk_sel
    B = 4.0 * (n2 * nk)**2 / (nnk2_sq * nnk2_sq - nnk2_sq)

    ab_col1 = A
    ab_col2 = np.concatenate(([2.0, B1], B))
    ab = np.column_stack((ab_col1, ab_col2))      # (N+1) x 2

    # off-diagonals for Jacobi matrix
    s = np.sqrt(ab[1:N, 1])                       # ab(2:N,2)

    # Build symmetric tridiagonal Jacobi matrix
    main_diag = ab[0:N, 0]                        # ab(1:N,1)
    J = np.diag(main_diag)
    J += np.diag(s, k=-1)
    J += np.diag(s, k=1)

    # Eigen-decompose (symmetric)
    vals, vecs = np.linalg.eigh(J)

    # Map eigenvalues from [-1,1] to [0,1]
    x = (vals + 1.0) / 2.0            # length N
    # Weights in that direction
    wx = ab[0, 1] * (vecs[0, :]**2) / 4.0   # length N

    # --- Second part: Gauss-Legendre-like rule in the collapsed direction ---
    # This block computes y and Wy via Newton iteration on Legendre polys.

    N_leg = N - 1            # corresponds to MATLAB's "N=N-1;"
    N1 = N_leg + 1           # = N
    N2 = N_leg + 2           # = N+1

    # Initial guess for roots (Chebyshev nodes)
    kvec = np.arange(N_leg, -1, -1, dtype=float)    # N_leg:-1:0
    y = np.cos((2 * kvec + 1) * np.pi / (2 * N_leg + 2))
    y0 = 2.0 * np.ones_like(y)

    L = np.zeros((N1, N2))
    eps_val = np.finfo(float).eps
    iters = 0

    # Newton-Raphson to find roots of Legendre polynomial of degree N2-1
    # L[:,m] will hold P_{m-1}(y)
    while np.max(np.abs(y - y0)) > eps_val:
        L[:, 0] = 1.0       # P0
        L[:, 1] = y         # P1
        for k in range(2, N1 + 1):   # k = 2..N1
            # Recurrence: P_k
            L[:, k] = ((2*k - 1) * y * L[:, k-1] - (k - 1) * L[:, k-2]) / k

        # Derivative of P_{N1} using the standard relation
        # In MATLAB: Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);
        # Careful with 0-based indexing:
        #   L(:,N1)   -> L[:, N1-1] (P_{N1-1})
        #   L(:,N2)   -> L[:, N1]   (P_{N1})
        Lp = (N2) * (L[:, N1-1] - y * L[:, N1]) / (1.0 - y**2)

        y0 = y.copy()
        # Newton update for roots of P_{N1} (which is L[:, N1])
        y = y0 - L[:, N1] / Lp

        iters += 1
        if iters > 1000:
            raise RuntimeError("Legendre root-finding did not converge")

    # Affine mapping coefficients from (x,t)->(X,Y) on triangle
    # cd = [1 0 0; -1 0 1; 0 1 -1] * v   in MATLAB
    cd = np.array([[1.0, 0.0, 0.0],
                   [-1.0, 0.0, 1.0],
                   [0.0, 1.0, -1.0]]) @ v    # shape (3,2)

    t = (1.0 + y) / 2.0

    # Wx, Wy (1D quadrature weights in each collapsed direction)
    # Wx = abs(det(cd(2:3,:)))*wx;
    # cd(2:3,:) is rows 2..3 in MATLAB -> cd[1:3,:] in Python
    Wx = abs(np.linalg.det(cd[1:3, :])) * wx

    # Wy = 1./((1-y.^2).*Lp.^2)*(N2/N1)^2;
    Wy = (1.0 / ((1.0 - y**2) * (Lp**2))) * (N2 / N1)**2

    # Now generate tensor-product nodes on the reference square, collapse to triangle
    # [tt,xx] = meshgrid(t,x); yy = tt.*xx;
    tt, xx = np.meshgrid(t, x)   # shape (N,N)
    yy = tt * xx

    # Physical coordinates:
    # X = cd(1,1)+cd(2,1)*xx+cd(3,1)*yy;
    # Y = cd(1,2)+cd(2,2)*xx+cd(3,2)*yy;
    X = cd[0, 0] + cd[1, 0] * xx + cd[2, 0] * yy
    Y = cd[0, 1] + cd[1, 1] * xx + cd[2, 1] * yy

    return X, Y, Wx, Wy
