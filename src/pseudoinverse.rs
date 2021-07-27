//! Compute the Moore-Penrose pseudoinverse of a matrix.

use ndarray::{Array, ArrayBase, Data, DataMut, Dim, Dimension, Ix, RawData};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_linalg::error::{LinalgError, Result};
use ndarray_linalg::qr::{QR, QRInto, QRSquare, QRSquareInto};
use ndarray_linalg::svddc::{SVDDC, SVDDCInto, UVTFlag};
use ndarray_linalg::triangular::{Diag, SolveTriangularInto};
use ndarray_linalg::solveh::UPLO;
use std::mem::drop;

// Turn an array of real numbers of type `A` into an array of (possibly) complex
// numbers of type `B`.  If `A` and `B` are the same type (i.e. `B` is real) the
// conversion is done efficiently without allocation of new memory.  If `A` and
// `B` are different types (i.e. `B` is complex) then a new array is created.
//
// In the future this will be possible with ArrayBase::mapv_into_any().
// https://github.com/rust-ndarray/ndarray/pull/1040
fn into_maybe_complex<A, B, S, D>(a: ArrayBase<S, D>) -> Array<B, D>
where
    S: RawData<Elem = A> + DataMut + core::any::Any,
    A: Scalar<Real=A>,
    B: Scalar<Real=A> + core::any::Any,
    D: Dimension + 'static,
{
    let mut a_opt = Some(a);
    match (&mut a_opt as &mut dyn std::any::Any).downcast_mut::<Option<Array<B, D>>>() {
        Some(b_opt) => b_opt.take().unwrap(),
        None => a_opt.take().unwrap().mapv(|a| B::from_real(a)),
    }
}

// Helper function to check if QR decomposition is full rank.
// Returns true if matrix is rank deficient.
// Takes the R matrix from QR decomposition and a reciprocal condition number.
// If any diagonal elements of `r` are smaller than the product of the largest
// diagonal element of `r` and `rcond` then the matrix is assumed to be
// rank-deficient.
fn is_rank_deficient<MaybeComplex, RealFloat, RData>(r: &ArrayBase<RData, Dim<[Ix; 2]>>, rcond: RealFloat) -> bool
where
    MaybeComplex: Lapack + Scalar<Real=RealFloat>,
    RealFloat: Scalar<Real=RealFloat> + std::cmp::PartialOrd,
    RData: Data<Elem = MaybeComplex>,
{
    let mut max = RealFloat::zero();
    let r_diag = r.diag().mapv(|el| {
        let el = el.abs();
        if el > max {
            max = el;
        }
        el
    });
    let eps = rcond * max;
    r_diag.into_iter().any(|el| el < &eps)
}

// Compute pseudoinverse of skinny matrix whose QR decomposition is `q` and `r`
// assuming the matrix is full rank.
fn pinv_qr<MaybeComplex, RealFloat, S>(q: ArrayBase<S, Dim<[Ix; 2]>>, r: ArrayBase<S, Dim<[Ix; 2]>>) -> Result<Array<MaybeComplex, Dim<[Ix; 2]>>>
where
    MaybeComplex: Lapack + Scalar<Real=RealFloat>,
    RealFloat: Scalar<Real=RealFloat> + std::cmp::PartialOrd,
    S: DataMut<Elem = MaybeComplex>,
{
    // Take q* where * is the conjugate/Hermitian transpose.
    let qh = q.mapv_into(|el| el.conj()).reversed_axes();
    // Cheaply invert upper triangular R using back
    // substitution.  The memory allocated for the identity
    // matrix is reused to store the matrix inverse.
    let identity_matrix = Array::<MaybeComplex, Dim<[Ix; 2]>>::eye(r.nrows());
    let r_inv = r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, identity_matrix)?;
    // No longer need R.
    drop(r);
    // Compute and return the pseudoinverse.
    Ok(r_inv.dot(&qh))
}

// Compute pseudoinverse of a matrix whose SVD decomposition is `u`, `s`,
// and `vh` using the reciprocal condition number `rcond`.  Returns a tuple of
// the matrix rank and the pseudoinverse.
fn pinv_svd_with_rcond<MaybeComplex, RealFloat, MaybeComplexData, RealData>(u: ArrayBase<MaybeComplexData, Dim<[Ix; 2]>>, s: ArrayBase<RealData, Dim<[Ix; 1]>>, vh: ArrayBase<MaybeComplexData, Dim<[Ix; 2]>>, rcond: RealFloat) -> (usize, Array<MaybeComplex, Dim<[Ix; 2]>>)
where
    MaybeComplex: Lapack + Scalar<Real=RealFloat>,
    RealFloat: Scalar<Real=RealFloat> + std::cmp::PartialOrd,
    MaybeComplexData: DataMut<Elem = MaybeComplex>,
    RealData: DataMut<Elem = RealFloat>,
{
    // Take v and u* where * is the conjugate/Hermitian transpose.
    let v = vh.mapv_into(|el| el.conj()).reversed_axes().into_owned();
    let uh = u.mapv_into(|el| el.conj()).reversed_axes();

    // Find the epsilon below which small singular values get zeroed out.
    // Epsilon is the product of the largest singular value and the reciprocal
    // condition number.
    let eps = rcond * s[0]; // First singular value always the largest.

    // Take the reciprocal of the singular values.
    // Zero out any singular values smaller than epsilon.
    // Rank is equal to the number of non-zero singular values.
    let mut rank = 0;
    let s: Array<MaybeComplex, _> = into_maybe_complex(s.into_owned());
    let sinv = s.mapv_into(|el| {
        if el.re() < eps {
            MaybeComplex::from_real(MaybeComplex::Real::zero())
        }
        else {
            rank += 1;
            MaybeComplex::from_real(MaybeComplex::Real::one() / el.re())
        }
    });

    // Alternative version of above using ArrayBase::mapv_into_any() instead of
    // into_maybe_complex(), pending:
    // https://github.com/rust-ndarray/ndarray/pull/1040
    //
    // let sinv = s.mapv_into_any(|el| {
    //     if el < eps {
    //         MaybeComplex::from_real(MaybeComplex::Real::zero())
    //     }
    //     else {
    //         rank += 1;
    //         MaybeComplex::from_real(MaybeComplex::Real::one() / el)
    //     }
    // });

    // Compute the pseudoinverse.
    let pinv = (v * sinv).dot(&uh);

    // Return the rank and the pseudoinverse.
    (rank, pinv)
}

/// Output of the ['PseudoInverse`] operations.
#[derive(Clone, Debug)]
pub struct PseudoInverseOutput<PInv> {
    /// The pseudoinverse matrix.
    pub pinv: PInv,
    /// The rank of the matrix that was pseudoinverted.
    pub rank: usize,
}

/// Pseudoinverse of a matrix.
pub trait PseudoInverse {
    /// Type of the returned pseudoinverse matrix.
    type PInv;
    /// Type of matrix element, may be complex.
    type Elem;
    /// Type of real part of matrix element.
    type Real;

    /// Attempt to compute the pseudoinverse using (fast) QR decomposition.
    /// If the matrix is rank deficient then falls back to (slower)
    /// [`PseudoInverse::pinv_svd()`].
    fn pinv(&self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverse::pinv()`].  Manually specify the reciprocal
    /// condition number for detecting matrix rank, default 1e-15.
    fn pinv_with_rcond(&self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Compute the pseudoinverse using singular value decomposition.
    /// This is slower than ['PseudoInverst::pinv()'] if the matrix is full-
    /// rank, but faster if you know in advance that the matrix is rank-
    /// deficient.
    fn pinv_svd(&self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverse::pinv_svd()`].  Manually specify the
    /// reciprocal condition number.  Singular values smaller than
    /// rcond * largest_singular_value are set to zero.  Default rcond = 1e-15.
    fn pinv_svd_with_rcond(&self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>>;
}

impl<MaybeComplex, RealFloat, SelfData> PseudoInverse for ArrayBase<SelfData, Dim<[Ix; 2]>>
where
    MaybeComplex: Lapack + Scalar<Real=RealFloat>,
    RealFloat: Scalar<Real=RealFloat> + std::cmp::PartialOrd,
    SelfData: Data<Elem = MaybeComplex>,
{
    type PInv = Array<MaybeComplex, Dim<[Ix; 2]>>;
    type Elem = MaybeComplex;
    type Real = RealFloat;

    fn pinv(&self) -> Result<PseudoInverseOutput<Self::PInv>> {
        // Reciprocal condition number for determining rank deficiency.
        const RCOND: f64 = 1e-15;

        // Delegate computation.
        self.pinv_with_rcond(MaybeComplex::real(RCOND))
    }

    fn pinv_with_rcond(&self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>> {
        // Ensure that reciprocal condition number is positive.
        let rcond = rcond.abs();

        // If matrix is "fat" then compute the QR decomposition of its "skinny"
        // transpose.  If matrix is square then used a memory-optimized version
        // of the QR decomposition.
        let nrows = self.nrows();
        let ncols = self.ncols();
        let res = if ncols > nrows {
            self.t().qr()
        }
        else if ncols < nrows {
            self.qr()
        }
        else {
            self.qr_square()
        };

        // Did QR decomposition succeed?
        match res {
            Err(e) => {
                if let LinalgError::Lapack(lax::error::Error::LapackComputationalFailure { return_code: _ }) = e {
                    // LAPACK failed to compute the QR decomposition due to
                    // numerical issues.  Try again using SVD.
                    self.pinv_svd_with_rcond(rcond)
                }
                else {
                    // Some other error happened that we can't recover from.
                    Err(e)
                }
            },
            Ok((q, r)) => {
                if is_rank_deficient(&r, rcond) {
                    // Fallback to SVD if matrix was rank deficient.
                    drop(q);
                    drop(r);
                    self.pinv_svd_with_rcond(rcond)
                }
                else {
                    // The matrix is full-rank, so go ahead using QR
                    // decomposition to compute the pseudoinverse.
                    let pinv = pinv_qr(q, r)?;

                    // Compose output.
                    if ncols > nrows {
                        Ok(PseudoInverseOutput{
                            pinv: pinv.reversed_axes(),
                            rank: nrows,
                        })
                    }
                    else {
                        Ok(PseudoInverseOutput{
                            pinv,
                            rank: ncols,
                        })
                    }
                }
            },
        }
    }

    fn pinv_svd(&self) ->Result<PseudoInverseOutput<Self::PInv>> {
        // Reciprocal condition number for determining rank deficiency.
        const RCOND: f64 = 1e-15;

        // Delegate computation.
        self.pinv_svd_with_rcond(MaybeComplex::real(RCOND))
    }

    fn pinv_svd_with_rcond(&self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>> {
        // Ensure that reciprocal condition number is positive.
        let rcond = rcond.abs();

        // TODO workaround for bug in ndarray-linalg.
        // https://github.com/rust-ndarray/ndarray-linalg/pull/303
        let a = self.to_owned();

        // Perform "economical" singular value decomposition using the
        // divide-and-conquer method.
        let (u, s, vh) = a.svddc(UVTFlag::Some)?;

        // Compute rank and pseudoinverse.
        let (rank, pinv) = pinv_svd_with_rcond(u.unwrap(), s, vh.unwrap(), rcond);

        // Compose return value.
        Ok(PseudoInverseOutput{ rank, pinv })
    }
}

/// Pseudoinverse of a matrix.  Attempts to be more memory-efficient than
/// [`PseudoInverse`] by consuming `self` and reusing its memory when possible.
pub trait PseudoInverseInto {
    /// Type of the returned pseudoinverse matrix.
    type PInv;
    /// Type of matrix element.
    type Elem;
    /// Type of real part of matrix element.
    type Real;

    /// Attempt to compute the pseudoinverse using (fast) QR decomposition.
    /// If the matrix is rank deficient then falls back to (slower)
    /// [`PseudoInverseInto::pinv_svd_into()`].
    fn pinv_into(self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverseInto::pinv_into()`]. Manually specify the
    /// reciprocal condition number for detecting matrix rank, default 1e-15.
    fn pinv_into_with_rcond(self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Compute the pseudoinverse using singular value decomposition.
    /// This is slower than ['PseudoInverseInto::pinv_into()'] if the matrix is
    /// full-rank, but faster if you know in advance that the matrix is rank-
    /// deficient.
    fn pinv_svd_into(self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverseInto::pinv_svd_into()`].  Manually specify the
    /// reciprocal condition number.  Singular values smaller than
    /// rcond * largest_singular_value are set to zero.  Default rcond = 1e-15.
    fn pinv_svd_into_with_rcond(self, eps: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Attempt to compute the pseudoinverse using QR decomposition only.
    /// This is not faster than [`PseudoInverseInto::pinv_into()`], but it is
    /// potentially more memory efficient.  However, it will fail if the matrix
    /// is rank-deficient.
    ///
    /// Note that there is no `PseudoInverse::pinv_qr()` because it would not
    /// offer any advantage over `PseudoInverse::pinv()`.
    fn pinv_qr_into(self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverseInto::pinv_qr_into()`].   Manually specify the
    /// reciprocal condition number for detecting matrix rank, default 1e-15.
    fn pinv_qr_into_with_rcond(self, eps: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>>;
}

// Essentially the same algorithm as PseudoInverse.
// See impl of PseudoInverse for detailed comments.
impl<MaybeComplex, RealFloat, SelfData> PseudoInverseInto for ArrayBase<SelfData, Dim<[Ix; 2]>>
where
    MaybeComplex: Lapack + Scalar<Real=RealFloat>,
    RealFloat: Scalar<Real=RealFloat> + std::cmp::PartialOrd,
    SelfData: DataMut<Elem = MaybeComplex>,
{
    type PInv = Array<MaybeComplex, Dim<[Ix; 2]>>;
    type Elem = MaybeComplex;
    type Real = RealFloat;

    fn pinv_into(self) -> Result<PseudoInverseOutput<Self::PInv>> {
        const RCOND: f64 = 1e-15;
        self.pinv_into_with_rcond(MaybeComplex::real(RCOND))
    }

    fn pinv_into_with_rcond(self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>> {
        let rcond = rcond.abs();

        // Attempt QR decomposition.
        let nrows = self.nrows();
        let ncols = self.ncols();
        let res = if ncols > nrows {
            self.t().qr()
        }
        else if ncols < nrows {
            self.qr()
        }
        else {
            self.qr_square()
        };
        match res {
            Err(e) => {
                if let LinalgError::Lapack(lax::error::Error::LapackComputationalFailure { return_code: _ }) = e {
                    self.pinv_svd_into_with_rcond(rcond)
                }
                else {
                    Err(e)
                }
            },
            Ok((q, r)) => {
                if is_rank_deficient(&r, rcond) {
                    // Fallback to SVD if matrix was rank deficient.
                    drop(q);
                    drop(r);
                    self.pinv_svd_into_with_rcond(rcond)
                }
                else {
                    // Compute the pseudoinverse from QR decomposition.
                    let pinv = pinv_qr(q, r)?;

                    // Compose output.
                    if ncols > nrows {
                        Ok(PseudoInverseOutput{
                            pinv: pinv.reversed_axes(),
                            rank: nrows,
                        })
                    }
                    else {
                        Ok(PseudoInverseOutput{
                            pinv,
                            rank: ncols,
                        })
                    }
                }
            },
        }
    }

    fn pinv_svd_into(self) ->Result<PseudoInverseOutput<Self::PInv>> {
        const RCOND: f64 = 1e-15;
        self.pinv_svd_into_with_rcond(MaybeComplex::real(RCOND))
    }

    fn pinv_svd_into_with_rcond(self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>> {
        let rcond = rcond.abs();
        let (u, s, vh) = self.svddc_into(UVTFlag::Some)?;
        let (rank, pinv) = pinv_svd_with_rcond(u.unwrap(), s, vh.unwrap(), rcond);
        Ok(PseudoInverseOutput{ rank, pinv })
    }

    fn pinv_qr_into(self) -> Result<PseudoInverseOutput<Self::PInv>> {
        const RCOND: f64 = 1e-15;
        self.pinv_qr_into_with_rcond(MaybeComplex::real(RCOND))
    }

    fn pinv_qr_into_with_rcond(self, rcond: Self::Real) -> Result<PseudoInverseOutput<Self::PInv>> {
        let rcond = rcond.abs();

        // QRInto requires self argument to be DataMut.
        // Make self be DataMut if it isn't already.
        let a = self.into_owned();

        // Attempt QR decomposition.
        let nrows = a.nrows();
        let ncols = a.ncols();
        let (q, r) = if ncols > nrows {
            a.reversed_axes().qr_into()?
        }
        else if ncols < nrows {
            a.qr_into()?
        }
        else {
            let (q, r) = a.qr_square_into()?;
            (q.into_owned(), r)
        };

        // Raise error if rank deficient.
        if is_rank_deficient(&r, rcond) {
            return Err(LinalgError::Lapack(lax::error::Error::LapackComputationalFailure{return_code: 2}));
        }

        // Compute the pseudoinverse.
        let pinv = pinv_qr(q, r)?;

        // Compose output.
        if ncols > nrows {
            Ok(PseudoInverseOutput{
                pinv: pinv.reversed_axes(),
                rank: nrows,
            })
        }
        else {
            Ok(PseudoInverseOutput{
                pinv,
                rank: ncols,
            })
        }
    }
}

#[cfg(test)]
mod test {
    extern crate blas_src;
    use approx::AbsDiffEq;
    use ndarray::{array, Data, Dim, Dimension, s, ShapeBuilder, Zip};
    use ndarray_rand::RandomExt;
    use num_complex::Complex64;
    use rand_distr::StandardNormal;
    use super::*;

    // Generate a matrix from the random normal distribution.
    // Odds are the matrix will be full rank.
    fn make_full_rank<Sh: ShapeBuilder>(shape: Sh) -> Array::<f64, <Sh as ShapeBuilder>::Dim> {
        Array::<f64, _>::random(shape, StandardNormal)
    }

    // Generate a complex matrix by zipping together a real matrix `r`
    // and an imaginary matrix `i`.
    fn make_complex<S, D>(r: &ArrayBase<S, D>, i: &ArrayBase<S, D>) -> Array<Complex64, D>
    where
        S: Data<Elem = f64>,
        D: Dimension,
    {
        let mut c = Array::<Complex64, _>::zeros(r.raw_dim());
        Zip::from(&mut c).and(r).and(i).apply(|c, r, i| *c = Complex64::new(*r, *i));
        c
    }

    // Split a complex matrix into its real and imaginary parts.
    // e.g. `(r, i) = split_complex(c);``
    fn split_complex<S, D>(c: &ArrayBase<S, D>) -> (Array<f64, D>, Array<f64, D>)
    where
        S: Data<Elem = Complex64>,
        D: Dimension,
    {
        let mut r = Array::<f64, _>::zeros(c.raw_dim());
        let mut i = Array::<f64, _>::zeros(c.raw_dim());
        Zip::from(c).and(&mut r).and(&mut i).apply(|c, r, i| {
            *r = c.re();
            *i = c.im();
        });
        (r, i)
    }

    // Generate a 2d matrix from the random normal distribution
    // where the third column is the sum of the first 2 columns.
    // The matrix will have rank n - 1 where n is the number of columns.
    fn make_rank_deficient<Sh: ShapeBuilder>(shape: Sh) -> Array::<f64, Dim<[Ix; 2]>> {
        let a = Array::<f64, _>::random(shape, StandardNormal);
        let mut a = a.into_dimensionality::<Dim<[Ix; 2]>>().unwrap();
        let shape = a.shape();
        assert!(shape[1] >= 3, "Matrix must have at least 3 columns.");
        let (col0, col1, mut col2) = a.multi_slice_mut((s![.., 0], s![.., 1], s![.., 2]));
        col2.assign(&(&col0.view() + &col1.view()));
        a
    }

    #[test]
    fn test_into_maybe_complex() {
        // Convert from real into real.
        let a: Array<f64, _> = array![1., 2., 3.];
        let b: Array<f64, _> = into_maybe_complex(a);
        assert!(b == array![1., 2., 3.]);
        // Convert from real into complex.
        let a: Array<f64, _> = array![1., 2., 3.];
        let b: Array<Complex64, _> = into_maybe_complex(a);
        assert!(b == array![Complex64::new(1., 0.), Complex64::new(2., 0.), Complex64::new(3., 0.)]);
    }

    #[test]
    fn test_is_rank_deficient() {
        // Generate a random, full-rank, 5x3 matrix.
        let a = make_full_rank((5, 3));
        // Perform QR decomposition.
        let (_, r) = a.qr_into().unwrap();
        // Check that matrix was full rank.
        assert!(!is_rank_deficient(&r, 1e-15));
        // Genreate a random, rank-deficient 5x3 matrix (rank is 2).
        let a = make_rank_deficient((5, 3));
        // Perform QR decomposition.
        let (_, r) = a.qr_into().unwrap();
        // Check that matrix was rank deficient.
        assert!(is_rank_deficient(&r, 1e-15));
    }

    #[test]
    fn test_svd_skinny() {
        // Generate a random, full-rank, 5x3 matrix.
        let a = make_full_rank((5, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_svd_skinny_complex() {
        // Generate a random, full-rank, 5x3 complex matrix.
        // Generate the real part.
        let r = make_full_rank((5, 3));
        // Generate the imaginary part.
        let i = make_full_rank((5, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_qr_into_skinny() {
        // Generate a random, full-rank, 5x3 matrix.
        let a = make_full_rank((5, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_qr_into().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_qr_into_skinny_complex() {
        // Generate a random, full-rank, 5x3 complex matrix.
        // Generate the real part.
        let r = make_full_rank((5, 3));
        // Generate the imaginary part.
        let i = make_full_rank((5, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_qr_into().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_pinv_skinny() {
        // Generate a random, full-rank, 5x3 matrix.
        let a = make_full_rank((5, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_pinv_skinny_complex() {
        // Generate a random, full-rank, 5x3 complex matrix.
        // Generate the real part.
        let r = make_full_rank((5, 3));
        // Generate the imaginary part.
        let i = make_full_rank((5, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_svd_fat() {
        // Generate a random, full-rank, 3x5 matrix.
        let a = make_full_rank((3, 5));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_svd_fat_complex() {
        // Generate a random, full-rank (rows), 3x5 complex matrix.
        // Generate the real part.
        let r = make_full_rank((3, 5));
        // Generate the imaginary part.
        let i = make_full_rank((3, 5));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_qr_into_fat() {
        // Generate a random, full-rank, 3x5 matrix.
        let a = make_full_rank((3, 5));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_qr_into().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_qr_into_fat_complex() {
        // Generate a random, full-rank (rows), 3x5 complex matrix.
        // Generate the real part.
        let r = make_full_rank((3, 5));
        // Generate the imaginary part.
        let i = make_full_rank((3, 5));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_qr_into().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_pinv_fat() {
        // Generate a random, full-rank, 3x5 matrix.
        let a = make_full_rank((3, 5));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_pinv_fat_complex() {
        // Generate a random, full-rank, 3x4 complex matrix.
        // Generate the real part.
        let r = make_full_rank((3, 5));
        // Generate the imaginary part.
        let i = make_full_rank((3, 5));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_svd_square() {
        // Generate a random, full-rank, square matrix.
        let a = make_full_rank((3, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_svd_square_complex() {
        // Generate a random, full-rank, square, complex matrix.
        // Generate the real part.
        let r = make_full_rank((3, 3));
        // Generate the imaginary part.
        let i = make_full_rank((3, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_qr_into_square() {
        // Generate a random, full-rank, square matrix.
        let a = make_full_rank((3, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_qr_into().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_qr_into_square_complex() {
        // Generate a random, full-rank, square, complex matrix.
        // Generate the real part.
        let r = make_full_rank((3, 3));
        // Generate the imaginary part.
        let i = make_full_rank((3, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_qr_into().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_pinv_square() {
        // Generate a random, full-rank, square matrix.
        let a = make_full_rank((3, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_pinv_square_complex() {
        // Generate a random, full-rank, square, complex matrix.
        // Generate the real part.
        let r = make_full_rank((3, 3));
        // Generate the imaginary part.
        let i = make_full_rank((3, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_svd_rank_deficient() {
        // Generate a random rank-deficient (5x3 with rank 2) matrix.
        let a = make_rank_deficient((5, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 2.
        assert!(rank == 2);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 2);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_svd_rank_deficient_complex() {
        // Generate a random rank-deficient (5x3 with rank 2) complex matrix.
        // Generate the real part.
        let r = make_rank_deficient((5, 3));
        // Generate the imaginary part.
        let i = make_rank_deficient((5, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 2.
        assert!(rank == 2);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 2);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_qr_into_rank_deficient() {
        // Generate a random rank-deficient (5x3 with rank 2) matrix.
        let a = make_rank_deficient((5, 3));
        // The pseudoinverse should fail.
        assert!(a.pinv_qr_into().is_err());
    }

    #[test]
    fn test_qr_into_rank_deficient_complex() {
        // Generate a random rank-deficient (5x3 with rank 2) complex matrix.
        // Generate the real part.
        let r = make_rank_deficient((5, 3));
        // Generate the imaginary part.
        let i = make_rank_deficient((5, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // The pseudoinverse should fail.
        assert!(a.pinv_qr_into().is_err());
    }

    #[test]
    fn test_pinv_rank_deficient() {
        // Generate a random rank-deficient (5x3 with rank 2) matrix.
        let a = make_rank_deficient((5, 3));
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 2.
        assert!(rank == 2);
        // a.dot(&pinv).dot(&a) should equal a.
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 2);
        assert!(a.abs_diff_eq(&a.dot(&pinv).dot(&a), 1e-4));
    }

    #[test]
    fn test_pinv_rank_deficient_complex() {
        // Generate a random rank-deficient (5x3 with rank 2) complex matrix.
        // Generate the real part.
        let r = make_rank_deficient((5, 3));
        // Generate the imaginary part.
        let i = make_rank_deficient((5, 3));
        // Zip parts together into a complex matrix.
        let a = make_complex(&r, &i);
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 2.
        assert!(rank == 2);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 2);
        let aa = a.dot(&pinv).dot(&a);
        let (rr, ii) = split_complex(&aa);
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }
}
