//! Compute the Moore-Penrose pseudoinverse of a matrix.

use ndarray::{Array, ArrayBase, Data, DataMut, Dim, Dimension, Ix, RawData};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_linalg::error::{LinalgError, Result};
use ndarray_linalg::qr::{QR, QRSquare};
use ndarray_linalg::svddc::{SVDDC, SVDDCInto, UVTFlag};
use ndarray_linalg::triangular::{Diag, SolveTriangularInto};
use ndarray_linalg::solveh::UPLO;
use std::mem::drop;

// Turn an array of real numbers of type `A` into an array of (possibly) complex
// numbers of type `B`.  If `A` and `B` are the same type (i.e. `B` is real) the
// conversion is done efficiently without allocation of new memory.  If `A` and
// `B` are different types (i.e. `B` is complex) then a new array is created.
//
// In the future this may be possible with mapv_into_any().
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
    /// Type of matrix element.
    type Elem;

    /// Attempt to compute the pseudoinverse using (fast) QR decomposition.
    /// If the matrix is rank deficient then falls back to (slower)
    /// [`PseudoInverse::pinv_svd()`].
    fn pinv(&self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverse::pinv()`].  Manually specify epsilon (small,
    // positive number close to zero) for detecting matrix rank.
    fn pinv_with_eps(&self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Compute the pseudoinverse using singular value decomposition.
    /// This is slower than ['PseudoInverst::pinv()'] if the matrix is full-
    /// rank, but faster if you know in advance that the matrix is rank-
    /// deficient.
    fn pinv_svd(&self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverse::pinv_svd()`].  Manually specify epsilon (small,
    /// positive number close to zero) for detecting matrix rank.
    fn pinv_svd_with_eps(&self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>>;
}

impl<MaybeComplex, RealFloat, SelfData> PseudoInverse for ArrayBase<SelfData, Dim<[Ix; 2]>>
where
    MaybeComplex: Lapack + Scalar<Real=RealFloat>,
    RealFloat: Scalar<Real=RealFloat> + std::cmp::PartialOrd,
    SelfData: Data<Elem = MaybeComplex>,
{
    type PInv = Array<MaybeComplex, Dim<[Ix; 2]>>;
    type Elem = MaybeComplex;

    fn pinv(&self) -> Result<PseudoInverseOutput<Self::PInv>> {
        // Small epsilon value for determining rank deficiency.
        const EPS: f64 = 1e-4;

        // Delegate computation.
        self.pinv_with_eps(MaybeComplex::from_real(MaybeComplex::real(EPS)))
    }

    fn pinv_with_eps(&self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>> {
        // Ensure that epsilon is a positive number.
        let eps = eps.abs();

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
                    self.pinv_svd_with_eps(MaybeComplex::from_real(eps))
                }
                else {
                    // Some other error happened that we can't recover from.
                    Err(e)
                }
            },
            Ok((q, r)) => {
                // If any of the diagonal elements of R is close to zero
                // then the matrix is rank deficient.
                if r.diag().iter().any(|&el| el.abs() < eps) {
                    // Try with SVD instead.
                    self.pinv_svd_with_eps(MaybeComplex::from_real(eps))
                }
                else {
                    // The matrix is full-rank, so go ahead using QR
                    // decomposition.
                    //
                    // Take q* where * is the conjugate/Hermitian transpose.
                    let qh = q.mapv_into(|el| el.conj()).reversed_axes();

                    // Cheaply invert upper triangular R using back
                    // substitution.  The memory allocated for the identity
                    // matrix is reused to store the matrix inverse.
                    let identity_matrix = Array::<MaybeComplex, Dim<[Ix; 2]>>::eye(r.nrows());
                    let r_inv = r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, identity_matrix)?;
                    drop(r);

                    // Compute the pseudoinverse.
                    let pinv =  r_inv.dot(&qh);
                    drop(r_inv);
                    drop(qh);

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
        // Small epsilon value for determining rank deficiency.
        const EPS: f64 = 1e-4;

        // Delegate computation.
        self.pinv_svd_with_eps(MaybeComplex::from_real(MaybeComplex::real(EPS)))
    }

    fn pinv_svd_with_eps(&self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>> {
        // Ensure that epsilon is a positive number.
        let eps = eps.abs();

        // TODO possible bug in ndarray_linalg!
        // SVDCC requires self argument to be DataMut.
        // Workaround by cloning an owned copy of self.
        let a = self.to_owned();

        // Perform "economical" singular value decomposition using the
        // divide-and-conquer method.
        let (u, s, vh) = a.svddc(UVTFlag::Some)?;

        // Take v and u* where * is the conjugate/Hermitian transpose.
        let v = vh.unwrap().mapv_into(|el| el.conj()).reversed_axes();
        let uh = u.unwrap().mapv_into(|el| el.conj()).reversed_axes();

        // Take the reciprocal of the singular values.
        // Zero out any singular values close to zero.
        // Note that singular values are always positive.
        // Rank is equal to the number of non-zero singular values.
        let mut rank = 0;
        let s: Array<MaybeComplex, _> = into_maybe_complex(s);
        let sinv = s.mapv_into(|el| {
            if el.re() < eps {
                MaybeComplex::from_real(MaybeComplex::Real::zero())
            }
            else {
                rank += 1;
                MaybeComplex::from_real(MaybeComplex::Real::one() / el.re())
            }
        });
        // let sinv = s.trans_mapv_into(|el| {
        //     if el < eps {
        //         MaybeComplex::from_real(MaybeComplex::Real::zero())
        //     }
        //     else {
        //         rank += 1;
        //         MaybeComplex::from_real(MaybeComplex::Real::one() / el)
        //     }
        // });

        // Compute the pseudoinverse.
        Ok(PseudoInverseOutput{
            pinv: (v * sinv).dot(&uh),
            rank,
        })
    }
}

/// Pseudoinverse of a matrix.  Attempts to be more memory-efficient than
/// [`PseudoInverse`] by consuming `self` and reusing its memory when possible.
pub trait PseudoInverseInto {
    /// Type of the returned pseudoinverse matrix.
    type PInv;

    /// Type of matrix element.
    type Elem;

    /// Attempt to compute the pseudoinverse using (fast) QR decomposition.
    /// If the matrix is rank deficient then falls back to (slower)
    /// [`PseudoInverseInto::pinv_svd_into()`].
    fn pinv_into(self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverseInto::pinv_into()`].  Manually specify epsilon
    ///(small, positive number close to zero) for detecting matrix rank.
    fn pinv_into_with_eps(self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Compute the pseudoinverse using singular value decomposition.
    /// This is slower than ['PseudoInverseInto::pinv_into()'] if the matrix is
    /// full-rank, but faster if you know in advance that the matrix is rank-
    /// deficient.
    fn pinv_svd_into(self) -> Result<PseudoInverseOutput<Self::PInv>>;

    /// Same as [`PseudoInverseInto::pinv_svd_into()`].  Manually specify
    /// epsilon (small, positive number close to zero) for detecting matrix
    /// rank.
    fn pinv_svd_into_with_eps(self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>>;
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

    fn pinv_into(self) -> Result<PseudoInverseOutput<Self::PInv>> {
        const EPS: f64 = 1e-4;
        self.pinv_into_with_eps(MaybeComplex::from_real(MaybeComplex::real(EPS)))
    }

    fn pinv_into_with_eps(self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>> {
        let eps = eps.abs();

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
                    self.pinv_svd_into_with_eps(MaybeComplex::from_real(eps))
                }
                else {
                    Err(e)
                }
            },
            Ok((q, r)) => {
                // Fallback to SVD if rank deficient.
                if r.diag().iter().any(|&el| el.abs() < eps) {
                    self.pinv_svd_into_with_eps(MaybeComplex::from_real(eps))
                }
                else {
                    // Compute the pseudoinverse.
                    drop(self);
                    let qh = q.mapv_into(|el| el.conj()).reversed_axes();
                    let identity_matrix = Array::<MaybeComplex, Dim<[Ix; 2]>>::eye(r.nrows());
                    let r_inv = r.solve_triangular_into(UPLO::Upper, Diag::NonUnit, identity_matrix)?;
                    drop(r);
                    let pinv =  r_inv.dot(&qh);
                    drop(r_inv);
                    drop(qh);

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
        const EPS: f64 = 1e-4;
        self.pinv_into_with_eps(MaybeComplex::from_real(MaybeComplex::real(EPS)))

    }

    fn pinv_svd_into_with_eps(self, eps: Self::Elem) -> Result<PseudoInverseOutput<Self::PInv>> {
        let eps = eps.abs();
        let (u, s, vh) = self.svddc_into(UVTFlag::Some)?;
        let v = vh.unwrap().mapv_into(|el| el.conj()).reversed_axes();
        let uh = u.unwrap().mapv_into(|el| el.conj()).reversed_axes();
        let mut rank = 0;
        let s: Array<MaybeComplex, _> = into_maybe_complex(s);
        let sinv = s.mapv_into(|el| {
            if el.re() < eps {
                MaybeComplex::from_real(MaybeComplex::Real::zero())
            }
            else {
                rank += 1;
                MaybeComplex::from_real(MaybeComplex::Real::one() / el.re())
            }
        });
        // let sinv = s.trans_mapv_into(|el| {
        //     if el < eps {
        //         MaybeComplex::from_real(MaybeComplex::Real::zero())
        //     }
        //     else {
        //         rank += 1;
        //         MaybeComplex::from_real(MaybeComplex::Real::one() / el)
        //     }
        // });

        Ok(PseudoInverseOutput{
            pinv: (v * sinv).dot(&uh),
            rank,
        })
    }
}

#[cfg(test)]
mod test {
    extern crate blas_src;
    use approx::AbsDiffEq;
    use ndarray::{array, s, Zip};
    use ndarray_rand::RandomExt;
    use num_complex::Complex64;
    use rand_distr::StandardNormal;
    use super::*;

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
    fn test_svd_skinny() {
        // Generate a random, full-rank, 5x3 matrix.
        let a = Array::<f64, _>::random((5, 3), StandardNormal);
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
        let r = Array::<f64, _>::random((5, 3), StandardNormal);
        // Generate the imaginary part.
        let i = Array::<f64, _>::random((5, 3), StandardNormal);
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((5, 3));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((5, 3));
        let mut ii = Array::<f64, _>::zeros((5, 3));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_pinv_skinny() {
        // Generate a random, full-rank, 5x3 matrix.
        let a = Array::<f64, _>::random((5, 3), StandardNormal);
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
        let r = Array::<f64, _>::random((5, 3), StandardNormal);
        // Generate the imaginary part.
        let i = Array::<f64, _>::random((5, 3), StandardNormal);
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((5,3));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((5,3));
        let mut ii = Array::<f64, _>::zeros((5,3));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_svd_fat() {
        // Generate a random, full-rank, 3x5 matrix.
        let a = Array::<f64, _>::random((3, 5), StandardNormal);
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
        let r = Array::<f64, _>::random((3,5), StandardNormal);
        // Generate the imaginary part.
        let i = Array::<f64, _>::random((3,5), StandardNormal);
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((3,5));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((3,5));
        let mut ii = Array::<f64, _>::zeros((3,5));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_pinv_fat() {
        // Generate a random, full-rank, 3x5 matrix.
        let a = Array::<f64, _>::random((3, 5), StandardNormal);
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
        let r = Array::<f64, _>::random((3, 5), StandardNormal);
        // Generate the imaginary part.
        let i = Array::<f64, _>::random((3, 5), StandardNormal);
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((3,5));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((3, 5));
        let mut ii = Array::<f64, _>::zeros((3, 5));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_svd_square() {
        // Generate a random, full-rank, square matrix.
        let a = Array::<f64, _>::random((3, 3), StandardNormal);
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
        let r = Array::<f64, _>::random((3, 3), StandardNormal);
        // Generate the imaginary part.
        let i = Array::<f64, _>::random((3, 3), StandardNormal);
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((3, 3));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((3, 3));
        let mut ii = Array::<f64, _>::zeros((3, 3));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_svd_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_pinv_square() {
        // Generate a random, full-rank, square matrix.
        let a = Array::<f64, _>::random((3, 3), StandardNormal);
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
        let r = Array::<f64, _>::random((3, 3), StandardNormal);
        // Generate the imaginary part.
        let i = Array::<f64, _>::random((3, 3), StandardNormal);
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((3,3));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 3.
        assert!(rank == 3);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((3, 3));
        let mut ii = Array::<f64, _>::zeros((3, 3));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 3);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_svd_rank_deficient() {
        // Generate a random rank-deficient (5x3 with rank 2) matrix.
        let a = {
            let mut a = Array::<f64, _>::zeros((5, 3));
            a.slice_mut(s![.., 0..2]).assign(&Array::<f64, _>::random((5,2), StandardNormal));
            {
                // Make it rank-deficient: last column = sum of first 2 columns.
                let (col0, col1, mut col2) = a.multi_slice_mut((s![.., 0], s![.., 1], s![.., 2]));
                col2.assign(&(&col0.view() + &col1.view()));
            }
            a
        };
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
        let r = {
            let mut r = Array::<f64, _>::zeros((5, 3));
            r.slice_mut(s![.., 0..2]).assign(&Array::<f64, _>::random((5,2), StandardNormal));
            {
                // Make it rank-deficient: last column = sum of first 2 columns.
                let (col0, col1, mut col2) = r.multi_slice_mut((s![.., 0], s![.., 1], s![.., 2]));
                col2.assign(&(&col0.view() + &col1.view()));
            }
            r
        };
        // Generate the imaginary part.
        let i = {
            let mut i =  Array::<f64, _>::zeros((5, 3));
            i.slice_mut(s![.., 0..2]).assign(&Array::<f64, _>::random((5,2), StandardNormal));
            {
                let (col0, col1, mut col2) = i.multi_slice_mut((s![.., 0], s![.., 1], s![.., 2]));
                col2.assign(&(&col0.view() + &col1.view()));
            }
            i
        };
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((5, 3));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv_svd().unwrap();
        // Rank of a should be 2.
        assert!(rank == 2);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((5, 3));
        let mut ii = Array::<f64, _>::zeros((5, 3));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 2);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }

    #[test]
    fn test_pinv_rank_deficient() {
        // Generate a random rank-deficient (5x3 with rank 2) matrix.
        let a = {
            let mut a = Array::<f64, _>::zeros((5, 3));
            a.slice_mut(s![.., 0..2]).assign(&Array::<f64, _>::random((5,2), StandardNormal));
            {
                // Make it rank-deficient: last column = sum of first 2 columns.
                let (col0, col1, mut col2) = a.multi_slice_mut((s![.., 0], s![.., 1], s![.., 2]));
                col2.assign(&(&col0.view() + &col1.view()));
            }
            a
        };
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
        let r = {
            let mut r = Array::<f64, _>::zeros((5, 3));
            r.slice_mut(s![.., 0..2]).assign(&Array::<f64, _>::random((5,2), StandardNormal));
            {
                // Make it rank-deficient: last column = sum of first 2 columns.
                let (col0, col1, mut col2) = r.multi_slice_mut((s![.., 0], s![.., 1], s![.., 2]));
                col2.assign(&(&col0.view() + &col1.view()));
            }
            r
        };
        // Generate the imaginary part.
        let i = {
            let mut i =  Array::<f64, _>::zeros((5, 3));
            i.slice_mut(s![.., 0..2]).assign(&Array::<f64, _>::random((5,2), StandardNormal));
            {
                let (col0, col1, mut col2) = i.multi_slice_mut((s![.., 0], s![.., 1], s![.., 2]));
                col2.assign(&(&col0.view() + &col1.view()));
            }
            i
        };
        // Zip parts together into a complex matrix.
        let a = {
            let mut a = Array::<Complex64, _>::zeros((5, 3));
            Zip::from(&mut a).and(&r).and(&i).apply(|a, r, i| *a = Complex64::new(*r, *i));
            a
        };
        // Do the pseudoinverse.
        let PseudoInverseOutput { pinv, rank } = a.pinv().unwrap();
        // Rank of a should be 2.
        assert!(rank == 2);
        // a.dot(&pinv).dot(&a) should equal a.
        let aa = a.dot(&pinv).dot(&a);
        // Check that the real parts are equal.
        let mut rr = Array::<f64, _>::zeros((5, 3));
        let mut ii = Array::<f64, _>::zeros((5, 3));
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        // Check that the imaginary parts are equal.
        assert!(i.abs_diff_eq(&ii, 1e-4));
        // Repeat for pinv_into().
        let PseudoInverseOutput { pinv, rank } = a.clone().pinv_into().unwrap();
        assert!(rank == 2);
        let aa = a.dot(&pinv).dot(&a);
        Zip::from(&aa).and(&mut rr).and(&mut ii).apply(|aa, rr, ii| {
            *rr = aa.re();
            *ii = aa.im();
        });
        assert!(r.abs_diff_eq(&rr, 1e-4));
        assert!(i.abs_diff_eq(&ii, 1e-4));
    }
}
