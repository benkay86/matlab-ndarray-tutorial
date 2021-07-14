//! Linear least squares (lls) related functionality.
//!
//! See <bin/least_squares.rs> for a more detailed explanation of how to solve
//! linear least squares system with Rust and ndarray.
//!
//! This module provides the [`SolvedModel`] struct for caching the solution of
//! a linear least squares system.  It is useful in the setting of permutation
//! testing, see <bin/parfor.rs> for an example.

use ndarray::{Array, ArrayBase, Data, Dim, Dimension, Axis, Ix, OwnedRepr, ScalarOperand, Zip};
use ndarray_linalg::{Lapack, Scalar};
use ndarray_linalg::error::Result;
use crate::pseudoinverse::{PseudoInverse, PseudoInverseOutput};

/// Operation for computing t-values associated with a solved model.
/// Generic over 1d and 2d matrices of observations, `y`.
pub trait TValuesWith<YData: Data, YDim: Dimension> {
    type Output;
    /// Compute t-values for a solved model with observations `y`.
    /// See [`SolvedModel::tvalues_with()`].
    fn tvalues_with(&self, y: &ArrayBase<YData, YDim>) -> Self::Output;
}
// General case for 2-dimensional y.
impl<F, YData, XData, XPinvData, VarBData> TValuesWith<YData, Dim<[Ix; 2]>> for SolvedModel<XData, XPinvData, VarBData>
where
    F: Lapack + Scalar + ScalarOperand,
    YData: Data<Elem = F>,
    XData: Data<Elem = F>,
    XPinvData: Data<Elem = F>,
    VarBData: Data<Elem = F>,
{
    type Output = Array<F, Dim<[Ix; 2]>>;
    fn tvalues_with(&self, y: &ArrayBase<YData, Dim<[Ix; 2]>>) -> Self::Output {
        // Compute beta values using cached pseudoinverse.
        let b = self.x_pinv.dot(y);
        // Compute sum squared error of model.
        let sse = {
            let yhat = self.x.dot(&b); // predicted y
            Zip::from(y).and(&yhat).fold(F::zero(), |sse, &y, &yhat| {
                let resid = y - yhat; // resudials
                sse + resid * resid // sum square of residuals
            })
        };
        // Mean squared error.
        let mse = sse / F::from_real(F::real(self.df));
        // Standard error of estimates.
        let se = (&self.var_b * mse).mapv(F::sqrt);
        // T-values.
        b / se
    }
}
// Also handle 1-dimensional y.
impl<F, YData, XData, XPinvData, VarBData> TValuesWith<YData, Dim<[Ix; 1]>> for SolvedModel<XData, XPinvData, VarBData>
where
    F: Lapack + Scalar + ScalarOperand,
    YData: Data<Elem = F>,
    XData: Data<Elem = F>,
    XPinvData: Data<Elem = F>,
    VarBData: Data<Elem = F>,
{
    type Output = Array<F, Dim<[Ix; 1]>>;
    fn tvalues_with(&self, y: &ArrayBase<YData, Dim<[Ix; 1]>>) -> Self::Output {
        // Broadcast to 2d array.
        let y = y.broadcast((1, y.len())).unwrap().reversed_axes();
        // Delegate to the more general 2d version to compute t-values.
        let t = self.tvalues_with(&y);
        // Pare back down to a 1d array.
        t.remove_axis(Axis(1))
    }
}

/// Helper trait to make compiler error messages more scrutable.
/// Implemented for non-zero dimensions (i.e. # of axes) <= 2.
pub trait DimLessEq2: Dimension { }
impl DimLessEq2 for Dim<[Ix; 1]> { }
impl DimLessEq2 for Dim<[Ix; 2]> { }

// Cache of computed values for solved model.
pub struct SolvedModel<XData, XPinvData, VarBData>
where
    XData: Data,
    XPinvData: Data,
    VarBData: Data,
{
    // Design matrix.
    pub x: ArrayBase<XData, Dim<[Ix; 2]>>,
    // Pseudoinverse of design matrix.
    pub x_pinv: ArrayBase<XPinvData, Dim<[Ix; 2]>>,
    // Variance of parameter estimates.
    pub var_b: ArrayBase<VarBData, Dim<[Ix; 1]>>,
    // Degrees of freedom.
    pub df: usize,
}
impl<F, XData, XPinvData, VarBData> SolvedModel<XData, XPinvData, VarBData>
where
    F: Lapack + Scalar + ScalarOperand,
    XData: Data<Elem = F>,
    XPinvData: Data<Elem = F>,
    VarBData: Data<Elem = F>,
{
    /// Compute t-values for a solved model with observations `y`.
    /// Each row in `y` is an observation.
    /// If `y` is 2d, then each column is an independent observed variable.
    /// If `y` is 1-dimensional returns a vector of t-values, one for each
    /// column in the design matrix `x`.
    /// If `y` is 2-dimensional returns a matrix of t-values.
    /// Each column of `t` corresponds to a column of `y`.
    /// Each row of `y` corresponds to a column in the design matrix `x`.
    pub fn tvalues_with<YData, YDim>(&self, y: &ArrayBase<YData, YDim>) -> Array<F, YDim>
    where
        YData: Data<Elem = F>,
        YDim: DimLessEq2,
        Self: TValuesWith<YData, YDim, Output=Array<F, YDim>>,
    {
        // Delegate to the `TValuesWith` trait.
        <Self as TValuesWith<YData, YDim>>::tvalues_with(self, y)
    }
}

/// Pre-solve a model speficied by the design matrix `x` where the rows in
/// `x` are observations and colums are predictor variables.
pub fn pre_solve<F, XData> (x: ArrayBase<XData, Dim<[Ix; 2]>>) -> Result<SolvedModel<XData, OwnedRepr<F>, OwnedRepr<F>>>
where
    F: Lapack + Scalar,
    XData: Data<Elem = F>,
{
    let PseudoInverseOutput{pinv: x_pinv, ..} = x.pinv()?;
    let var_b = x_pinv.fold_axis(Axis(1), F::zero(), |&sum, &col| sum + col * col);
    let df = x.nrows() - x.ncols() - 1;
    Ok(SolvedModel {
        x,      // design matrix
        x_pinv, // pseudoinverse of design matrix
        var_b,  // variance of estimated parameters
        df      // degrees of freedom
    })
}
