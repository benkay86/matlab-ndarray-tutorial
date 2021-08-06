//! Solve over- and under-determined linear systems using linear least squares
//! with an L2 norm.  Use the pseudoinverse to calculate t-statistics and
//! p-values.

extern crate blas_src;
use matlab_ndarray_tutorial::pseudoinverse::{PseudoInverse, PseudoInverseOutput};
use ndarray::{array, s, Array, Axis, Zip};
use ndarray_linalg::{LeastSquaresResult, LeastSquaresSvd};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use statrs::distribution::{ContinuousCDF, StudentsT};

fn main() {
    // Generate a full-rank linear least squares system.
    //
    // ```matlab
    // % Simulate 32 observations with random error.
    // n = 32;
    // x = ones(n,1);
    // x(:,2) = linspace(0, 3, n);
    // x(:,3) = normrnd(0, 1, n, 1);
    // b = [1; 2; 0]
    // y = x * b + normrnd(0, 1, n, 1);
    // ```
    let n = 32;
    let x = {
        let mut x = Array::<f64, _>::ones((n, 3));
        x.slice_mut(s![.., 1]).assign(&Array::linspace(0., 3., n));
        x.slice_mut(s![.., -1])
            .assign(&Array::<f64, _>::random(n, StandardNormal));
        x
    };
    let b = array![1., 2., 0.];
    println!("True beta coefficients:\n{:?}", b);
    let y = x.dot(&b) + &Array::<f64, _>::random(n, StandardNormal);
    // Solve the system using singular value decomposition.
    // This is roughly equivalent to Matlab's `mldivide` operator, although the
    // latter will actually end up using QR decomposition in this case (more on
    // that later).
    //
    // ```matlab
    // b = x\y
    // ```
    let LeastSquaresResult {
        rank, solution: b, ..
    } = x.least_squares(&y).unwrap();
    println!("Estimated beta coefficeints from least squares:\n{:?}", b);
    // We get the rank for free.
    println!("Rank of x is {:?}.", rank);
    // For fun, we can also compute the coefficient of multiple determination,
    // R2 (R squared).
    //
    // ```matlab
    // ybar = mean(y);
    // yhat = x*b;
    // sse = sum((y - yhat).^2); % sum square residuals
    // sst = sum((y - ybar).^2); % sum square total
    // r2 = 1 - sse/sst
    // ```
    let sse = {
        let yhat = x.dot(&b);
        Zip::from(&y).and(&yhat).fold(0., |sse, y, yhat| {
            let resid = y - yhat;
            sse + resid * resid
        })
    };
    let sst = {
        let ybar = y.mean().unwrap();
        y.fold(0., |sst, y| {
            let y0 = y - ybar;
            sst + y0 * y0
        })
    };
    let r2 = 1. - sse / sst;
    println!("Coefficient of determination:\n{:?}", r2);
    // In Matlab we could also solve the linear system by first using the pinv()
    // convenience function to compute the pseudoinverse, and then multiplying
    // by `y` to get `b`.
    //
    // ```matlab
    // x_pinv = pinv(x);
    // b = x_pinv * y;
    // ```
    //
    // Ndarray doesn't have a convenience method like Matlab's `pinv()`. to
    // compute the pseudoinverse.  The reasons for this are somewhat complex and
    // have to do with lapack not having a pseudoinverse algorithm. One way we
    // can get the pseudoinverse is by solving into an identity matrix.  This
    // uses lapack's dgesvd algorithm which is the same algorithm Matlab's
    // `pinv()` uses.  However, as explored further below, there are even more
    // efficient algorithms to consider.
    //
    // Note, if we weren't using `x` later in the tutorial, it would be more
    // memory efficient to use `least_squares_into()`.
    let LeastSquaresResult {
        solution: x_pinv, ..
    } = x.least_squares(&Array::<f64, _>::eye(n)).unwrap();
    let b = x_pinv.dot(&y);
    println!("Estimated beta coefficeints from pseudoinverse:\n{:?}", b);
    // Having the pseudoinverse, we can now compute t-values as well.
    // Recall the variance-covariance matrix of the estimates of b is given by
    // inv(x'*x).  Since the inverse of x'*x is guaranteed to exist, it is true
    // that inv(x'*x) = pinv(x'*x).  Then, using Greville's proof:
    // inv(x'*x) = pinv(x'*x) = pinv(x)*pinv(x') = pinv(x)*pinv(x)'
    // https://epubs.siam.org/doi/10.1137/1008107
    // The last step is true because the pseudoinverse operation commutes with
    // the complex conjugate (transpose).
    //
    // (Note also that we can compute the variance-covariance matrix directly
    // from the singular value decomposition as v*inv(s).^2*v')
    //
    // For t-values, we only need the diagonal of the variance-covariance
    // matrix.  For any matrix A we can efficiently compute diag(A*A') =
    // sum(A .* A, 2).  Now we can compute the t-values like this:
    //
    // ```matlab
    // % Mean squared error of regression.
    // df = size(x,1) - size(x,2) - 1;
    // mse = sum((y - x*b).^2) ./ df;
    // % Standard error of coefficients.
    // var_b = sum(x_pinv .* x_pinv, 2);
    // se = sqrt(var_b .* mse);
    // % t-values of coefficients.
    // t = b ./ se;
    // ```
    let df = (x.nrows() - x.ncols() - 1) as f64;
    let mse = sse / df;
    let var_b = x_pinv.fold_axis(Axis(1), 0., |sum, col| sum + col * col);
    let se = (var_b * mse).mapv(f64::sqrt);
    let t = &b / &se;
    println!("t-values of coefficeints:\n{:?}", t);
    // We can use the cumulative distribution function (CDF) to get a p-value.
    //
    // ```matlab
    // p = cdf('T', t, df)
    // ```
    let tdist = StudentsT::new(0., 1., df).unwrap();
    let p = t.mapv(|t| 1. - tdist.cdf(t));
    println!("p-values of coefficients:\n{:?}\n", p);

    // As remarked upon above, singular value decomposition using lapack's
    // dgesvd is not the most efficient way to obtain the pseudoinverse.  If the
    // matrix is full-rank it will be more efficient to use QR decomposition.
    // If the matrix is rank-deficient then it will be more efficient to use
    // lapack's dgesdd algorithm to perform singular value decomposition with
    // divide and conquer.  This approach is similar to Matlab's mldivide
    // operator, but will give us a pseudoinverse to work with.  See
    // <../pseudoinverse.rs> for an implementation with detailed comments.
    // Repeating the problem above with this faster algorithm.
    //
    // ```matlab
    // rank(x)
    // x_pinv = pinv(x);
    // b = x_pinv * y
    // ```
    println!("Using faster pinv:");
    let PseudoInverseOutput { pinv: x_pinv, rank } = x.pinv().unwrap();
    let b = x_pinv.dot(&y);
    println!("Rank of x is {:?}.", rank);
    println!("Estimated beta coefficeints from pseudoinverse:\n{:?}", b);
}
