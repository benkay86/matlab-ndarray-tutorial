//! For loops and more sophisticated forms of iteration.
//!
//! See also <fold.rs>.

extern crate blas_src;
use ndarray::{array, Array, Axis, Zip};

fn main() {
    // As with Matlab, users of ndarray should prefer so-called "vectorized"
    // mathematical operations, which call into the highly-optimized BLAS/LAPACK
    // backends.  But sometimes we need to do something with each element of an
    // array that isn't possible or practical with simple linear algebra.
    //
    // In Matlab we use the for loop to iterate over the indices of an array.
    // We can do the same thing in Rust, but as we'll see shortly there are more
    // efficient alternatives.
    //
    // ```matlab
    // mat = [1,2,3; 4,5,6]
    // sum = 0;
    // for x=1:2
    //     for y = 1:3
    //         fprintf(1, "%f ", mat(x,y));
    //         sum = sum + mat(x,y);
    //     end
    // end
    // ```
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("mat = \n{:?}", mat);
    let mut sum = 0.;
    for x in 0..2 {
        for y in 0..3 {
            print!("{:?} ", mat[[x, y]]);
            sum += mat[[x, y]];
        }
    }
    println!("\nsum = {:?}", sum);

    // The problem with this form of iteration is that the indices and ranges
    // may be dynamic, and the for loop may end early (i.e. with a `break`
    // statement), so the compiler has a difficult time proving *in general*
    // that `mat[[x,y]]` indexes a valid element in the array.
    //
    // For example, uncomment the following line to get a panic at runtime:
    // println!("This will panic: {:?}", mat[[2,3]]);
    //
    // To make sure our program panics instead of having undefined behavior, the
    // compiler checks the array bounds with each call of the indexing []
    // operator.  All that extra bound checking adds up to suboptimal
    // performance when done repeatedly in a tight loop.
    //
    // Rust's iterators solve this problem by proving to the compiler at compile
    // time that we will never access an index that is out-of-bounds.  The more
    // correct way to right the above loop using iterators is:
    sum = 0.;
    for el in mat.iter() {
        print!("{:?} ", el);
        sum += el;
    }
    println!("\nsum = {:?}", sum);
    // Or we can write it almost equivalently using iterator combinators.  The
    // disadvantage of the `for_each()` combinator is that we can't `break`, and
    // if the iterator is fallible we have to remember to use `try_for_each()`
    // instead.  The advantage will come later when we start to use parallel
    // iterators.
    sum = 0.;
    mat.iter().for_each(|el| {
        print!("{:?} ", el);
        sum += el;
    });
    println!("\nsum = {:?}", sum);
    // Ndarray's `iter()` visits the elements of the array in logical order,
    // that is, the outermost dimension (column) varies the fastest.  There is
    // also a `for_each()` we can call directly on the array which visits the
    // elements in an arbitrary (and possibly more efficient) order.  The sum
    // will be the same, but the order in which the elements are printed out may
    // be different.
    sum = 0.;
    mat.for_each(|el| {
        print!("{:?} ", el);
        sum += el;
    });
    println!("\nsum = {:?}\n", sum);

    // Sometimes we actually need to know the indices of of each element of the
    // array as we iterate over it.  As before, we will visit the elements in
    // logical order.
    mat.indexed_iter().for_each(|((x, y), el)| {
        println!("mat[[{:?},{:?}]] = {:?}", x, y, el);
    });

    // As we saw in <math.rs>, we can use the `map()` family of functions to
    // efficiently iterate over an array to create a new array of the same size,
    // or to modify the elements of the array in place.  Obviously there is a
    // more efficient way to get an array of ones, but as a didactic example:
    let mut mat = Array::<f64, _>::zeros((2, 3));
    println!("\nzeros:\n{:?}", mat);
    mat.mapv_inplace(|el| el + 1.);
    println!("ones:\n{:?}", mat);
    // We can also modify in place while doing something with the index, here
    // replacing each element of the array with the sum of its x and y indices.
    mat.indexed_iter_mut()
        .for_each(|((x, y), el)| *el = (x + y) as f64);
    println!("sum of indices:\n{:?}\n", mat);

    // We don't have to iterate over just the elements of an array.  We can also
    // iterate over slices and views!  We can use `rows()` and `columns()`.
    // There are `_mut()` versions of each of these.
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("mat = \n{:?}", mat);
    println!("Rows:");
    mat.rows()
        .into_iter()
        .for_each(|row| println!("{:?}", row));
    // For higher-dimensional arrays, the more generic version of this is:
    println!("Columns (lanes):");
    mat.lanes(Axis(0))
        .into_iter()
        .for_each(|col| println!("{:?}", col));
    // Or if we want to iterate over high-dimensional subviews along an axis:
    println!("Columns (axis iterator):");
    mat.axis_iter(Axis(1)).for_each(|col| println!("{:?}", col));
    // As with individual elements, we can access the index of each row/column
    // by constructing an indexed iterator.  In this case we use the
    // `enumerate()` method:
    println!("Columns with colum number:");
    mat.axis_iter(Axis(1))
        .enumerate()
        .for_each(|(index, col)| println!("col num {}: {:?}", index, col));

    // Often we want to iterate over two or more arrays in lockstep.  We can do
    // this with the `Zip` helper.  Here we assign the elements of one array
    // into the elements of another using `Zip`.  The `Zip` struct implements
    // `IntoIterator` meaning we can call `for_each()` on it.
    let mut mat2 = Array::<f64, _>::zeros((2, 3)); // allocate space
    Zip::from(&mut mat2)
        .and(&mat)
        .for_each(|el2, &el| *el2 = el);
    println!("\nmat2 = mat = \n{:?}", mat2);
    // If desired, we can keep the indices from *one* of the arrays being zipped
    // together with `Zip::indexed()`.
    Zip::indexed(&mut mat2)
        .and(&mat)
        .for_each(|(x, y), el2, &el| *el2 = el + (x + y) as f64);
    println!("mat2 = mat + x + y = \n{:?}", mat2);
    // We zip together arrays and slices in interesting ways.  Here we compute
    // a vector that is the sum of each column.
    let mut col_sums = Array::<f64, _>::zeros(3);
    Zip::from(&mut col_sums)
        .and(mat.columns())
        .for_each(|col_sum, col| *col_sum = col.sum());
    println!("col_sums =\n{:?}", col_sums);
    // Wow, that was fun!  Don't forget how to do it the easy way:
    col_sums = mat.sum_axis(Axis(0));
    println!("col_sums =\n{:?}", col_sums);
}
