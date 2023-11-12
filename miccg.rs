use rayon::prelude::*;

// Multiply a sparse laplacian matrix with a dense vector
// Takes advantage of symetry within the matrix
pub fn sparse_dot_vec(
    diag: &Vec<f64>,
    up: &Vec<f64>,
    right: &Vec<f64>,
    b: &Vec<f64>,
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    let mut result = vec![0.0; b.len() + cols];
    for i in 1..rows - 1 {
        for j in 1..cols - 1 {
            let gridindex = i * cols + j;
            let arrayindex = (i - 1) * (cols - 2) + j - 1;
            // main
            result[arrayindex] += diag[gridindex] * b[arrayindex]
                + up[gridindex] * b.get(arrayindex + cols - 2).unwrap_or(&0.0)
                + right[gridindex] * b.get(arrayindex + 1).unwrap_or(&0.0);

            // up symmetry
            result[arrayindex + cols - 2] += up[gridindex] * b[arrayindex];

            // right symmetry
            result[arrayindex + 1] += right[gridindex] * b[arrayindex];
        }
    }
    result.drain(b.len()..);
    result
}
// a * b element wise sum, just a dot product
fn vec_dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    return a.par_iter().zip(b).map(|(i, j)| j * i).sum();
}
// add vectors a and b element wise
fn add_vec(a: &Vec<f64>, b: &Vec<f64>, c: f64) -> Vec<f64> {
    return a.par_iter().zip(b).map(|(i, j)| i + j * c).collect();
}

// Applies the MIC preconditioner by doing forward and backward substitution with the preconditioner diagonal vector
// ie preconditioner M = F *E^-1 + E
// where E is the diagonal preconditioner and F is the strict lower diagonal of the matrix A
// This function solves the equation Mz = R and returns the solution z
fn apply_mic(
    r: &Vec<f64>,
    right: &Vec<f64>,
    up: &Vec<f64>,
    precon: &Vec<f64>,
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    let arraylen = (rows - 2) * (cols - 2);
    let mut q = vec![0.0; rows * cols];
    let mut z = vec![0.0; arraylen + cols];

    // Solve Lq = r
    for i in 0..(rows - 2) {
        for j in 0..(cols - 2) {
            let gridindex = j + 1 + (i + 1) * cols;
            let arrayindex = i * (cols - 2) + j;
            let t = r[arrayindex]
                - right[gridindex - 1] * precon[gridindex - 1] * q[gridindex - 1]
                - up[gridindex - cols] * precon[gridindex - cols] * q[gridindex - cols];
            q[gridindex] = t * precon[gridindex];
        }
    }
    //println!("Q:    {:?}", q);
    // Then Solve Ltz = q
    for i in (0..(rows - 2)).rev() {
        for j in (0..(cols - 2)).rev() {
            let gridindex = j + 1 + (i + 1) * cols;
            let arrayindex = i * (cols - 2) + j;
            let t = q[gridindex]
                - right[gridindex] * precon[gridindex] * z[arrayindex + 1]
                - up[gridindex] * precon[gridindex] * z[arrayindex + cols - 2];
            //println!("t   {}", t);
            z[arrayindex] = t * precon[gridindex];
        }
    }

    // return the resulting vector of correct length
    //println!("Z:    {:?}", z);
    z.drain(arraylen..);

    return z;
}

// Modified Incomplete Cholosky Congugate Gradient Descent
// Iteravely searches for solution to Ax = b where A is a sparse matrix and x and b are dense vectors
pub fn miccg(
    diag: &Vec<f64>,
    right: &Vec<f64>,
    up: &Vec<f64>,
    precon: &Vec<f64>,
    div: &Vec<f64>,
    old_guess: &Vec<f64>,
    rows: usize,
    cols: usize,
) -> Vec<f64> {
    //let mut p = vec![0.0; div.len()];
    let mut p = old_guess.clone();
    if &p == div {
        return p;
    }
    //let mut r = div.clone();
    let mut r = add_vec(div, &sparse_dot_vec(diag, up, right, &p, rows, cols), -1.);
    let mut z;
    let mut s = apply_mic(&r, right, up, precon, rows, cols);
    let mut sigma = vec_dot_product(&s, &r);
    let mut sigma2;
    let mut alpha;
    let tolerance = vec_dot_product(&r, &r) * 0.000001; //0.00001

    for i in 0..100 {
        z = sparse_dot_vec(diag, up, right, &s, rows, cols);
        alpha = sigma / vec_dot_product(&z, &s);

        p = add_vec(&p, &s, alpha);
        r = add_vec(&r, &z, -alpha);
        //println!("{:?}", p);

        let norm = vec_dot_product(&r, &r);
        //println!("{:?}", r);
        //println!("{}", norm);
        if norm <= tolerance {
            //println!("{}", i);
            return p;
        }
        z = apply_mic(&r, right, up, precon, rows, cols);
        sigma2 = vec_dot_product(&z, &r);
        s = add_vec(&z, &s, sigma2 / sigma);
        sigma = sigma2;
    }
    println!("{}", 100);

    p
}
