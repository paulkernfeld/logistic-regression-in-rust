//! This package contains a toy implementation of logistic regression in Rust.
#[cfg(test)]
#[macro_use(array)]
extern crate ndarray;
#[cfg(not(test))]
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate spectral;

use ndarray::prelude::*;

use MinimizeInnerResult::*;

// This value is stolen from scipy.optimize
// https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
const FTOL: f64 = 2.220446049250313e-09;

// Wikipedia says Armijo used 0.5 in 1966
const ARMIJO_GOLDSTEIN_CONTROL: f64 = 0.5;

fn sigmoid(a: &f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}

fn norm_l2(a_s: &Array1<f64>) -> f64 {
    a_s.fold(0.0, |b, a| b + a * a)
}

pub enum MinimizeInnerResult {
    ArmijoGoldsteinFailure,
    Success {
        w: Array1<f64>,
        f: f64,
        g: Array1<f64>,
    },
}

fn minimize_inner<FG>(w_init: Array1<f64>, fg: FG, epsilon: f64) -> MinimizeInnerResult
where
    FG: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut w = w_init;

    let (mut f_previous, mut g_previous) = fg(&w);

    loop {
        w.scaled_add(-epsilon, &g_previous);
        let (f, g) = fg(&w);

        let expected_decrease = epsilon * norm_l2(&g);
        let actual_decrease = f_previous - f;
        if actual_decrease < expected_decrease * ARMIJO_GOLDSTEIN_CONTROL {
            return ArmijoGoldsteinFailure;
        }
        if actual_decrease < FTOL {
            return Success { w, f, g };
        }
        f_previous = f;
        g_previous = g;
    }
}

/// Let f(w) be a function that returns a scalar, given vector w.
/// g returns the gradient of f at a given point w'.
/// This function finds a value of w that minimizes f(w).
///
/// fg returns the value of f and g at a given point
pub fn minimize<FG>(w_init: Array1<f64>, fg: FG) -> Array1<f64>
where
    FG: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{

    // Start with a large step size and decrease it until we either succeed or fail
    for i in 0..20 {
        let epsilon = 2.0_f64.powi(-i);

        if let Success { w, .. } = minimize_inner(w_init.clone(), &fg, epsilon) {
            return w;
        }
    }
    panic!("Even a very small value of epsilon didn't work")
}

pub struct LogisticRegressionResult {
    pub ws: Array1<f64>,
    pub yhats: Array1<f64>,
}

pub fn predict(ws: &Array1<f64>, xs: &Array2<f64>) -> Array1<f64> {
    xs.dot(ws).t().map(sigmoid)
}

pub fn logistic_regression(xs: Array2<f64>, ys: Array1<f64>) -> LogisticRegressionResult {
    let (n_data_points, n_features) = xs.dim();
    assert_eq!(ys.dim(), n_data_points);
    let w_init = Array1::zeros(n_features);
    let fg = |w: &Array1<f64>| {
        let yhats = predict(w, &xs);
        let f = -ys.iter()
            .zip(yhats.iter())
            .map(|(y, yhat)| y * yhat.ln() + (1.0 - y) * (1.0 - yhat).ln())
            .sum::<f64>();
        let g = (&yhats - &ys).dot(&xs);
        (f, g)
    };
    let ws = minimize(w_init, fg);
    let yhats = predict(&ws, &xs);
    LogisticRegressionResult { ws, yhats }
}

#[cfg(test)]
mod test {
    use ndarray_rand::RandomExt;
    use rand::distributions::Normal;
    use spectral::prelude::*;
    use super::*;

    /// In this test, the data are perfectly linearly separable. This means that we should be able
    /// to recover the test labels almost perfectly.
    #[test]
    fn test_1() {
        let xs = array![[-1.0], [1.0]];
        let ys = array![0.0, 1.0];
        let result = logistic_regression(xs, ys);
        assert_that(&result.yhats[0]).is_close_to(0.0, 1e-4);
        assert_that(&result.yhats[1]).is_close_to(1.0, 1e-4);
    }

    #[test]
    fn test_2() {
        let ws_true = array![-1.0, 0.0, 1.0];

        let xs = Array::random((100, 3), Normal::new(0., 1.));
        // These ys are between 0 and 1, but I think that is okay.
        let ys = predict(&ws_true, &xs);
        let result = logistic_regression(xs, ys);
        assert_that(&result.ws[0]).is_close_to(-1.0, 1e-4);
        assert_that(&result.ws[1]).is_close_to(0.0, 1e-4);
        assert_that(&result.ws[2]).is_close_to(1.0, 1e-4);
    }
}
