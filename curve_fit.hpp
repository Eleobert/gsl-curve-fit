#pragma once

#include "utils.hpp"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>


// All of this template metaprograming just because we cant pass lambdas as function 
// pointer argument.

// For information about non-linear least-squares fit with gsl
// see https://www.gnu.org/software/gsl/doc/html/nls.html

template<typename C1>
struct fit_data
{
    const arma::vec& t;
    const arma::vec& y;
    // the actual function to be fitted
    C1 f;
};

template <typename R, typename ... Types> constexpr size_t getArgumentCount( R(*f)(Types ...))
{
   return sizeof...(Types);
}

template<typename FitData, int n_params>
int internal_f(const gsl_vector* x, void* params, gsl_vector *f)
{
    auto* d  = static_cast<FitData*>(params);
    // Convert the parameter values from gsl_vector (in x) into std::tuple
    auto init_args = [x](int index)
    {
        return gsl_vector_get(x, index);
    };
    auto parameters = gen_tuple<n_params>(init_args);

    // Calculate the error for each...
    for (int i = 0; i < d->t.n_elem; ++i)
    {
        double ti = d->t[i];
        double yi = d->y[i];
        auto func = [ti, &d](auto ...xs)
        {
            // call the actual function to be fitted
            return d->f(ti, xs...);
        };
        auto y = std::apply(func, parameters);
        gsl_vector_set(f, i, yi - y);
    }
    return GSL_SUCCESS;
}

using func_f_type   = int (*) (const gsl_vector*, void*, gsl_vector*);
using func_df_type  = int (*) (const gsl_vector*, void*, gsl_matrix*);
using func_fvv_type = int (*) (const gsl_vector*, const gsl_vector *, void *, gsl_vector *);


auto internal_make_gsl_vector_ptr(const std::vector<double>& vec) -> gsl_vector*;


auto internal_solve_system(gsl_vector* initial_params, gsl_multifit_nlinear_fdf *fdf,
             gsl_multifit_nlinear_parameters *params) -> std::vector<double>;

template<typename C1>
auto curve_fit_impl(func_f_type f, func_df_type df, func_fvv_type fvv, gsl_vector* initial_params, fit_data<C1>& fd) -> std::vector<double>
{
    assert(fd.t.n_elem == fd.y.n_elem);

    auto fdf = gsl_multifit_nlinear_fdf();
    auto fdf_params = gsl_multifit_nlinear_default_parameters();
    
    fdf.f   = f;
    fdf.df  = df;
    fdf.fvv = fvv;
    fdf.n   = fd.t.n_elem;
    fdf.p   = initial_params->size;
    fdf.params = &fd;

    // "This selects the Levenberg-Marquardt algorithm with geodesic acceleration."
    fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel;
    return internal_solve_system(initial_params, &fdf, &fdf_params);
}


/**
 * Performs a non-linear least-squares fit.
 * 
 * @param f a function of type double (double x, double c1, double c2, ..., double cn)
 * where  c1, ..., cn are the coefficients to be fitted.
 * @param initial_params intial guess for the parameters. The size of the array must to 
 * be equal to the number of coefficients to be fitted.
 * @param x the idependent data.
 * @param y the dependent data, must to have the same size as x.
 * @return std::vector<double> with the computed coefficients
 */
template<typename Callable>
auto curve_fit(Callable f, const std::vector<double>& initial_params, const arma::vec& x, const arma::vec& y) -> std::vector<double>
{
    constexpr auto n = decltype(n_params(std::function(f)))::n_args - 1;
    assert(initial_params.size() == n);

    auto params = internal_make_gsl_vector_ptr(initial_params);
    auto fd = fit_data<Callable>{x, y, f};
    return curve_fit_impl(internal_f<decltype(fd), n>, nullptr, nullptr, params,  fd);
}