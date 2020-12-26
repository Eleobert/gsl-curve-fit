#include "curve_fit.hpp"


auto internal_solve_system(gsl_vector* initial_params, gsl_multifit_nlinear_fdf *fdf,
             gsl_multifit_nlinear_parameters *params) -> std::vector<double>
{
  // This specifies a trust region method
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  const size_t max_iter = 200;
  const double xtol = 1.0e-8;
  const double gtol = 1.0e-8;
  const double ftol = 1.0e-8;

  auto *work = gsl_multifit_nlinear_alloc(T, params, fdf->n, fdf->p);
  int info;

  // initialize solver
  gsl_multifit_nlinear_init(initial_params, fdf, work);
  //iterate until convergence
  gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol, nullptr, nullptr, &info, work);

  // result will be stored here
  gsl_vector * y    = gsl_multifit_nlinear_position(work);
  auto result = std::vector<double>(initial_params->size);

  for(int i = 0; i < result.size(); i++)
  {
    result[i] = gsl_vector_get(y, i);
  }

  auto niter = gsl_multifit_nlinear_niter(work);
  auto nfev  = fdf->nevalf;
  auto njev  = fdf->nevaldf;
  auto naev  = fdf->nevalfvv;

  // nfev - number of function evaluations
  // njev - number of Jacobian evaluations
  // naev - number of f_vv evaluations
  //logger::debug("curve fitted after ", niter, " iterations {nfev = ", nfev, "} {njev = ", njev, "} {naev = ", naev, "}");

  gsl_multifit_nlinear_free(work);
  gsl_vector_free(initial_params);
  return result;
}

auto internal_make_gsl_vector_ptr(const std::vector<double>& vec) -> gsl_vector*
{
    auto* result = gsl_vector_alloc(vec.size());
    int i = 0;
    for(const auto e: vec)
    {
        gsl_vector_set(result, i, e);
        i++;
    }
    return result;
}