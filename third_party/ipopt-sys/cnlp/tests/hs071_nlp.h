#ifndef __HS071_NLP_HPP__
#define __HS071_NLP_HPP__

#include "../src/c_api.h"

/** Example of interacing with CNLP from C/C++.
 *  This is imlements the canonical problem 71 of the
 *  Hock-Schittkowski test suite used by Ipopt itself. 
 *
 * Problem hs071 looks like this
 *
 *     min   x1*x4*(x1 + x2 + x3)  +  x3
 *     s.t.  x1*x2*x3*x4                   >=  25
 *           x1**2 + x2**2 + x3**2 + x4**2  =  40
 *           1 <=  x1,x2,x3,x4  <= 5
 *
 *     Starting point:
 *        x = (1, 5, 5, 1)
 *
 *     Optimal solution:
 *        x = (1.00000000, 4.74299963, 3.82114998, 1.37940829)
 *
 *
 */
CNLP_Bool hs071_sizes(CNLP_Index* n, CNLP_Index* m, CNLP_Index* nnz_jac_g,
                      CNLP_Index* nnz_h_lag, CNLP_UserDataPtr user_data);

CNLP_Bool hs071_bounds(CNLP_Index n, CNLP_Number* x_l, CNLP_Number* x_u,
                       CNLP_Index m, CNLP_Number* g_l, CNLP_Number* g_u,
                       CNLP_UserDataPtr user_data);

CNLP_Bool hs071_init(CNLP_Index n, CNLP_Bool init_x, CNLP_Number* x,
                     CNLP_Bool init_z, CNLP_Number* z_L, CNLP_Number* z_U,
                     CNLP_Index m, CNLP_Bool init_lambda,
                     CNLP_Number* lambda, CNLP_UserDataPtr user_data);

CNLP_Bool hs071_eval_f(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
                       CNLP_Number* obj_value, CNLP_UserDataPtr user_data);

CNLP_Bool hs071_eval_grad_f(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
                            CNLP_Number* grad_f, CNLP_UserDataPtr user_data);

CNLP_Bool hs071_eval_g(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x, CNLP_Index m,
                       CNLP_Number* g, CNLP_UserDataPtr user_data);

CNLP_Bool hs071_eval_jac_g(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
                           CNLP_Index m, CNLP_Index nele_jac, CNLP_Index* iRow,
                           CNLP_Index *jCol, CNLP_Number* values, CNLP_UserDataPtr user_data);

CNLP_Bool hs071_eval_h(CNLP_Index n, const CNLP_Number* x, CNLP_Bool new_x,
                  CNLP_Number obj_factor, CNLP_Index m, const CNLP_Number* lambda,
                  CNLP_Bool new_lambda, CNLP_Index nele_hess, CNLP_Index* iRow,
                  CNLP_Index* jCol, CNLP_Number* values, CNLP_UserDataPtr user_data);

CNLP_Bool hs071_intermediate_callback(enum CNLP_AlgorithmMode alg_mod,
                                 CNLP_Index iter_count,
                                 CNLP_Number obj_value,
                                 CNLP_Number inf_pr,
                                 CNLP_Number inf_du,
                                 CNLP_Number mu,
                                 CNLP_Number d_norm,
                                 CNLP_Number regularization_size,
                                 CNLP_Number alpha_du,
                                 CNLP_Number alpha_pr,
                                 CNLP_Index ls_trials,
                                 CNLP_Index x_count,
                                 const CNLP_Number* x,
                                 CNLP_Index s_count,
                                 const CNLP_Number* s,
                                 CNLP_Index y_c_count,
                                 const CNLP_Number* y_c,
                                 CNLP_Index y_d_count,
                                 const CNLP_Number* y_d,
                                 CNLP_Index z_l_count,
                                 const CNLP_Number* z_l,
                                 CNLP_Index z_u_count,
                                 const CNLP_Number* z_u,
                                 CNLP_Index v_l_count,
                                 const CNLP_Number* v_l,
                                 CNLP_Index v_u_count,
                                 const CNLP_Number* v_u,
                                 CNLP_Index kkt_x_stationarity_count,
                                 const CNLP_Number* kkt_x_stationarity,
                                 CNLP_Index kkt_slack_stationarity_count,
                                 const CNLP_Number* kkt_slack_stationarity,
                                 CNLP_Index kkt_equality_residual_count,
                                 const CNLP_Number* kkt_equality_residual,
                                 CNLP_Index kkt_inequality_residual_count,
                                 const CNLP_Number* kkt_inequality_residual,
                                 CNLP_Index kkt_slack_complementarity_count,
                                 const CNLP_Number* kkt_slack_complementarity,
                                 CNLP_Index kkt_slack_sigma_count,
                                 const CNLP_Number* kkt_slack_sigma,
                                 CNLP_Index kkt_slack_distance_count,
                                 const CNLP_Number* kkt_slack_distance,
                                 CNLP_UserDataPtr user_data);

#endif
