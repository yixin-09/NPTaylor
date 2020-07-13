#include "eft_patch.h"
static double array_idx_sf_bessel_j1_0[2] = {
4.493409058617158003e+00,
4.493409553914382570e+00,
};
static double array_cof_float_sf_bessel_j1_0[1][3] = {
{
4.834494391087783283e-02,
-2.172336282112216632e-01,
-7.218300729428198140e-18,
}
};
static double array_cof_err_sf_bessel_j1_0[1][3] = {
{
-1.905394200647906896e-18,
9.022415591735671687e-18,
6.878347149357537706e-34,
}
};
static double array_point_sf_bessel_j1_0[1] = {
4.493409457909064209e+00,
};
static double array_cofidx_sf_bessel_j1_0[1] = {
2.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_j1_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_j1_0[idx])&&(x<=array_idx_sf_bessel_j1_0[idx+1])){
         double point = array_point_sf_bessel_j1_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_j1_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_j1_0[idx],array_cof_err_sf_bessel_j1_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_j1_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_j1_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_j1(double x)
{
if(x<=4.493409553914383){
 return accuracy_improve_patch_of_gsl_sf_bessel_j1_0(x);
}
}
