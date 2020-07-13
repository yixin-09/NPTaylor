#include "eft_patch.h"
static double array_idx_sf_bessel_Y1_0[2] = {
3.061827987652972283e+01,
3.061829310675407712e+01,
};
static double array_cof_float_sf_bessel_Y1_0[1][3] = {
{
2.354246780882451959e-03,
-1.441660048181650500e-01,
-1.524456280251315065e-17,
}
};
static double array_cof_err_sf_bessel_Y1_0[1][3] = {
{
-2.114968789682136710e-19,
1.240234918529896909e-17,
-2.204723241223432954e-34,
}
};
static double array_point_sf_bessel_Y1_0[1] = {
3.061828649164111482e+01,
};
static double array_cofidx_sf_bessel_Y1_0[1] = {
2.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_Y1_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_Y1_0[idx])&&(x<=array_idx_sf_bessel_Y1_0[idx+1])){
         double point = array_point_sf_bessel_Y1_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_Y1_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_Y1_0[idx],array_cof_err_sf_bessel_Y1_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_Y1_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_Y1_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_Y1(double x)
{
if(x<=30.618293106754077){
 return accuracy_improve_patch_of_gsl_sf_bessel_Y1_0(x);
}
}
