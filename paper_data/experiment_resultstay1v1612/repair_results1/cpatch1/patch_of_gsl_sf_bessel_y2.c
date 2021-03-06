#include "eft_patch.h"
static double array_idx_sf_bessel_y2_0[2] = {
7.451595452337139491e+00,
7.451624676148651538e+00,
};
static double array_cof_float_sf_bessel_y2_0[1][4] = {
{
1.705396587759204496e-02,
1.751753881269556626e-02,
-1.305338685169504243e-01,
-2.058941623624766705e-17,
}
};
static double array_cof_err_sf_bessel_y2_0[1][4] = {
{
3.205108770238536943e-19,
1.663063300958386343e-18,
1.336150241361363460e-17,
-7.108532374045325577e-34,
}
};
static double array_point_sf_bessel_y2_0[1] = {
7.451610064214503559e+00,
};
static double array_cofidx_sf_bessel_y2_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_y2_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_y2_0[idx])&&(x<=array_idx_sf_bessel_y2_0[idx+1])){
         double point = array_point_sf_bessel_y2_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_y2_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_y2_0[idx],array_cof_err_sf_bessel_y2_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_y2_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_y2_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_y2(double x)
{
if(x<=7.4516246761486515){
 return accuracy_improve_patch_of_gsl_sf_bessel_y2_0(x);
}
}
