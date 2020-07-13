#include "eft_patch.h"
static double array_idx_sf_bessel_y1_0[2] = {
2.798244479882426194e+00,
2.798502917020114111e+00,
};
static double array_cof_float_sf_bessel_y1_0[1][4] = {
{
1.210618901271392480e-03,
-1.202508915542180590e-01,
3.365084169183952811e-01,
9.865851509229451309e-18,
}
};
static double array_cof_err_sf_bessel_y1_0[1][4] = {
{
7.701702174358598064e-20,
4.601458874418102206e-18,
3.428607506654347762e-18,
8.218572898229961874e-35,
}
};
static double array_point_sf_bessel_y1_0[1] = {
2.798386045783887166e+00,
};
static double array_cofidx_sf_bessel_y1_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_y1_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_y1_0[idx])&&(x<=array_idx_sf_bessel_y1_0[idx+1])){
         double point = array_point_sf_bessel_y1_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_y1_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_y1_0[idx],array_cof_err_sf_bessel_y1_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_y1_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_y1_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_y1(double x)
{
if(x<=2.798502917020114){
 return accuracy_improve_patch_of_gsl_sf_bessel_y1_0(x);
}
}
