#include "eft_patch.h"
static double array_idx_sf_legendre_P3_0[2] = {
7.745152762047033956e-01,
7.746780366285308528e-01,
};
static double array_cof_float_sf_legendre_P3_0[1][4] = {
{
2.500000000000000000e+00,
5.809475019311125976e+00,
3.000000000000000444e+00,
8.172618520478209880e-17,
}
};
static double array_cof_err_sf_legendre_P3_0[1][4] = {
{
0.000000000000000000e+00,
-4.440892098500626162e-16,
-1.275650556028785879e-16,
-5.062998230187522610e-33,
}
};
static double array_point_sf_legendre_P3_0[1] = {
7.745966692414834043e-01,
};
static double array_cofidx_sf_legendre_P3_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_legendre_P3_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_legendre_P3_0[idx])&&(x<=array_idx_sf_legendre_P3_0[idx+1])){
         double point = array_point_sf_legendre_P3_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_legendre_P3_0[idx];
         eft_tay1v(array_cof_float_sf_legendre_P3_0[idx],array_cof_err_sf_legendre_P3_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_legendre_P3_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_legendre_P3_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_legendre_P3(double x)
{
if(x<=0.7746780366285309){
 return accuracy_improve_patch_of_gsl_sf_legendre_P3_0(x);
}
}
