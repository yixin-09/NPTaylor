#include "eft_patch.h"
static double array_idx_sf_legendre_P2_0[2] = {
5.773458651565938071e-01,
5.773546740058054239e-01,
};
static double array_cof_float_sf_legendre_P2_0[1][3] = {
{
1.500000000000000000e+00,
1.732050807568877193e+00,
-5.793758576800781262e-17,
}
};
static double array_cof_err_sf_legendre_P2_0[1][3] = {
{
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
}
};
static double array_point_sf_legendre_P2_0[1] = {
5.773502691896257311e-01,
};
static double array_cofidx_sf_legendre_P2_0[1] = {
2.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_legendre_P2_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_legendre_P2_0[idx])&&(x<=array_idx_sf_legendre_P2_0[idx+1])){
         double point = array_point_sf_legendre_P2_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_legendre_P2_0[idx];
         eft_tay1v(array_cof_float_sf_legendre_P2_0[idx],array_cof_err_sf_legendre_P2_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_legendre_P2_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_legendre_P2_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_legendre_P2(double x)
{
if(x<=0.5773546740058054){
 return accuracy_improve_patch_of_gsl_sf_legendre_P2_0(x);
}
}
