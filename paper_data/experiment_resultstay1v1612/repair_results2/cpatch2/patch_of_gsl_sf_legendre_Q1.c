#include "eft_patch.h"
static double array_idx_sf_legendre_Q1_0[2] = {
8.334944426981359955e-01,
8.336186554332131182e-01,
};
static double array_cof_float_sf_legendre_Q1_0[1][5] = {
{
1.719248961510101594e+02,
3.910126647809942568e+01,
1.073687777945740152e+01,
3.931008032372373950e+00,
2.131319626955773201e-16,
}
};
static double array_cof_err_sf_legendre_Q1_0[1][5] = {
{
5.256866329216981039e-15,
-2.599613332624613048e-15,
-2.391311754464641992e-16,
-1.507644162908361016e-16,
-9.164198777980748770e-33,
}
};
static double array_point_sf_legendre_Q1_0[1] = {
8.335565596009647527e-01,
};
static double array_cofidx_sf_legendre_Q1_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_legendre_Q1_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_legendre_Q1_0[idx])&&(x<=array_idx_sf_legendre_Q1_0[idx+1])){
         double point = array_point_sf_legendre_Q1_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_legendre_Q1_0[idx];
         eft_tay1v(array_cof_float_sf_legendre_Q1_0[idx],array_cof_err_sf_legendre_Q1_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_legendre_Q1_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_legendre_Q1_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_legendre_Q1(double x)
{
if(x<=0.8336186554332131){
 return accuracy_improve_patch_of_gsl_sf_legendre_Q1_0(x);
}
}
