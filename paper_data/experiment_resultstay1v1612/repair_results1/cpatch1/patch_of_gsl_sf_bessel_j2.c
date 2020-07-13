#include "eft_patch.h"
static double array_idx_sf_bessel_j2_0[2] = {
9.095002495415849708e+00,
9.095020165649900790e+00,
};
static double array_cof_float_sf_bessel_j2_0[1][4] = {
{
-1.538031433374267669e-02,
-1.186812675586828777e-02,
1.079407473161516778e-01,
-3.170877217559426233e-18,
}
};
static double array_cof_err_sf_bessel_j2_0[1][4] = {
{
-6.425390713504469186e-19,
6.033789159131728876e-20,
-6.835687842595502324e-19,
-8.082251019941126990e-35,
}
};
static double array_point_sf_bessel_j2_0[1] = {
9.095011330476355127e+00,
};
static double array_cofidx_sf_bessel_j2_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_j2_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_j2_0[idx])&&(x<=array_idx_sf_bessel_j2_0[idx+1])){
         double point = array_point_sf_bessel_j2_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_j2_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_j2_0[idx],array_cof_err_sf_bessel_j2_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_j2_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_j2_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_j2(double x)
{
if(x<=9.0950201656499){
 return accuracy_improve_patch_of_gsl_sf_bessel_j2_0(x);
}
}
