#include "eft_patch.h"
static double array_idx_sf_bessel_J0_0[2] = {
2.404810861735132921e+00,
2.404840253746268974e+00,
};
static double array_cof_float_sf_bessel_J0_0[1][4] = {
{
5.660177443794622842e-02,
1.079387017549201105e-01,
-5.191474972894667417e-01,
-6.108765259736730323e-17,
}
};
static double array_cof_err_sf_bessel_J0_0[1][4] = {
{
-1.062259036469890243e-18,
1.800225289209694109e-19,
-2.106135543036934373e-17,
-7.416300334440819944e-34,
}
};
static double array_point_sf_bessel_J0_0[1] = {
2.404825557695772886e+00,
};
static double array_cofidx_sf_bessel_J0_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_J0_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_J0_0[idx])&&(x<=array_idx_sf_bessel_J0_0[idx+1])){
         double point = array_point_sf_bessel_J0_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_J0_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_J0_0[idx],array_cof_err_sf_bessel_J0_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_J0_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_J0_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_J0(double x)
{
if(x<=2.404840253746269){
 return accuracy_improve_patch_of_gsl_sf_bessel_J0_0(x);
}
}
