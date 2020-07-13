#include "eft_patch.h"
static double array_idx_sf_bessel_Y0_0[2] = {
3.957645988389550062e+00,
3.957716042609162432e+00,
};
static double array_cof_float_sf_bessel_Y0_0[1][4] = {
{
5.852382210517023675e-02,
5.085590959215825074e-02,
-4.025426717750242300e-01,
-4.333106464293519434e-17,
}
};
static double array_cof_err_sf_bessel_Y0_0[1][4] = {
{
3.412905233262102517e-18,
2.216441372273089305e-18,
5.256416660090713007e-18,
-2.051184634865321070e-33,
}
};
static double array_point_sf_bessel_Y0_0[1] = {
3.957678419314857976e+00,
};
static double array_cofidx_sf_bessel_Y0_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_Y0_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_Y0_0[idx])&&(x<=array_idx_sf_bessel_Y0_0[idx+1])){
         double point = array_point_sf_bessel_Y0_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_Y0_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_Y0_0[idx],array_cof_err_sf_bessel_Y0_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_Y0_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_Y0_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_Y0(double x)
{
if(x<=3.9577160426091624){
 return accuracy_improve_patch_of_gsl_sf_bessel_Y0_0(x);
}
}
