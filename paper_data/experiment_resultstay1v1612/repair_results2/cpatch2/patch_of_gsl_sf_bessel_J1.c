#include "eft_patch.h"
static double array_idx_sf_bessel_J1_0[2] = {
3.831088815794024161e+00,
3.832312188098114802e+00,
};
static double array_cof_float_sf_bessel_J1_0[1][5] = {
{
-5.179719245638572146e-03,
5.341044413272480473e-02,
5.255614585697725855e-02,
-4.027593957025529803e-01,
-6.149807356994905840e-17,
}
};
static double array_cof_err_sf_bessel_J1_0[1][5] = {
{
2.358083234611443885e-19,
1.231442727077160195e-18,
1.185807573245637287e-18,
2.423224183401403964e-17,
-2.513304530441144964e-33,
}
};
static double array_point_sf_bessel_J1_0[1] = {
3.831705970207512468e+00,
};
static double array_cofidx_sf_bessel_J1_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_J1_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_J1_0[idx])&&(x<=array_idx_sf_bessel_J1_0[idx+1])){
         double point = array_point_sf_bessel_J1_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_J1_0[idx];
         eft_tay1v(array_cof_float_sf_bessel_J1_0[idx],array_cof_err_sf_bessel_J1_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_J1_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_J1_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_J1(double x)
{
if(x<=3.832312188098115){
 return accuracy_improve_patch_of_gsl_sf_bessel_J1_0(x);
}
}
