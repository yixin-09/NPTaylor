#include "eft_patch.h"
static double array_idx_sf_zeta_0[2] = {
-3.999999999999999556e+00,
-3.999514879261854894e+00,
};
static double array_cof_float_sf_zeta_0[1][5] = {
{
-6.781921885345418071e-04,
-1.795996387069395423e-03,
2.868707923051890853e-03,
7.983811450268626583e-03,
3.545524518541676095e-18,
}
};
static double array_cof_err_sf_zeta_0[1][5] = {
{
-8.500450823179858789e-21,
-4.196441259228804947e-21,
-8.929133513234870873e-20,
2.453057948634014588e-19,
3.135567521190728698e-34,
}
};
static double array_point_sf_zeta_0[1] = {
-3.999999999999999556e+00,
};
static double array_cofidx_sf_zeta_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_zeta_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_zeta_0[idx])&&(x<=array_idx_sf_zeta_0[idx+1])){
         double point = array_point_sf_zeta_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_zeta_0[idx];
         eft_tay1v(array_cof_float_sf_zeta_0[idx],array_cof_err_sf_zeta_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_zeta_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_zeta_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_zeta(double x)
{
if(x<=-3.999514879261855){
 return accuracy_improve_patch_of_gsl_sf_zeta_0(x);
}
}
