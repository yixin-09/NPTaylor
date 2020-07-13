#include "eft_patch.h"
static double array_idx_sf_zeta_0[2] = {
-4.000000000000011546e+00,
-3.999969661538037347e+00,
};
static double array_cof_float_sf_zeta_0[1][4] = {
{
-1.795996387069394122e-03,
2.868707923051893021e-03,
7.983811450268624849e-03,
0.000000000000000000e+00,
}
};
static double array_cof_err_sf_zeta_0[1][4] = {
{
-1.005277157106767137e-19,
1.350521691782867278e-19,
-5.678951988372493315e-19,
0.000000000000000000e+00,
}
};
static double array_point_sf_zeta_0[1] = {
-4.000000000000000000e+00,
};
static double array_cofidx_sf_zeta_0[1] = {
3.000000000000000000e+00,
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
if(x<=-3.9999696615380373){
 return accuracy_improve_patch_of_gsl_sf_zeta_0(x);
}
}
