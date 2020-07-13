#include "eft_patch.h"
static double array_idx_sf_lnsinh_0[2] = {
8.812009640651985443e-01,
8.815462310473199947e-01,
};
static double array_cof_float_sf_lnsinh_0[1][5] = {
{
-4.166666666666666297e-01,
4.714045207910316226e-01,
-4.999999999999999445e-01,
1.414213562373094923e+00,
3.182752524377405545e-17,
}
};
static double array_cof_err_sf_lnsinh_0[1][5] = {
{
5.429266170860189760e-18,
2.278517612682223769e-17,
-2.368362598748377158e-17,
1.028662128622435222e-16,
2.023594140346954188e-33,
}
};
static double array_point_sf_lnsinh_0[1] = {
8.813735870195430477e-01,
};
static double array_cofidx_sf_lnsinh_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_lnsinh_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_lnsinh_0[idx])&&(x<=array_idx_sf_lnsinh_0[idx+1])){
         double point = array_point_sf_lnsinh_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_lnsinh_0[idx];
         eft_tay1v(array_cof_float_sf_lnsinh_0[idx],array_cof_err_sf_lnsinh_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_lnsinh_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_lnsinh_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_lnsinh(double x)
{
if(x<=0.88154623104732){
 return accuracy_improve_patch_of_gsl_sf_lnsinh_0(x);
}
}
