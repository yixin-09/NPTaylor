#include "eft_patch.h"
static double array_idx_sf_expint_Ei_0[2] = {
3.724760811272705974e-01,
3.725387420889791912e-01,
};
static double array_cof_float_sf_expint_Ei_0[1][4] = {
{
6.522376145438926187e+00,
-3.281607866398561946e+00,
3.896215733907167245e+00,
-5.119698936555684652e-17,
}
};
static double array_cof_err_sf_expint_Ei_0[1][4] = {
{
1.922995458560042558e-16,
1.769195098074498263e-17,
1.517887794630455990e-16,
-5.046173363100548144e-34,
}
};
static double array_point_sf_expint_Ei_0[1] = {
3.725074107813666213e-01,
};
static double array_cofidx_sf_expint_Ei_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_expint_Ei_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_expint_Ei_0[idx])&&(x<=array_idx_sf_expint_Ei_0[idx+1])){
         double point = array_point_sf_expint_Ei_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_expint_Ei_0[idx];
         eft_tay1v(array_cof_float_sf_expint_Ei_0[idx],array_cof_err_sf_expint_Ei_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_expint_Ei_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_expint_Ei_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_expint_Ei(double x)
{
if(x<=0.3725387420889792){
 return accuracy_improve_patch_of_gsl_sf_expint_Ei_0(x);
}
}
