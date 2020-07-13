#include "eft_patch.h"
static double array_idx_sf_airy_Ai_deriv_scaled_0[2] = {
-4.648315320408228501e+01,
-4.648159833389217255e+01,
};
static double array_cof_float_sf_airy_Ai_deriv_scaled_0[1][5] = {
{
1.673938806241844768e+00,
-7.780865243984204938e+01,
-1.080369999677472520e-01,
1.004363283744715218e+01,
7.250766601159764274e-15,
}
};
static double array_cof_err_sf_airy_Ai_deriv_scaled_0[1][5] = {
{
-1.754455787808109136e-17,
-2.348441212118516859e-15,
4.409410858789329500e-18,
-2.044942738425776249e-16,
-1.843350926575048947e-31,
}
};
static double array_point_sf_airy_Ai_deriv_scaled_0[1] = {
-4.648237566972975543e+01,
};
static double array_cofidx_sf_airy_Ai_deriv_scaled_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_airy_Ai_deriv_scaled_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_airy_Ai_deriv_scaled_0[idx])&&(x<=array_idx_sf_airy_Ai_deriv_scaled_0[idx+1])){
         double point = array_point_sf_airy_Ai_deriv_scaled_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_airy_Ai_deriv_scaled_0[idx];
         eft_tay1v(array_cof_float_sf_airy_Ai_deriv_scaled_0[idx],array_cof_err_sf_airy_Ai_deriv_scaled_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_airy_Ai_deriv_scaled_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_airy_Ai_deriv_scaled_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_airy_Ai_deriv_scaled(double x)
{
if(x<=-46.48159833389217){
 return accuracy_improve_patch_of_gsl_sf_airy_Ai_deriv_scaled_0(x);
}
}
