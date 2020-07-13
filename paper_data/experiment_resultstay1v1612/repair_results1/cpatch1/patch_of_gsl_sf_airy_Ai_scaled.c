#include "eft_patch.h"
static double array_idx_sf_airy_Ai_scaled_0[2] = {
-2.375169042693386814e+02,
-2.375027670539626570e+02,
};
static double array_cof_float_sf_airy_Ai_scaled_0[1][8] = {
{
-5.887886137943299218e+03,
-4.383750405488668456e+00,
1.041183838455605155e+03,
1.845713207082292362e-01,
-8.767500811543338557e+01,
1.806036499934248477e-12,
2.214855848927702375e+00,
-1.520809860277674365e-14,
}
};
static double array_cof_err_sf_airy_Ai_scaled_0[1][8] = {
{
2.396840301763672744e-13,
-1.179172594223168154e-16,
2.232736525727273256e-15,
1.929118320523228590e-19,
3.019403312379172849e-15,
6.222606860354313609e-29,
-1.063410138164868323e-16,
-3.664196770027701718e-32,
}
};
static double array_point_sf_airy_Ai_scaled_0[1] = {
-2.375098356614411443e+02,
};
static double array_cofidx_sf_airy_Ai_scaled_0[1] = {
7.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_airy_Ai_scaled_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_airy_Ai_scaled_0[idx])&&(x<=array_idx_sf_airy_Ai_scaled_0[idx+1])){
         double point = array_point_sf_airy_Ai_scaled_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_airy_Ai_scaled_0[idx];
         eft_tay1v(array_cof_float_sf_airy_Ai_scaled_0[idx],array_cof_err_sf_airy_Ai_scaled_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_airy_Ai_scaled_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_airy_Ai_scaled_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_airy_Ai_scaled(double x)
{
if(x<=-237.50276705396266){
 return accuracy_improve_patch_of_gsl_sf_airy_Ai_scaled_0(x);
}
}
