#include "eft_patch.h"
static double array_idx_sf_airy_Bi_deriv_scaled_0[2] = {
-3.163079945414391148e+01,
-3.162978715727367529e+01,
};
static double array_cof_float_sf_airy_Bi_deriv_scaled_0[1][5] = {
{
-1.254151076466979431e+00,
3.966918205241272943e+01,
1.189508964812723002e-01,
-7.524906458799131670e+00,
-1.097364444966437889e-14,
}
};
static double array_cof_err_sf_airy_Bi_deriv_scaled_0[1][5] = {
{
4.168980871333520247e-17,
-1.932813678458447139e-16,
-4.878042683664449295e-18,
5.237885267044645513e-17,
-2.960392066708032970e-31,
}
};
static double array_point_sf_airy_Bi_deriv_scaled_0[1] = {
-3.163030578754333533e+01,
};
static double array_cofidx_sf_airy_Bi_deriv_scaled_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_airy_Bi_deriv_scaled_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_airy_Bi_deriv_scaled_0[idx])&&(x<=array_idx_sf_airy_Bi_deriv_scaled_0[idx+1])){
         double point = array_point_sf_airy_Bi_deriv_scaled_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_airy_Bi_deriv_scaled_0[idx];
         eft_tay1v(array_cof_float_sf_airy_Bi_deriv_scaled_0[idx],array_cof_err_sf_airy_Bi_deriv_scaled_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_airy_Bi_deriv_scaled_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_airy_Bi_deriv_scaled_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_airy_Bi_deriv_scaled(double x)
{
if(x<=-31.629787157273675){
 return accuracy_improve_patch_of_gsl_sf_airy_Bi_deriv_scaled_0(x);
}
}
