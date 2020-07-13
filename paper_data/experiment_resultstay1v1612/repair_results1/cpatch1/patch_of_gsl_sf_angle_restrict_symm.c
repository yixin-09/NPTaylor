#include "eft_patch.h"
static double array_idx_sf_angle_restrict_symm_0[2] = {
-2.261946710584650759e+02,
-2.261946405408870930e+02,
};
static double array_cof_float_sf_angle_restrict_symm_0[1][2] = {
{
1.000000000000000000e+00,
1.277016499120526047e-11,
}
};
static double array_cof_err_sf_angle_restrict_symm_0[1][2] = {
{
0.000000000000000000e+00,
-1.897394559056775720e-28,
}
};
static double array_point_sf_angle_restrict_symm_0[1] = {
-2.261946710584523430e+02,
};
static double array_cofidx_sf_angle_restrict_symm_0[1] = {
1.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_angle_restrict_symm_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_angle_restrict_symm_0[idx])&&(x<=array_idx_sf_angle_restrict_symm_0[idx+1])){
         double point = array_point_sf_angle_restrict_symm_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_angle_restrict_symm_0[idx];
         eft_tay1v(array_cof_float_sf_angle_restrict_symm_0[idx],array_cof_err_sf_angle_restrict_symm_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_angle_restrict_symm_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_angle_restrict_symm_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_angle_restrict_symm(double x)
{
if(x<=-226.1946405408871){
 return accuracy_improve_patch_of_gsl_sf_angle_restrict_symm_0(x);
}
}
