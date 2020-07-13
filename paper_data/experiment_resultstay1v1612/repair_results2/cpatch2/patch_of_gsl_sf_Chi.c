#include "eft_patch.h"
static double array_idx_sf_Chi_0[2] = {
5.237665008765427510e-01,
5.239340072935383397e-01,
};
static double array_cof_float_sf_Chi_0[1][5] = {
{
-3.309111772607951529e+00,
2.341633731889031900e+00,
-1.554810664803668585e+00,
2.176998509070992238e+00,
5.806356010173173728e-17,
}
};
static double array_cof_err_sf_Chi_0[1][5] = {
{
9.963242640032543102e-17,
5.669913555646764444e-17,
-1.304463200357040495e-17,
4.500079196424844728e-17,
2.896881910010415761e-33,
}
};
static double array_point_sf_Chi_0[1] = {
5.238225713898644331e-01,
};
static double array_cofidx_sf_Chi_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_Chi_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_Chi_0[idx])&&(x<=array_idx_sf_Chi_0[idx+1])){
         double point = array_point_sf_Chi_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_Chi_0[idx];
         eft_tay1v(array_cof_float_sf_Chi_0[idx],array_cof_err_sf_Chi_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_Chi_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_Chi_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_Chi(double x)
{
if(x<=0.5239340072935383){
 return accuracy_improve_patch_of_gsl_sf_Chi_0(x);
}
}
