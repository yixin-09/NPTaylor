#include "eft_patch.h"
static double array_idx_sf_clausen_0[2] = {
-3.143001535633751509e+00,
-3.140183771545858260e+00,
};
static double array_cof_float_sf_clausen_0[1][4] = {
{
4.166666666666666435e-02,
1.530808498934191509e-17,
-6.931471805599452862e-01,
-8.488604760107495465e-17,
}
};
static double array_cof_err_sf_clausen_0[1][4] = {
{
2.312964634635743049e-18,
-3.743462262147924155e-34,
-2.319046813846299558e-17,
5.639425589331043367e-33,
}
};
static double array_point_sf_clausen_0[1] = {
-3.141592653589793116e+00,
};
static double array_cofidx_sf_clausen_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_clausen_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_clausen_0[idx])&&(x<=array_idx_sf_clausen_0[idx+1])){
         double point = array_point_sf_clausen_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_clausen_0[idx];
         eft_tay1v(array_cof_float_sf_clausen_0[idx],array_cof_err_sf_clausen_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_clausen_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_clausen_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_clausen(double x)
{
if(x<=-3.1401837715458583){
 return accuracy_improve_patch_of_gsl_sf_clausen_0(x);
}
}
