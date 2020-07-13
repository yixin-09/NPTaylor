#include "eft_patch.h"
static double array_idx_sf_eta_0[2] = {
-2.000286386489099844e+00,
-1.999713660703564821e+00,
};
static double array_cof_float_sf_eta_0[1][4] = {
{
-2.673779125529428952e-02,
6.133020935658128892e-02,
2.131391994087528940e-01,
0.000000000000000000e+00,
}
};
static double array_cof_err_sf_eta_0[1][4] = {
{
7.458490485538093674e-19,
2.200320196705660308e-18,
1.412403717078070706e-18,
0.000000000000000000e+00,
}
};
static double array_point_sf_eta_0[1] = {
-2.000000000000000000e+00,
};
static double array_cofidx_sf_eta_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_eta_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_eta_0[idx])&&(x<=array_idx_sf_eta_0[idx+1])){
         double point = array_point_sf_eta_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_eta_0[idx];
         eft_tay1v(array_cof_float_sf_eta_0[idx],array_cof_err_sf_eta_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_eta_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_eta_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_eta(double x)
{
if(x<=-1.9997136607035648){
 return accuracy_improve_patch_of_gsl_sf_eta_0(x);
}
}
