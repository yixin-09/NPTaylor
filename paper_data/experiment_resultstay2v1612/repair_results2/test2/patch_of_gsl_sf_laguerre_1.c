#include "eft_patch.h"
static double array_idx_sf_laguerre_1_0[2] = {
1.548829415615331406e+01,
1.550605772454731657e+01,
};
static double array_cof_float_sf_laguerre_1_0[1][4] = {
{
-1.000000000000000000e+00,
-1.776356839400250465e-15,
1.000000000000000000e+00,
1.000000000000000000e+00,
}
};
static double array_cof_err_sf_laguerre_1_0[1][4] = {
{
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
}
};
static double array_pointx_sf_laguerre_1_0[1] = {
1.549717594035031532e+01,
};
static double array_pointy_sf_laguerre_1_0[1] = {
1.649717594035031709e+01,
};
static double array_cofidx_sf_laguerre_1_0[1] = {
1.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_laguerre_1_0(double x,double y)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_laguerre_1_0[idx])&&(x<=array_idx_sf_laguerre_1_0[idx+1])){
         double pointx = array_pointx_sf_laguerre_1_0[idx];
         double pointy = array_pointy_sf_laguerre_1_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_laguerre_1_0[idx];
         eft_tay2v(array_cof_float_sf_laguerre_1_0[idx],array_cof_err_sf_laguerre_1_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(x<array_idx_sf_laguerre_1_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_laguerre_1_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_laguerre_1(double x,double y)
{
if((x<=15.506057724547317)&&(y<=16.51493950874432)){
 return accuracy_improve_patch_of_gsl_sf_laguerre_1_0(x,y);
}
}
