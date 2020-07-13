#include "eft_patch.h"
static double array_idx_sf_laguerre_2_0[2] = {
2.788481117429416400e+01,
2.792033831108216901e+01,
};
static double array_cof_float_sf_laguerre_2_0[1][7] = {
{
5.000000000000000000e-01,
-5.468324674220447434e+00,
-1.971526623413934138e-15,
-1.000000000000000000e+00,
4.968324674220447434e+00,
5.000000000000000000e-01,
2.000000000000000000e+00,
}
};
static double array_cof_err_sf_laguerre_2_0[1][7] = {
{
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
}
};
static double array_pointx_sf_laguerre_2_0[1] = {
2.790257474268816651e+01,
};
static double array_pointy_sf_laguerre_2_0[1] = {
2.443425006846771907e+01,
};
static double array_cofidx_sf_laguerre_2_0[1] = {
2.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_laguerre_2_0(double x,double y)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_laguerre_2_0[idx])&&(x<=array_idx_sf_laguerre_2_0[idx+1])){
         double pointx = array_pointx_sf_laguerre_2_0[idx];
         double pointy = array_pointy_sf_laguerre_2_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_laguerre_2_0[idx];
         eft_tay2v(array_cof_float_sf_laguerre_2_0[idx],array_cof_err_sf_laguerre_2_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(x<array_idx_sf_laguerre_2_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_laguerre_2_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_laguerre_2(double x,double y)
{
if((x<=27.92033831108217)&&(y<=24.452013636861714)){
 return accuracy_improve_patch_of_gsl_sf_laguerre_2_0(x,y);
}
}
