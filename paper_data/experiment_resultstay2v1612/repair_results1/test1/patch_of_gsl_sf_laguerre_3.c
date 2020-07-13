#include "eft_patch.h"
static double array_idx_sf_laguerre_3_0[2] = {
9.108461422183616378e+04,
9.123013337411983230e+04,
};
static double array_cof_float_sf_laguerre_3_0[1][11] = {
{
-1.666666666666666574e-01,
-2.616436201210017316e+02,
-9.133458098023397906e+04,
-1.583946662989040992e-07,
5.000000000000000000e-01,
5.237872402420034632e+02,
9.159655793368832383e+04,
-5.000000000000000000e-01,
-2.621436201210017316e+02,
1.666666666666666574e-01,
3.000000000000000000e+00,
}
};
static double array_cof_err_sf_laguerre_3_0[1][11] = {
{
-9.251858538542970657e-18,
0.000000000000000000e+00,
-3.149538308378375256e-12,
8.370654867166265291e-24,
0.000000000000000000e+00,
0.000000000000000000e+00,
-6.551738510532859551e-12,
0.000000000000000000e+00,
0.000000000000000000e+00,
9.251858538542970657e-18,
}
};
static double array_pointx_sf_laguerre_3_0[1] = {
9.115737383962428430e+04,
};
static double array_pointy_sf_laguerre_3_0[1] = {
9.168366107986628776e+04,
};
static double array_cofidx_sf_laguerre_3_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_laguerre_3_0(double x,double y)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_laguerre_3_0[idx])&&(x<=array_idx_sf_laguerre_3_0[idx+1])){
         double pointx = array_pointx_sf_laguerre_3_0[idx];
         double pointy = array_pointy_sf_laguerre_3_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_laguerre_3_0[idx];
         eft_tay2v(array_cof_float_sf_laguerre_3_0[idx],array_cof_err_sf_laguerre_3_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(x<array_idx_sf_laguerre_3_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_laguerre_3_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_laguerre_3(double x,double y)
{
if((x<=91230.13337411983)&&(y<=91756.42061460292)){
 return accuracy_improve_patch_of_gsl_sf_laguerre_3_0(x,y);
}
}
