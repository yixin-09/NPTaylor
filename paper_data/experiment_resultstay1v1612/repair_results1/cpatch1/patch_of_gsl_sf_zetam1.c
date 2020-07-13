#include "eft_patch.h"
static double array_idx_sf_zetam1_0[2] = {
-2.399999999999991829e+01,
-2.399990758855387085e+01,
};
static double array_cof_float_sf_zetam1_0[1][4] = {
{
1.158378576533671367e+04,
-2.945441179698775886e+04,
2.164234080545553661e+04,
9.756319734811011143e-12,
}
};
static double array_cof_err_sf_zetam1_0[1][4] = {
{
2.476874495627223287e-13,
9.509963515391706704e-13,
1.924121962196631466e-14,
-1.309485544647502082e-28,
}
};
static double array_point_sf_zetam1_0[1] = {
-2.399995379718249922e+01,
};
static double array_cofidx_sf_zetam1_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_zetam1_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_zetam1_0[idx])&&(x<=array_idx_sf_zetam1_0[idx+1])){
         double point = array_point_sf_zetam1_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_zetam1_0[idx];
         eft_tay1v(array_cof_float_sf_zetam1_0[idx],array_cof_err_sf_zetam1_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_zetam1_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_zetam1_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_zetam1(double x)
{
if(x<=-23.99990758855387){
 return accuracy_improve_patch_of_gsl_sf_zetam1_0(x);
}
}
