#include "eft_patch.h"
static double array_idx_sf_Ci_0[2] = {
6.164719240816939339e-01,
6.165491270786274125e-01,
};
static double array_cof_float_sf_Ci_0[1][4] = {
{
1.447169470772171840e+00,
-1.542257855747528827e+00,
1.323433347953753447e+00,
5.571548945612863202e-17,
}
};
static double array_cof_err_sf_Ci_0[1][4] = {
{
-6.526862797260315321e-17,
-4.338695691778796379e-17,
-5.758471531524557359e-17,
-7.367374693083367525e-34,
}
};
static double array_point_sf_Ci_0[1] = {
6.165054856207162759e-01,
};
static double array_cofidx_sf_Ci_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_Ci_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_Ci_0[idx])&&(x<=array_idx_sf_Ci_0[idx+1])){
         double point = array_point_sf_Ci_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_Ci_0[idx];
         eft_tay1v(array_cof_float_sf_Ci_0[idx],array_cof_err_sf_Ci_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_Ci_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_Ci_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_Ci(double x)
{
if(x<=0.6165491270786274){
 return accuracy_improve_patch_of_gsl_sf_Ci_0(x);
}
}
