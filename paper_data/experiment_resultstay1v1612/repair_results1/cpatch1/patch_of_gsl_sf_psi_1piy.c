#include "eft_patch.h"
static double array_idx_sf_psi_1piy_0[2] = {
8.837543958804204847e-01,
8.838760659680020559e-01,
};
static double array_cof_float_sf_psi_1piy_0[1][4] = {
{
-1.345041647166839438e-01,
-1.527243575768862105e-01,
8.339357939112115314e-01,
7.765374422987675065e-19,
}
};
static double array_cof_err_sf_psi_1piy_0[1][4] = {
{
1.940315028789972146e-18,
-1.194582520854643374e-17,
-4.283548366751246638e-17,
4.473195090382002107e-35,
}
};
static double array_point_sf_psi_1piy_0[1] = {
8.838206378941847463e-01,
};
static double array_cofidx_sf_psi_1piy_0[1] = {
3.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_psi_1piy_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_psi_1piy_0[idx])&&(x<=array_idx_sf_psi_1piy_0[idx+1])){
         double point = array_point_sf_psi_1piy_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_psi_1piy_0[idx];
         eft_tay1v(array_cof_float_sf_psi_1piy_0[idx],array_cof_err_sf_psi_1piy_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_psi_1piy_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_psi_1piy_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_psi_1piy(double x)
{
if(x<=0.8838760659680021){
 return accuracy_improve_patch_of_gsl_sf_psi_1piy_0(x);
}
}
