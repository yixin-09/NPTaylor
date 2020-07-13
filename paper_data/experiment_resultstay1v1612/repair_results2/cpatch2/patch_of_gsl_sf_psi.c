#include "eft_patch.h"
static double array_idx_sf_psi_0[2] = {
-1.573450976327126050e+01,
-1.572772274369095413e+01,
};
static double array_cof_float_sf_psi_0[1][8] = {
{
3.647426071208139911e+04,
-9.799840358205741722e+03,
2.645474571637989357e+03,
-7.052813983902887003e+02,
1.950289737140369652e+02,
-4.915880694488895131e+01,
1.757583029204344172e+01,
-1.854593944673241691e-15,
}
};
static double array_cof_err_sf_psi_0[1][8] = {
{
-2.943108644113077553e-12,
-2.781839938215224618e-13,
-5.415623152634271596e-14,
2.683399173630814211e-14,
2.647970061363738855e-15,
3.143697899299794210e-15,
-1.083505272452589822e-16,
-4.016137443983636400e-32,
}
};
static double array_point_sf_psi_0[1] = {
-1.573098890633288249e+01,
};
static double array_cofidx_sf_psi_0[1] = {
7.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_psi_0(double x)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_psi_0[idx])&&(x<=array_idx_sf_psi_0[idx+1])){
         double point = array_point_sf_psi_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_psi_0[idx];
         eft_tay1v(array_cof_float_sf_psi_0[idx],array_cof_err_sf_psi_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_psi_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_psi_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_psi(double x)
{
if(x<=-15.727722743690954){
 return accuracy_improve_patch_of_gsl_sf_psi_0(x);
}
}
