static double array_x_sf_bessel_Y1_0[2] = {
3.061828649164111482e+01,
3.061828649164111482e+01,
};
static double array_y_sf_bessel_Y1_0[2] = {
-1.524456280251315065e-17,
-1.524456280251315065e-17,
};
static double array_e_y_sf_bessel_Y1_0[2] = {
9.536742838099055489e-07,
-9.536743041818835284e-07,
};
static double array_detla_sf_bessel_Y1_0[2] = {
-5.121805926603487624e-16,
-5.121804820032187735e-16,
};
static double array_idx_sf_bessel_Y1_0[3] = {
0.000000000000000000e+00,
1.861988325000000000e+09,
3.723977092000000000e+09,
};
static double array_maxE_sf_bessel_Y1_0[2] = {
2.354009123708587867e-03,
2.354484426490519120e-03,
};
double accuracy_improve_patch_of_gsl_sf_bessel_Y1_0(double x)
{
 long int n = 3723977093;
 int len_glob = 2;
 double ulp_x = 3.552713678800501e-15;
 double x_0 = 30.618279876529723;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_Y1_0[idx])&&(n_x<array_idx_sf_bessel_Y1_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_Y1_0[idx+1])*(n_x-array_idx_sf_bessel_Y1_0[idx])*array_maxE_sf_bessel_Y1_0[idx];
         return (x-array_x_sf_bessel_Y1_0[idx])/ulp_x*array_detla_sf_bessel_Y1_0[idx]+array_y_sf_bessel_Y1_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_Y1_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_Y1_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_Y1_0[idx]){
         return array_y_sf_bessel_Y1_0[idx];
     }
     else{
         return array_e_y_sf_bessel_Y1_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_Y1(double x)
{
if(x<=30.618293106754077){
 return accuracy_improve_patch_of_gsl_sf_bessel_Y1_0(x);
}
}
