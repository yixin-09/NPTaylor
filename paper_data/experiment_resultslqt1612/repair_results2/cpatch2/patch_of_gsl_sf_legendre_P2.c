static double array_x_sf_legendre_P2_0[2] = {
5.773502691896257311e-01,
5.773502691896257311e-01,
};
static double array_y_sf_legendre_P2_0[2] = {
-5.793758576800781262e-17,
-5.793758576800781262e-17,
};
static double array_e_y_sf_legendre_P2_0[2] = {
-1.220702785793555627e-04,
1.220703124970892560e-04,
};
static double array_detla_sf_legendre_P2_0[2] = {
1.922845310923686683e-16,
1.923080047548867442e-16,
};
static double array_idx_sf_legendre_P2_0[3] = {
0.000000000000000000e+00,
6.348419079050000000e+11,
1.269606501572000000e+12,
};
static double array_maxE_sf_legendre_P2_0[2] = {
1.500000000007333245e+00,
1.499999999998473887e+00,
};
double accuracy_improve_patch_of_gsl_sf_legendre_P2_0(double x)
{
 long int n = 1269606501573;
 int len_glob = 2;
 double ulp_x = 1.1102230246251565e-16;
 double x_0 = 0.5772797875793104;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_legendre_P2_0[idx])&&(n_x<array_idx_sf_legendre_P2_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_legendre_P2_0[idx+1])*(n_x-array_idx_sf_legendre_P2_0[idx])*array_maxE_sf_legendre_P2_0[idx];
         return (x-array_x_sf_legendre_P2_0[idx])/ulp_x*array_detla_sf_legendre_P2_0[idx]+array_y_sf_legendre_P2_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_legendre_P2_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_legendre_P2_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_legendre_P2_0[idx]){
         return array_y_sf_legendre_P2_0[idx];
     }
     else{
         return array_e_y_sf_legendre_P2_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_legendre_P2(double x)
{
if(x<=0.5774207422163363){
 return accuracy_improve_patch_of_gsl_sf_legendre_P2_0(x);
}
}
