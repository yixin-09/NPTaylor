static double array_x_sf_bessel_j2_0[2] = {
9.095011330476355127e+00,
9.095011330476355127e+00,
};
static double array_y_sf_bessel_j2_0[2] = {
-3.170877217559426233e-18,
-3.170877217559426233e-18,
};
static double array_e_y_sf_bessel_j2_0[2] = {
-9.536639599366175795e-07,
9.536743087435071504e-07,
};
static double array_detla_sf_bessel_j2_0[2] = {
1.917414710038817745e-16,
1.917410984795103291e-16,
};
static double array_idx_sf_bessel_j2_0[3] = {
0.000000000000000000e+00,
4.973696900000000000e+09,
9.947457436000000000e+09,
};
static double array_maxE_sf_bessel_j2_0[2] = {
-1.186792292854709462e-02,
-1.186833058655357986e-02,
};
double accuracy_improve_patch_of_gsl_sf_bessel_j2_0(double x)
{
 long int n = 9947457437;
 int len_glob = 2;
 double ulp_x = 1.7763568394002505e-15;
 double x_0 = 9.09500249541585;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_j2_0[idx])&&(n_x<array_idx_sf_bessel_j2_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_j2_0[idx+1])*(n_x-array_idx_sf_bessel_j2_0[idx])*array_maxE_sf_bessel_j2_0[idx];
         return (x-array_x_sf_bessel_j2_0[idx])/ulp_x*array_detla_sf_bessel_j2_0[idx]+array_y_sf_bessel_j2_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_j2_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_j2_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_j2_0[idx]){
         return array_y_sf_bessel_j2_0[idx];
     }
     else{
         return array_e_y_sf_bessel_j2_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_j2(double x)
{
if(x<=9.0950201656499){
 return accuracy_improve_patch_of_gsl_sf_bessel_j2_0(x);
}
}
