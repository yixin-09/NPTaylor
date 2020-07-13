static double array_x_sf_bessel_y2_0[4] = {
7.451602758275821969e+00,
7.451610064214503559e+00,
7.451610064214503559e+00,
7.451617370181577549e+00,
};
static double array_y_sf_bessel_y2_0[4] = {
9.536733742574341173e-07,
-2.058941623624766705e-17,
-2.058941623624766705e-17,
-9.536752104028642642e-07,
};
static double array_e_y_sf_bessel_y2_0[4] = {
1.907348618669722382e-06,
9.536733742574341173e-07,
-9.536752104028642642e-07,
-1.907348550672470757e-06,
};
static double array_detla_sf_bessel_y2_0[4] = {
-1.159377060637926476e-16,
-1.159374787268281742e-16,
-1.159372513845709618e-16,
-1.159370240370210105e-16,
};
static double array_idx_sf_bessel_y2_0[5] = {
0.000000000000000000e+00,
8.225755682000000000e+09,
1.645151136300000000e+10,
2.467729901100000000e+10,
3.290308665900000000e+10,
};
static double array_maxE_sf_bessel_y2_0[4] = {
1.751697813814706572e-02,
1.751735191971122776e-02,
1.751772570523587452e-02,
1.751809948757026072e-02,
};
double accuracy_improve_patch_of_gsl_sf_bessel_y2_0(double x)
{
 long int n = 32903086660;
 int len_glob = 4;
 double ulp_x = 8.881784197001252e-16;
 double x_0 = 7.4515954523371395;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_y2_0[idx])&&(n_x<array_idx_sf_bessel_y2_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_y2_0[idx+1])*(n_x-array_idx_sf_bessel_y2_0[idx])*array_maxE_sf_bessel_y2_0[idx];
         return (x-array_x_sf_bessel_y2_0[idx])/ulp_x*array_detla_sf_bessel_y2_0[idx]+array_y_sf_bessel_y2_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_y2_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_y2_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_y2_0[idx]){
         return array_y_sf_bessel_y2_0[idx];
     }
     else{
         return array_e_y_sf_bessel_y2_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_y2(double x)
{
if(x<=7.4516246761486515){
 return accuracy_improve_patch_of_gsl_sf_bessel_y2_0(x);
}
}
