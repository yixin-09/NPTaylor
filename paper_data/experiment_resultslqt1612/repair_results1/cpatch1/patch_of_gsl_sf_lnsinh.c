static double array_x_sf_lnsinh_0[5] = {
8.813681922435200322e-01,
8.813735870195430477e-01,
8.813735870195430477e-01,
8.813776911425579286e-01,
8.813817952655726984e-01,
};
static double array_y_sf_lnsinh_0[5] = {
-7.629379969560098666e-06,
3.182752524377405545e-17,
3.182752524377405545e-17,
5.804098007443585347e-06,
1.160817917106813874e-05,
};
static double array_e_y_sf_lnsinh_0[5] = {
-1.525878904320445224e-05,
-7.629379969560098666e-06,
5.804098007443585347e-06,
1.160817917106813874e-05,
2.321629096836575414e-05,
};
static double array_detla_sf_lnsinh_0[5] = {
1.570101442897228887e-16,
1.570095453401283685e-16,
1.570090180446656995e-16,
1.570085624007682241e-16,
1.570078789455003153e-16,
};
static double array_idx_sf_lnsinh_0[6] = {
0.000000000000000000e+00,
4.859182257400000000e+10,
9.718364514800000000e+10,
1.341502989090000000e+11,
1.711169526690000000e+11,
2.450502601900000000e+11,
};
static double array_maxE_sf_lnsinh_0[5] = {
-5.000114445126958573e-01,
-5.000038147106885544e-01,
-4.999970978270797395e-01,
-4.999912937778879241e-01,
-4.999825880020101199e-01,
};
double accuracy_improve_patch_of_gsl_sf_lnsinh_0(double x)
{
 long int n = 245050260191;
 int len_glob = 5;
 double ulp_x = 1.1102230246251565e-16;
 double x_0 = 0.881362797467497;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_lnsinh_0[idx])&&(n_x<array_idx_sf_lnsinh_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_lnsinh_0[idx+1])*(n_x-array_idx_sf_lnsinh_0[idx])*array_maxE_sf_lnsinh_0[idx];
         return (x-array_x_sf_lnsinh_0[idx])/ulp_x*array_detla_sf_lnsinh_0[idx]+array_y_sf_lnsinh_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_lnsinh_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_lnsinh_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_lnsinh_0[idx]){
         return array_y_sf_lnsinh_0[idx];
     }
     else{
         return array_e_y_sf_lnsinh_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_lnsinh(double x)
{
if(x<=0.8813900035116023){
 return accuracy_improve_patch_of_gsl_sf_lnsinh_0(x);
}
}
