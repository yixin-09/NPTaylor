static double array_x_sf_bessel_J0_0[4] = {
2.404818209715452682e+00,
2.404825557695772886e+00,
2.404825557695772886e+00,
2.404832905721020708e+00,
};
static double array_y_sf_bessel_J0_0[4] = {
3.814691421197356335e-06,
-6.108765259736730323e-17,
-6.108765259736730323e-17,
-3.814703089479618604e-06,
};
static double array_e_y_sf_bessel_J0_0[4] = {
7.629394497919178140e-06,
3.814691421197356335e-06,
-3.814703089479618604e-06,
-7.629394523022759666e-06,
};
static double array_detla_sf_bessel_J0_0[4] = {
-2.305488585207817000e-16,
-2.305481540866813539e-16,
-2.305474496422842529e-16,
-2.305467451875905942e-16,
};
static double array_idx_sf_bessel_J0_0[5] = {
0.000000000000000000e+00,
1.654618071500000000e+10,
3.309236143100000000e+10,
4.963864331500000000e+10,
6.618492520000000000e+10,
};
static double array_maxE_sf_bessel_J0_0[4] = {
1.079368301604958641e-01,
1.079380778832029764e-01,
1.079393256506182980e-01,
1.079405733423794700e-01,
};
double accuracy_improve_patch_of_gsl_sf_bessel_J0_0(double x)
{
 long int n = 66184925201;
 int len_glob = 4;
 double ulp_x = 4.440892098500626e-16;
 double x_0 = 2.404810861735133;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_J0_0[idx])&&(n_x<array_idx_sf_bessel_J0_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_J0_0[idx+1])*(n_x-array_idx_sf_bessel_J0_0[idx])*array_maxE_sf_bessel_J0_0[idx];
         return (x-array_x_sf_bessel_J0_0[idx])/ulp_x*array_detla_sf_bessel_J0_0[idx]+array_y_sf_bessel_J0_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_J0_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_J0_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_J0_0[idx]){
         return array_y_sf_bessel_J0_0[idx];
     }
     else{
         return array_e_y_sf_bessel_J0_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_J0(double x)
{
if(x<=2.404840253746269){
 return accuracy_improve_patch_of_gsl_sf_bessel_J0_0(x);
}
}
