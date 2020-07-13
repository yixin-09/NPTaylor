static double array_x_sf_bessel_y1_0[2] = {
2.798386045783887166e+00,
2.798386045783887166e+00,
};
static double array_y_sf_bessel_y1_0[2] = {
9.865851509229451309e-18,
9.865851509229451309e-18,
};
static double array_e_y_sf_bessel_y1_0[2] = {
-3.542199166366137919e-06,
2.468628639408662252e-06,
};
static double array_detla_sf_bessel_y1_0[2] = {
1.494403191035873128e-16,
1.494393652177510847e-16,
};
static double array_idx_sf_bessel_y1_0[3] = {
0.000000000000000000e+00,
2.370310226600000000e+10,
4.022236836300000000e+10,
};
static double array_maxE_sf_bessel_y1_0[2] = {
-1.202509106671825895e-01,
-1.202508782376265628e-01,
};
double accuracy_improve_patch_of_gsl_sf_bessel_y1_0(double x)
{
 long int n = 40222368364;
 int len_glob = 2;
 double ulp_x = 4.440892098500626e-16;
 double x_0 = 2.798375519491931;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_y1_0[idx])&&(n_x<array_idx_sf_bessel_y1_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_y1_0[idx+1])*(n_x-array_idx_sf_bessel_y1_0[idx])*array_maxE_sf_bessel_y1_0[idx];
         return (x-array_x_sf_bessel_y1_0[idx])/ulp_x*array_detla_sf_bessel_y1_0[idx]+array_y_sf_bessel_y1_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_y1_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_y1_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_y1_0[idx]){
         return array_y_sf_bessel_y1_0[idx];
     }
     else{
         return array_e_y_sf_bessel_y1_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_y1(double x)
{
if(x<=2.7983933818117155){
 return accuracy_improve_patch_of_gsl_sf_bessel_y1_0(x);
}
}
