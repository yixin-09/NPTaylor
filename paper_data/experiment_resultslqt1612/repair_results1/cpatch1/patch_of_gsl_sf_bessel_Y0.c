static double array_x_sf_bessel_Y0_0[6] = {
3.957662203852204019e+00,
3.957670311583530776e+00,
3.957678419314857976e+00,
3.957678419314857976e+00,
3.957687825138433979e+00,
3.957697230962009982e+00,
};
static double array_y_sf_bessel_Y0_0[6] = {
6.527429032614406182e-06,
3.263711173439649871e-06,
-4.333106464293519434e-17,
-4.333106464293519434e-17,
-3.786240853325649994e-06,
-7.572472707920242975e-06,
};
static double array_e_y_sf_bessel_Y0_0[6] = {
1.305488480800571164e-05,
6.527429032614406182e-06,
3.263711173439649871e-06,
-3.786240853325649994e-06,
-7.572472707920242975e-06,
-1.514490942005623654e-05,
};
static double array_detla_sf_bessel_Y0_0[6] = {
-1.787659556489689789e-16,
-1.787654063561955258e-16,
-1.787650401473455348e-16,
-1.787646446108107359e-16,
-1.787642197442278745e-16,
-1.787635824167620505e-16,
};
static double array_idx_sf_bessel_Y0_0[7] = {
0.000000000000000000e+00,
3.651397578300000000e+10,
5.477096367400000000e+10,
7.302795156600000000e+10,
9.420798334200000000e+10,
1.153880151180000000e+11,
1.577480786710000000e+11,
};
static double array_maxE_sf_bessel_Y0_0[6] = {
5.085163909552497391e-02,
5.085377435614243935e-02,
5.085519786438316703e-02,
5.085673530895586103e-02,
5.085838667318195311e-02,
5.086086373148118039e-02,
};
double accuracy_improve_patch_of_gsl_sf_bessel_Y0_0(double x)
{
 long int n = 157748078672;
 int len_glob = 6;
 double ulp_x = 4.440892098500626e-16;
 double x_0 = 3.95764598838955;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_Y0_0[idx])&&(n_x<array_idx_sf_bessel_Y0_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_Y0_0[idx+1])*(n_x-array_idx_sf_bessel_Y0_0[idx])*array_maxE_sf_bessel_Y0_0[idx];
         return (x-array_x_sf_bessel_Y0_0[idx])/ulp_x*array_detla_sf_bessel_Y0_0[idx]+array_y_sf_bessel_Y0_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_Y0_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_Y0_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_Y0_0[idx]){
         return array_y_sf_bessel_Y0_0[idx];
     }
     else{
         return array_e_y_sf_bessel_Y0_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_Y0(double x)
{
if(x<=3.9577160426091624){
 return accuracy_improve_patch_of_gsl_sf_bessel_Y0_0(x);
}
}
