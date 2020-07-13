static double array_x_sf_bessel_j1_0[2] = {
4.493409457909064209e+00,
4.493409457909064209e+00,
};
static double array_y_sf_bessel_j1_0[2] = {
-7.218300729428198140e-18,
-7.218300729428198140e-18,
};
static double array_e_y_sf_bessel_j1_0[2] = {
8.673963720104619618e-08,
-2.085558319693935341e-08,
};
static double array_detla_sf_bessel_j1_0[2] = {
-1.929422377555338413e-16,
-1.929422164880009925e-16,
};
static double array_idx_sf_bessel_j1_0[3] = {
0.000000000000000000e+00,
4.495627200000000000e+08,
5.576550990000000000e+08,
};
static double array_maxE_sf_bessel_j1_0[2] = {
4.834493080468757475e-02,
4.834494699177572380e-02,
};
double accuracy_improve_patch_of_gsl_sf_bessel_j1_0(double x)
{
 long int n = 557655100;
 int len_glob = 2;
 double ulp_x = 8.881784197001252e-16;
 double x_0 = 4.493409058617158;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_j1_0[idx])&&(n_x<array_idx_sf_bessel_j1_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_j1_0[idx+1])*(n_x-array_idx_sf_bessel_j1_0[idx])*array_maxE_sf_bessel_j1_0[idx];
         return (x-array_x_sf_bessel_j1_0[idx])/ulp_x*array_detla_sf_bessel_j1_0[idx]+array_y_sf_bessel_j1_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_j1_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_j1_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_j1_0[idx]){
         return array_y_sf_bessel_j1_0[idx];
     }
     else{
         return array_e_y_sf_bessel_j1_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_j1(double x)
{
if(x<=4.493409553914383){
 return accuracy_improve_patch_of_gsl_sf_bessel_j1_0(x);
}
}
