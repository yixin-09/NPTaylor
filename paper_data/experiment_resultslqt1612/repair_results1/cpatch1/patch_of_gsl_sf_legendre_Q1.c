static double array_x_sf_legendre_Q1_0[6] = {
8.335526779119158958e-01,
8.335555891787024274e-01,
8.335565596009646416e-01,
8.335565596009646416e-01,
8.335575300369271190e-01,
8.335585004728895964e-01,
};
static double array_y_sf_legendre_Q1_0[6] = {
-1.525878905429011622e-05,
-3.814727596690355385e-06,
-2.232976000570467603e-16,
-2.232976000570467603e-16,
3.814801674625224054e-06,
7.629623572510641578e-06,
};
static double array_e_y_sf_legendre_Q1_0[6] = {
-7.629434971120319415e-06,
-7.629434971120319415e-06,
-3.814727596690355385e-06,
3.814801674625224054e-06,
7.629623572510641578e-06,
1.525932803868626273e-05,
};
static double array_detla_sf_legendre_Q1_0[6] = {
4.364226222155529991e-16,
4.364260924554724872e-16,
4.364284059814560346e-16,
4.364307195482993721e-16,
4.364330331560033380e-16,
4.364365036166194340e-16,
};
static double array_idx_sf_legendre_Q1_0[7] = {
0.000000000000000000e+00,
1.748157335300000000e+10,
2.622236003000000000e+10,
3.496314670700000000e+10,
4.370405678500000000e+10,
5.244496686300000000e+10,
6.992678702000000000e+10,
};
static double array_maxE_sf_legendre_Q1_0[6] = {
1.073653628529548598e+01,
1.073670703095645962e+01,
1.073682086221599263e+01,
1.073693469660782185e+01,
1.073704853544646021e+01,
1.073721929607374292e+01,
};
double accuracy_improve_patch_of_gsl_sf_legendre_Q1_0(double x)
{
 long int n = 69926787021;
 int len_glob = 6;
 double ulp_x = 1.1102230246251565e-16;
 double x_0 = 0.8335526779119159;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_legendre_Q1_0[idx])&&(n_x<array_idx_sf_legendre_Q1_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_legendre_Q1_0[idx+1])*(n_x-array_idx_sf_legendre_Q1_0[idx])*array_maxE_sf_legendre_Q1_0[idx];
         return (x-array_x_sf_legendre_Q1_0[idx])/ulp_x*array_detla_sf_legendre_Q1_0[idx]+array_y_sf_legendre_Q1_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_legendre_Q1_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_legendre_Q1_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_legendre_Q1_0[idx]){
         return array_y_sf_legendre_Q1_0[idx];
     }
     else{
         return array_e_y_sf_legendre_Q1_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_legendre_Q1(double x)
{
if(x<=0.8335604413448147){
 return accuracy_improve_patch_of_gsl_sf_legendre_Q1_0(x);
}
}
