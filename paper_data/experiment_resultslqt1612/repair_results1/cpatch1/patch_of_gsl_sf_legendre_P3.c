static double array_x_sf_legendre_P3_0[4] = {
7.745915829283659404e-01,
7.745966692414834043e-01,
7.745966692414834043e-01,
7.745992123479418234e-01,
};
static double array_y_sf_legendre_P3_0[4] = {
-1.525878905814407886e-05,
8.172618520478209880e-17,
8.172618520478209880e-17,
7.629356947523615395e-06,
};
static double array_e_y_sf_legendre_P3_0[4] = {
-7.629432102364984745e-06,
-7.629432102364984745e-06,
7.629356947523615395e-06,
1.525878903949883556e-05,
};
static double array_detla_sf_legendre_P3_0[4] = {
3.330619865348982201e-16,
3.330652671009373272e-16,
3.330685476454331854e-16,
3.330718281683858932e-16,
};
static double array_idx_sf_legendre_P3_0[5] = {
0.000000000000000000e+00,
2.290671786100000000e+10,
4.581343572100000000e+10,
6.871970231800000000e+10,
9.162596891500000000e+10,
};
static double array_maxE_sf_legendre_P3_0[4] = {
5.809446409410495171e+00,
5.809465482038821627e+00,
5.809484555856104571e+00,
5.809503629776426870e+00,
};
double accuracy_improve_patch_of_gsl_sf_legendre_P3_0(double x)
{
 long int n = 91625968916;
 int len_glob = 4;
 double ulp_x = 1.1102230246251565e-16;
 double x_0 = 0.7745915829283659;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_legendre_P3_0[idx])&&(n_x<array_idx_sf_legendre_P3_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_legendre_P3_0[idx+1])*(n_x-array_idx_sf_legendre_P3_0[idx])*array_maxE_sf_legendre_P3_0[idx];
         return (x-array_x_sf_legendre_P3_0[idx])/ulp_x*array_detla_sf_legendre_P3_0[idx]+array_y_sf_legendre_P3_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_legendre_P3_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_legendre_P3_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_legendre_P3_0[idx]){
         return array_y_sf_legendre_P3_0[idx];
     }
     else{
         return array_e_y_sf_legendre_P3_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_legendre_P3(double x)
{
if(x<=0.7746017554544002){
 return accuracy_improve_patch_of_gsl_sf_legendre_P3_0(x);
}
}
