static double array_x_sf_legendre_P2_0[2] = {
5.773502691896257311e-01,
5.773502691896257311e-01,
};
static double array_y_sf_legendre_P2_0[2] = {
-5.793758576800781262e-17,
-5.793758576800781262e-17,
};
static double array_e_y_sf_legendre_P2_0[2] = {
-7.627979876301384398e-06,
7.629394524779788908e-06,
};
static double array_detla_sf_legendre_P2_0[2] = {
1.922955352195253754e-16,
1.922970021876076906e-16,
};
static double array_idx_sf_legendre_P2_0[3] = {
0.000000000000000000e+00,
3.966800304300000000e+10,
7.934306005400000000e+10,
};
static double array_maxE_sf_legendre_P2_0[2] = {
1.500000000152338586e+00,
1.500000000054831695e+00,
};
double accuracy_improve_patch_of_gsl_sf_legendre_P2_0(double x)
{
 long int n = 79343060055;
 int len_glob = 2;
 double ulp_x = 1.1102230246251565e-16;
 double x_0 = 0.5773458651565938;
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
if(x<=0.5773546740058054){
 return accuracy_improve_patch_of_gsl_sf_legendre_P2_0(x);
}
}
