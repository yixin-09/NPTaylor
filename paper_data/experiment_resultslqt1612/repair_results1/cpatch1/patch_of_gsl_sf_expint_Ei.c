static double array_x_sf_expint_Ei_0[2] = {
3.725074107813666213e-01,
3.725074107813666213e-01,
};
static double array_y_sf_expint_Ei_0[2] = {
-5.119698936555684652e-17,
-5.119698936555684652e-17,
};
static double array_e_y_sf_expint_Ei_0[2] = {
-7.629394497300877969e-06,
7.629394510460177703e-06,
};
static double array_detla_sf_expint_Ei_0[2] = {
2.162837775442671479e-16,
2.162830641263861238e-16,
};
static double array_idx_sf_expint_Ei_0[3] = {
0.000000000000000000e+00,
3.527492715300000000e+10,
7.054997072300000000e+10,
};
static double array_maxE_sf_expint_Ei_0[2] = {
-3.281627023823301403e+00,
-3.281588708500070339e+00,
};
double accuracy_improve_patch_of_gsl_sf_expint_Ei_0(double x)
{
 long int n = 70549970724;
 int len_glob = 2;
 double ulp_x = 5.551115123125783e-17;
 double x_0 = 0.37250545262955076;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_expint_Ei_0[idx])&&(n_x<array_idx_sf_expint_Ei_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_expint_Ei_0[idx+1])*(n_x-array_idx_sf_expint_Ei_0[idx])*array_maxE_sf_expint_Ei_0[idx];
         return (x-array_x_sf_expint_Ei_0[idx])/ulp_x*array_detla_sf_expint_Ei_0[idx]+array_y_sf_expint_Ei_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_expint_Ei_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_expint_Ei_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_expint_Ei_0[idx]){
         return array_y_sf_expint_Ei_0[idx];
     }
     else{
         return array_e_y_sf_expint_Ei_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_expint_Ei(double x)
{
if(x<=0.3725093689396449){
 return accuracy_improve_patch_of_gsl_sf_expint_Ei_0(x);
}
}
