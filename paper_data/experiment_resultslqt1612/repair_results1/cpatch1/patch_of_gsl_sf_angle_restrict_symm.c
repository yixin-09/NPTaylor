static double array_x_sf_angle_restrict_symm_0[2] = {
-2.261946710584650759e+02,
-2.261946710584523430e+02,
};
static double array_y_sf_angle_restrict_symm_0[2] = {
3.723916638426495072e-14,
1.277016499120526047e-11,
};
static double array_e_y_sf_angle_restrict_symm_0[2] = {
1.277016499120526047e-11,
3.051757802013061941e-05,
};
static double array_detla_sf_angle_restrict_symm_0[2] = {
2.842170943040400743e-14,
2.842170943040400743e-14,
};
static double array_idx_sf_angle_restrict_symm_0[3] = {
0.000000000000000000e+00,
4.480000000000000000e+02,
1.073741819000000000e+09,
};
static double array_maxE_sf_angle_restrict_symm_0[2] = {
-0.000000000000000000e+00,
-0.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_angle_restrict_symm_0(double x)
{
 long int n = 1073741820;
 int len_glob = 2;
 double ulp_x = 2.842170943040401e-14;
 double x_0 = -226.19467105846508;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_angle_restrict_symm_0[idx])&&(n_x<array_idx_sf_angle_restrict_symm_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_angle_restrict_symm_0[idx+1])*(n_x-array_idx_sf_angle_restrict_symm_0[idx])*array_maxE_sf_angle_restrict_symm_0[idx];
         return (x-array_x_sf_angle_restrict_symm_0[idx])/ulp_x*array_detla_sf_angle_restrict_symm_0[idx]+array_y_sf_angle_restrict_symm_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_angle_restrict_symm_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_angle_restrict_symm_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_angle_restrict_symm_0[idx]){
         return array_y_sf_angle_restrict_symm_0[idx];
     }
     else{
         return array_e_y_sf_angle_restrict_symm_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_angle_restrict_symm(double x)
{
if(x<=-226.1946405408871){
 return accuracy_improve_patch_of_gsl_sf_angle_restrict_symm_0(x);
}
}
