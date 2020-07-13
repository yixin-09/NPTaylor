static double array_x_sf_angle_restrict_symm_0[2] = {
-2.261946710584622053e+02,
-2.261946710584523430e+02,
};
static double array_y_sf_angle_restrict_symm_0[2] = {
2.907831818855069891e-12,
1.277016499120526047e-11,
};
static double array_e_y_sf_angle_restrict_symm_0[2] = {
1.277016499120526047e-11,
4.882812496677569236e-04,
};
static double array_detla_sf_angle_restrict_symm_0[2] = {
2.842170943040400743e-14,
2.842170943040400743e-14,
};
static double array_idx_sf_angle_restrict_symm_0[3] = {
0.000000000000000000e+00,
3.470000000000000000e+02,
1.717986907000000000e+10,
};
static double array_maxE_sf_angle_restrict_symm_0[2] = {
-0.000000000000000000e+00,
-0.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_angle_restrict_symm_0(double x)
{
 long int n = 17179869071;
 int len_glob = 2;
 double ulp_x = 2.842170943040401e-14;
 double x_0 = -226.1946710584622;
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
if(x<=-226.19418277721545){
 return accuracy_improve_patch_of_gsl_sf_angle_restrict_symm_0(x);
}
}
