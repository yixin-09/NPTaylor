static double array_x_sf_Chi_0[3] = {
5.238225713898643221e-01,
5.238225713898643221e-01,
5.238245407643808971e-01,
};
static double array_y_sf_Chi_0[3] = {
-1.836318268327936040e-16,
-1.836318268327936040e-16,
4.287319355986743013e-06,
};
static double array_e_y_sf_Chi_0[3] = {
-7.629394521241749228e-06,
4.287319355986743013e-06,
8.574626651555466671e-06,
};
static double array_detla_sf_Chi_0[3] = {
2.416959918864843798e-16,
2.416950469847437824e-16,
2.416943670892136913e-16,
};
static double array_idx_sf_Chi_0[4] = {
0.000000000000000000e+00,
3.156607795400000000e+10,
4.930462663200000000e+10,
6.704317530900000000e+10,
};
static double array_maxE_sf_Chi_0[3] = {
-1.554822974306490746e+00,
-1.554803747343886489e+00,
-1.554789913038158167e+00,
};
double accuracy_improve_patch_of_gsl_sf_Chi_0(double x)
{
 long int n = 67043175310;
 int len_glob = 3;
 double ulp_x = 1.1102230246251565e-16;
 double x_0 = 0.5238190668512102;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_Chi_0[idx])&&(n_x<array_idx_sf_Chi_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_Chi_0[idx+1])*(n_x-array_idx_sf_Chi_0[idx])*array_maxE_sf_Chi_0[idx];
         return (x-array_x_sf_Chi_0[idx])/ulp_x*array_detla_sf_Chi_0[idx]+array_y_sf_Chi_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_Chi_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_Chi_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_Chi_0[idx]){
         return array_y_sf_Chi_0[idx];
     }
     else{
         return array_e_y_sf_Chi_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_Chi(double x)
{
if(x<=0.5238265101388974){
 return accuracy_improve_patch_of_gsl_sf_Chi_0(x);
}
}
