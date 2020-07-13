static double array_x_sf_eta_0[2] = {
-2.000017897772433884e+00,
-2.000000000000000000e+00,
};
static double array_y_sf_eta_0[2] = {
-3.814697241682906473e-06,
0.000000000000000000e+00,
};
static double array_e_y_sf_eta_0[2] = {
-1.907353532379386207e-06,
-1.907353532379386207e-06,
};
static double array_detla_sf_eta_0[2] = {
9.465208744899586643e-17,
9.465257491993849398e-17,
};
static double array_idx_sf_eta_0[3] = {
0.000000000000000000e+00,
2.015110031600000000e+10,
4.030220063200000000e+10,
};
static double array_maxE_sf_eta_0[2] = {
6.133128609167538825e-02,
6.133056826270258388e-02,
};
double accuracy_improve_patch_of_gsl_sf_eta_0(double x)
{
 long int n = 40302200633;
 int len_glob = 2;
 double ulp_x = 4.440892098500626e-16;
 double x_0 = -2.000017897772434;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_eta_0[idx])&&(n_x<array_idx_sf_eta_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_eta_0[idx+1])*(n_x-array_idx_sf_eta_0[idx])*array_maxE_sf_eta_0[idx];
         return (x-array_x_sf_eta_0[idx])/ulp_x*array_detla_sf_eta_0[idx]+array_y_sf_eta_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_eta_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_eta_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_eta_0[idx]){
         return array_y_sf_eta_0[idx];
     }
     else{
         return array_e_y_sf_eta_0[idx];
     }
 }
}
static double array_x_sf_eta_1[2] = {
-1.999999999999999778e+00,
-1.999991051205900039e+00,
};
static double array_y_sf_eta_1[2] = {
4.732640932675400909e-17,
1.907343721499648460e-06,
};
static double array_e_y_sf_eta_1[2] = {
1.907343721499648460e-06,
3.814697265548345636e-06,
};
static double array_detla_sf_eta_1[2] = {
4.732653119133341809e-17,
4.732677491859050814e-17,
};
static double array_idx_sf_eta_1[3] = {
0.000000000000000000e+00,
4.030178577300000000e+10,
8.060357154500000000e+10,
};
static double array_maxE_sf_eta_1[2] = {
6.132985044239903927e-02,
6.132913266515455097e-02,
};
double accuracy_improve_patch_of_gsl_sf_eta_1(double x)
{
 long int n = 80603571546;
 int len_glob = 2;
 double ulp_x = 2.220446049250313e-16;
 double x_0 = -1.9999999999999998;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_eta_1[idx])&&(n_x<array_idx_sf_eta_1[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_eta_1[idx+1])*(n_x-array_idx_sf_eta_1[idx])*array_maxE_sf_eta_1[idx];
         return (x-array_x_sf_eta_1[idx])/ulp_x*array_detla_sf_eta_1[idx]+array_y_sf_eta_1[idx]+compen;
     }
     else if(n_x<array_idx_sf_eta_1[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_eta_1[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_eta_1[idx]){
         return array_y_sf_eta_1[idx];
     }
     else{
         return array_e_y_sf_eta_1[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_eta(double x)
{
if(x<=-2.0){
 return accuracy_improve_patch_of_gsl_sf_eta_0(x);
}
if(x<=-1.9999821024118005){
 return accuracy_improve_patch_of_gsl_sf_eta_1(x);
}
}
