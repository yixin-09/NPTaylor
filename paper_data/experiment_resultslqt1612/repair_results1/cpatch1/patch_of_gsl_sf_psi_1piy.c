static double array_x_sf_psi_1piy_0[8] = {
8.837875168873026155e-01,
8.838040773907436254e-01,
8.838123576424641303e-01,
8.838206378941846353e-01,
8.838206378941846353e-01,
8.838275664034118684e-01,
8.838344949126389904e-01,
8.838483519310933456e-01,
};
static double array_y_sf_psi_1piy_0[8] = {
-2.762096070328816545e-05,
-1.381043846882579883e-05,
-6.905208763513654434e-06,
-9.180893450362987341e-17,
-9.180893450362987341e-17,
5.777924511480507244e-06,
1.155583435985836814e-05,
2.311161006632748223e-05,
};
static double array_e_y_sf_psi_1piy_0[8] = {
-5.524225645482672560e-05,
-2.762096070328816545e-05,
-1.381043846882579883e-05,
-6.905208763513654434e-06,
5.777924511480507244e-06,
1.155583435985836814e-05,
2.311161006632748223e-05,
4.622298551023460908e-05,
};
static double array_detla_sf_psi_1piy_0[8] = {
9.258715661172634876e-17,
9.258631430750003414e-17,
9.258589313388186295e-17,
9.258561234327821719e-17,
9.258535446672701055e-17,
9.258511950545648715e-17,
9.258476705494916612e-17,
9.258406212383214658e-17,
};
static double array_idx_sf_psi_1piy_0[9] = {
0.000000000000000000e+00,
2.983275085050000000e+11,
4.474912627570000000e+11,
5.220731398830000000e+11,
5.966550170090000000e+11,
6.590614801570000000e+11,
7.214679433040000000e+11,
8.462808695990000000e+11,
1.095906722189000000e+12,
};
static double array_maxE_sf_psi_1piy_0[8] = {
-1.527043071628047810e-01,
-1.527143332318560698e-01,
-1.527193456110993441e-01,
-1.527226869419422428e-01,
-1.527257554487244928e-01,
-1.527285511212486957e-01,
-1.527327441967667454e-01,
-1.527411297174788574e-01,
};
double accuracy_improve_patch_of_gsl_sf_psi_1piy_0(double x)
{
 long int n = 1095906722190;
 int len_glob = 8;
 double ulp_x = 1.1102230246251565e-16;
 double x_0 = 0.8837543958804205;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_psi_1piy_0[idx])&&(n_x<array_idx_sf_psi_1piy_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_psi_1piy_0[idx+1])*(n_x-array_idx_sf_psi_1piy_0[idx])*array_maxE_sf_psi_1piy_0[idx];
         return (x-array_x_sf_psi_1piy_0[idx])/ulp_x*array_detla_sf_psi_1piy_0[idx]+array_y_sf_psi_1piy_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_psi_1piy_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_psi_1piy_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_psi_1piy_0[idx]){
         return array_y_sf_psi_1piy_0[idx];
     }
     else{
         return array_e_y_sf_psi_1piy_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_psi_1piy(double x)
{
if(x<=0.8838760659680021){
 return accuracy_improve_patch_of_gsl_sf_psi_1piy_0(x);
}
}
