static double array_x_sf_clausen_0[8] = {
-3.141636681146387389e+00,
-3.141614667368090252e+00,
-3.141603660478941684e+00,
-3.141592653589793116e+00,
-3.141592653589793116e+00,
-3.141581650263285841e+00,
-3.141570646936778566e+00,
-3.141548640283763572e+00,
};
static double array_y_sf_clausen_0[8] = {
3.051757671662266543e-05,
1.525878835960239219e-05,
7.629394179925440583e-06,
-8.488604760107495465e-17,
-8.488604760107495465e-17,
-7.626924745327617163e-06,
-1.525384949023729641e-05,
-3.050769897803310844e-05,
};
static double array_e_y_sf_clausen_0[8] = {
6.103515623838991590e-05,
3.051757671662266543e-05,
1.525878835960239219e-05,
7.629394179925440583e-06,
-7.626924745327617163e-06,
-1.525384949023729641e-05,
-3.050769897803310844e-05,
-6.103515623640495857e-05,
};
static double array_detla_sf_clausen_0[8] = {
-3.078191834735881575e-16,
-3.078191836618956651e-16,
-3.078191837089725173e-16,
-3.078191837224230395e-16,
-3.078191837224244693e-16,
-3.078191837089826739e-16,
-3.078191836619362421e-16,
-3.078191834736346510e-16,
};
static double array_idx_sf_clausen_0[9] = {
0.000000000000000000e+00,
9.914125291800000000e+10,
1.487118747860000000e+11,
1.734971857200000000e+11,
1.982824966540000000e+11,
2.230597852330000000e+11,
2.478370738120000000e+11,
2.973916509710000000e+11,
3.965649933010000000e+11,
};
static double array_maxE_sf_clausen_0[8] = {
-8.255170537528151644e-06,
-4.127565058783016918e-06,
-2.063782529391508459e-06,
-6.879368318134007599e-07,
6.877107624655510858e-07,
2.063104303235413574e-06,
4.126264574626772426e-06,
8.254272892286975005e-06,
};
double accuracy_improve_patch_of_gsl_sf_clausen_0(double x)
{
 long int n = 396564993302;
 int len_glob = 8;
 double ulp_x = 4.440892098500626e-16;
 double x_0 = -3.1416807087070593;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_clausen_0[idx])&&(n_x<array_idx_sf_clausen_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_clausen_0[idx+1])*(n_x-array_idx_sf_clausen_0[idx])*array_maxE_sf_clausen_0[idx];
         return (x-array_x_sf_clausen_0[idx])/ulp_x*array_detla_sf_clausen_0[idx]+array_y_sf_clausen_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_clausen_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_clausen_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_clausen_0[idx]){
         return array_y_sf_clausen_0[idx];
     }
     else{
         return array_e_y_sf_clausen_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_clausen(double x)
{
if(x<=-3.14150459847253){
 return accuracy_improve_patch_of_gsl_sf_clausen_0(x);
}
}
