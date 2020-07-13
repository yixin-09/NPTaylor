static double array_x_sf_bessel_J1_0[7] = {
3.831673649909406620e+00,
3.831689810058459322e+00,
3.831697890132985673e+00,
3.831705970207512024e+00,
3.831705970207512024e+00,
3.831715441659108823e+00,
3.831724913110705621e+00,
};
static double array_y_sf_bessel_J1_0[7] = {
1.301735863241064660e-05,
6.508665591880791712e-06,
3.254329364818747255e-06,
1.173630282272863872e-16,
1.173630282272863872e-16,
-3.814711406662341816e-06,
-7.629413383714740072e-06,
};
static double array_e_y_sf_bessel_J1_0[7] = {
2.603482705453872762e-05,
1.301735863241064660e-05,
6.508665591880791712e-06,
3.254329364818747255e-06,
-3.814711406662341816e-06,
-7.629413383714740072e-06,
-1.525878904772732866e-05,
};
static double array_detla_sf_bessel_J1_0[7] = {
-1.788633646539610334e-16,
-1.788622332689583181e-16,
-1.788616675439367983e-16,
-1.788612903815339828e-16,
-1.788608807350522599e-16,
-1.788604386021747449e-16,
-1.788597753773249678e-16,
};
static double array_idx_sf_bessel_J1_0[8] = {
0.000000000000000000e+00,
7.277884125300000000e+10,
1.091682618790000000e+11,
1.273629721920000000e+11,
1.455576825050000000e+11,
1.668854954460000000e+11,
1.882133083870000000e+11,
2.308689342700000000e+11,
};
static double array_maxE_sf_bessel_J1_0[7] = {
5.254837769106501105e-02,
5.255226180996549340e-02,
5.255420379085047611e-02,
5.255549850531852918e-02,
5.255690466871468236e-02,
5.255842229262357862e-02,
5.256069869769627867e-02,
};
double accuracy_improve_patch_of_gsl_sf_bessel_J1_0(double x)
{
 long int n = 230868934271;
 int len_glob = 7;
 double ulp_x = 4.440892098500626e-16;
 double x_0 = 3.8316413296113008;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_J1_0[idx])&&(n_x<array_idx_sf_bessel_J1_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_J1_0[idx+1])*(n_x-array_idx_sf_bessel_J1_0[idx])*array_maxE_sf_bessel_J1_0[idx];
         return (x-array_x_sf_bessel_J1_0[idx])/ulp_x*array_detla_sf_bessel_J1_0[idx]+array_y_sf_bessel_J1_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_J1_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_J1_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_J1_0[idx]){
         return array_y_sf_bessel_J1_0[idx];
     }
     else{
         return array_e_y_sf_bessel_J1_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_J1(double x)
{
if(x<=3.8317438560138997){
 return accuracy_improve_patch_of_gsl_sf_bessel_J1_0(x);
}
}
