static double array_x_sf_bessel_y1_0[10] = {
2.798244479882426194e+00,
2.798315262833156680e+00,
2.798350654308522145e+00,
2.798368350046204434e+00,
2.798377197915045578e+00,
2.798386045783887166e+00,
2.798386045783887166e+00,
2.798400654688415479e+00,
2.798415263592944235e+00,
2.798444481402000861e+00,
};
static double array_y_sf_bessel_y1_0[10] = {
-4.764052733031255227e-05,
-2.381966117970526152e-05,
-1.190967996857263731e-05,
-5.954802329076158866e-06,
-2.977391750790396092e-06,
9.865851509229451309e-18,
9.865851509229451309e-18,
4.915993671791825502e-06,
9.831936015833303956e-06,
1.966366672003817817e-05,
};
static double array_e_y_sf_bessel_y1_0[10] = {
-3.572994363356088074e-05,
-3.572994363356088074e-05,
-2.381966117970526152e-05,
-1.190967996857263731e-05,
-5.954802329076158866e-06,
-2.977391750790396092e-06,
4.915993671791825502e-06,
9.831936015833303956e-06,
1.966366672003817817e-05,
3.932651219476457713e-05,
};
static double array_detla_sf_bessel_y1_0[10] = {
1.494529868616674100e-16,
1.494492068896637040e-16,
1.494454269217068299e-16,
1.494425919482671530e-16,
1.494411744624317016e-16,
1.494402294722115283e-16,
1.494389768307777695e-16,
1.494374165384209481e-16,
1.494350761012619453e-16,
1.494303952317582101e-16,
};
static double array_idx_sf_bessel_y1_0[11] = {
0.000000000000000000e+00,
7.969451763300000000e+10,
1.593890352670000000e+11,
2.390835529010000000e+11,
2.789308117170000000e+11,
2.988544411250000000e+11,
3.187780705340000000e+11,
3.516743990290000000e+11,
3.845707275250000000e+11,
4.503633845150000000e+11,
5.819486984950000000e+11,
};
static double array_maxE_sf_bessel_y1_0[10] = {
-1.202513419372025216e-01,
-1.202512131666424872e-01,
-1.202510844531997219e-01,
-1.202509879782527047e-01,
-1.202509397431713312e-01,
-1.202509076176588171e-01,
-1.202508650293067777e-01,
-1.202508119981212514e-01,
-1.202507324458070848e-01,
-1.202505734694356843e-01,
};
double accuracy_improve_patch_of_gsl_sf_bessel_y1_0(double x)
{
 long int n = 581948698496;
 int len_glob = 10;
 double ulp_x = 4.440892098500626e-16;
 double x_0 = 2.798244479882426;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_y1_0[idx])&&(n_x<array_idx_sf_bessel_y1_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_y1_0[idx+1])*(n_x-array_idx_sf_bessel_y1_0[idx])*array_maxE_sf_bessel_y1_0[idx];
         return (x-array_x_sf_bessel_y1_0[idx])/ulp_x*array_detla_sf_bessel_y1_0[idx]+array_y_sf_bessel_y1_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_y1_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_y1_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_y1_0[idx]){
         return array_y_sf_bessel_y1_0[idx];
     }
     else{
         return array_e_y_sf_bessel_y1_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_y1(double x)
{
if(x<=2.798502917020114){
 return accuracy_improve_patch_of_gsl_sf_bessel_y1_0(x);
}
}