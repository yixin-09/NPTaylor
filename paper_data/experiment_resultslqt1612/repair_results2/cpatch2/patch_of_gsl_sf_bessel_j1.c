static double array_x_sf_bessel_j1_0[17] = {
4.493340628202902032e+00,
4.493375043055983120e+00,
4.493375043055983120e+00,
4.493392250482523664e+00,
4.493400854195794381e+00,
4.493405156052428850e+00,
4.493407306980746085e+00,
4.493409457909064209e+00,
4.493409457909064209e+00,
4.493411431854255511e+00,
4.493413405799445925e+00,
4.493417353689828531e+00,
4.493425249470591964e+00,
4.493433145251356287e+00,
4.493441041032120609e+00,
4.493448936812884043e+00,
4.493456832593648365e+00,
};
static double array_y_sf_bessel_j1_0[17] = {
1.495235582673361591e-05,
7.476120657149520115e-06,
7.476120657149520115e-06,
3.738046014183588341e-06,
1.869019428353055382e-06,
9.345088196045045701e-07,
4.672541862282403294e-07,
-7.218300729428198140e-18,
-7.218300729428198140e-18,
-4.288070874296780105e-07,
-8.576137979099435903e-07,
-1.715226089004566978e-06,
-3.430446149772518423e-06,
-5.145660182632390887e-06,
-6.860868187326680848e-06,
-8.576070163597886902e-06,
-1.029126611176732337e-05,
};
static double array_e_y_sf_bessel_j1_0[17] = {
1.121422392822229332e-05,
1.121422392822229332e-05,
5.607079757207902766e-06,
5.607079757207902766e-06,
3.738046014183588341e-06,
1.869019428353055382e-06,
9.345088196045045701e-07,
4.672541862282403294e-07,
-4.288070874296780105e-07,
-8.576137979099435903e-07,
-1.715226089004566978e-06,
-3.430446149772518423e-06,
-5.145660182632390887e-06,
-6.860868187326680848e-06,
-8.576070163597886902e-06,
-1.029126611176732337e-05,
-1.372163992277089043e-05,
};
static double array_detla_sf_bessel_j1_0[17] = {
-1.929473924777490547e-16,
-1.929459148440615808e-16,
-1.929448065972369307e-16,
-1.929440677545222168e-16,
-1.929433289031841686e-16,
-1.929427747592909314e-16,
-1.929424976854579984e-16,
-1.929423129688507160e-16,
-1.929421358511857442e-16,
-1.929419663325198318e-16,
-1.929417120536131445e-16,
-1.929412034926224108e-16,
-1.929405254052492012e-16,
-1.929398473106135410e-16,
-1.929391692087154054e-16,
-1.929384910995545972e-16,
-1.929374739212888903e-16,
};
static double array_idx_sf_bessel_j1_0[18] = {
0.000000000000000000e+00,
1.937383993900000000e+10,
3.874767987800000000e+10,
4.843459984700000000e+10,
5.812151981700000000e+10,
6.780843978700000000e+10,
7.265189977100000000e+10,
7.507362976300000000e+10,
7.749535975600000000e+10,
7.971782446300000000e+10,
8.194028916900000000e+10,
8.638521858300000000e+10,
9.527507740900000000e+10,
1.041649362360000000e+11,
1.130547950630000000e+11,
1.219446538890000000e+11,
1.308345127160000000e+11,
1.486142303690000000e+11,
};
static double array_maxE_sf_bessel_j1_0[17] = {
4.834099417574022617e-02,
4.834212267752945047e-02,
4.834296905314056092e-02,
4.834353325972592930e-02,
4.834409754494858186e-02,
4.834452069300813198e-02,
4.834473242869707227e-02,
4.834487338569935788e-02,
4.834500863529769127e-02,
4.834513805021559835e-02,
4.834533229479962696e-02,
4.834572062100103707e-02,
4.834623842185058196e-02,
4.834675623357673896e-02,
4.834727394748720292e-02,
4.834779185702903909e-02,
4.834856849853740729e-02,
};
double accuracy_improve_patch_of_gsl_sf_bessel_j1_0(double x)
{
 long int n = 148614230370;
 int len_glob = 17;
 double ulp_x = 8.881784197001252e-16;
 double x_0 = 4.493340628202902;
 double compen = 0.0;
 double n_x = ((x-x_0)/ulp_x);
 int idx = floor(len_glob/2);
 while((idx>=0)&&(idx<len_glob)){
     if((n_x>array_idx_sf_bessel_j1_0[idx])&&(n_x<array_idx_sf_bessel_j1_0[idx+1])){
         compen = ulp_x*ulp_x * (n_x-array_idx_sf_bessel_j1_0[idx+1])*(n_x-array_idx_sf_bessel_j1_0[idx])*array_maxE_sf_bessel_j1_0[idx];
         return (x-array_x_sf_bessel_j1_0[idx])/ulp_x*array_detla_sf_bessel_j1_0[idx]+array_y_sf_bessel_j1_0[idx]+compen;
     }
     else if(n_x<array_idx_sf_bessel_j1_0[idx]){
         idx = idx - 1;
     }
     else if(n_x>array_idx_sf_bessel_j1_0[idx+1]){
         idx = idx + 1;
     }
     else if(x==array_x_sf_bessel_j1_0[idx]){
         return array_y_sf_bessel_j1_0[idx];
     }
     else{
         return array_e_y_sf_bessel_j1_0[idx];
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_j1(double x)
{
if(x<=4.493472624155176){
 return accuracy_improve_patch_of_gsl_sf_bessel_j1_0(x);
}
}
