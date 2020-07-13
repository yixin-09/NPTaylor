#include "eft_patch.h"
static double array_idx_sf_airy_Bi_scaled_0[4] = {
-4.221090676344383610e+02,
-4.221008328470001629e+02,
-4.220929200687095317e+02,
-4.220853292995664674e+02,
};
static double array_cof_float_sf_airy_Bi_scaled_0[3][8] = {
{
-3.762187121388220839e+04,
2.180333188053828053e+03,
3.743057759675096349e+03,
-1.553815872579734787e+02,
-1.773415162360328168e+02,
4.423307077582606439e+00,
2.520766786745870736e+00,
-2.095832837335866172e-02,
},
{
-3.815780472943288623e+04,
-8.995157752831403286e+00,
3.796826540148999811e+03,
2.131065567305953645e-01,
-1.799031549473994858e+02,
-1.103531711986189677e-11,
2.557278676109173254e+00,
5.228809756556051548e-14,
},
{
-3.768654544731362694e+04,
-2.028098092364341937e+03,
3.750342148912102402e+03,
1.437261131663026958e+02,
-1.777134703121812720e+02,
-4.080150407093294795e+00,
2.526244085994103550e+00,
1.933312264392783933e-02,
}
};
static double array_cof_err_sf_airy_Bi_scaled_0[3][8] = {
{
-1.730349129856728456e-12,
2.928761625291954808e-14,
1.971362099472051564e-13,
5.820697471254631757e-15,
-9.610837119327207668e-15,
-1.769419931503479637e-17,
-5.210172478327842135e-17,
7.797106891525696702e-19,
},
{
1.903686462073266809e-12,
5.763182281952989339e-16,
-5.698098444963627676e-14,
-3.519649135127473561e-18,
7.046220291313829189e-15,
1.859480393701605427e-28,
-3.258913840121758547e-17,
-1.304986990420984012e-30,
},
{
1.923805041476844088e-12,
3.362939150571198728e-14,
3.602861167646303447e-14,
1.184931620410398706e-14,
1.628307540066178208e-15,
-3.524323754014177940e-16,
1.571139473754087456e-16,
6.480512284413887990e-19,
}
};
static double array_point_sf_airy_Bi_scaled_0[3] = {
-4.221049502407192904e+02,
-4.220967154532810355e+02,
-4.220891246841380280e+02,
};
static double array_cofidx_sf_airy_Bi_scaled_0[3] = {
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_airy_Bi_scaled_0(double x)
{
 int len_glob = 3;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_airy_Bi_scaled_0[idx])&&(x<=array_idx_sf_airy_Bi_scaled_0[idx+1])){
         double point = array_point_sf_airy_Bi_scaled_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_airy_Bi_scaled_0[idx];
         eft_tay1v(array_cof_float_sf_airy_Bi_scaled_0[idx],array_cof_err_sf_airy_Bi_scaled_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_airy_Bi_scaled_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_airy_Bi_scaled_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_airy_Bi_scaled(double x)
{
if(x<=-422.08532929956647){
 return accuracy_improve_patch_of_gsl_sf_airy_Bi_scaled_0(x);
}
}