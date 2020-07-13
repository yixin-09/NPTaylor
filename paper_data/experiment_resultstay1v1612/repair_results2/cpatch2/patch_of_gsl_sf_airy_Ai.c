#include "eft_patch.h"
static double array_idx_sf_airy_Ai_0[26] = {
-2.375782781079786332e+02,
-2.375732082971240118e+02,
-2.375681384862693619e+02,
-2.375630686754147405e+02,
-2.375579988645601190e+02,
-2.375529290537054692e+02,
-2.375478592428508477e+02,
-2.375427894319962263e+02,
-2.375377196211416049e+02,
-2.375326498102869834e+02,
-2.375275799994323620e+02,
-2.375225101885777121e+02,
-2.375174403777230907e+02,
-2.375022316804675881e+02,
-2.374971623598185602e+02,
-2.374920930391695322e+02,
-2.374870237185205042e+02,
-2.374819543978714762e+02,
-2.374768850772224198e+02,
-2.374718157565733918e+02,
-2.374667464359243638e+02,
-2.374616771152753074e+02,
-2.374566077946262794e+02,
-2.374515384739772514e+02,
-2.374464691533282235e+02,
-2.374413998326791955e+02,
};
static double array_cof_float_sf_airy_Ai_0[25][8] = {
{
2.272346252194879980e+03,
5.498954628664837401e+02,
-2.871362951510723178e+02,
-4.623115509674219936e+01,
1.450822610872836016e+01,
1.167058518931956224e+00,
-1.221355843248609491e-01,
},
{
2.154900967431334266e+03,
6.172656284329009395e+02,
-2.723355327211305621e+02,
-5.190687410541042368e+01,
1.376153377764162400e+01,
1.310453803333286427e+00,
-1.158521258636345114e-01,
},
{
2.024319806363128009e+03,
6.808615267619028373e+02,
-2.558739931966087795e+02,
-5.726544349879329587e+01,
1.293088612262017456e+01,
1.445848196090494531e+00,
-1.088616022497539415e-01,
},
{
1.881401623964384953e+03,
7.402955326154498152e+02,
-2.378523149869790245e+02,
-6.227418780237628226e+01,
1.202135788404479300e+01,
1.572415729979593779e+00,
-1.012066919809516685e-01,
},
{
1.727020426065297215e+03,
7.952054770123696699e+02,
-2.183806446177784153e+02,
-6.690256991193058411e+01,
1.103850476040421924e+01,
1.689384359579276129e+00,
-9.293412642618228969e-02,
},
{
1.562120021382716232e+03,
8.452568530081497329e+02,
-1.975779631388870996e+02,
-7.112237717511239055e+01,
9.988329452544846276e+00,
1.796040668238105154e+00,
-8.409440437560515569e-02,
},
{
1.387708248719708990e+03,
8.901448522539255919e+02,
-1.755713587340925983e+02,
-7.490789326090246902e+01,
8.877244984331854383e+00,
1.891734216765623255e+00,
-7.474148362290715386e-02,
},
{
1.204850814672169236e+03,
9.295962199359959186e+02,
-1.524952499824965173e+02,
-7.823605476989685314e+01,
7.712035524035350420e+00,
1.975881507340133458e+00,
-6.493245146517057087e-02,
},
{
1.014664779558883993e+03,
9.633709168119237347e+02,
-1.284905645238477803e+02,
-8.108659163207418885e+01,
6.499814946036155838e+00,
2.047969538485505225e+00,
-5.472717613447652257e-02,
},
{
8.183117313554145085e+02,
9.912635782541882463e+02,
-1.037038781418007289e+02,
-8.344215043905801110e+01,
5.247983385729445338e+00,
2.107558929500683575e+00,
-4.418794128762554979e-02,
},
{
6.169906894185670581e+02,
1.013104761428523602e+03,
-7.828651953379390704e+01,
-8.528839996009476465e+01,
3.964182053425890029e+00,
2.154286595303098739e+00,
-3.337906578952118397e-02,
},
{
4.119307813040886117e+02,
1.028761973053519569e+03,
-5.239364622907726954e+01,
-8.661411820190163269e+01,
2.656246582849529858e+00,
2.187867955445875090e+00,
-2.236651110865927120e-02,
},
{
-5.887886137943299218e+03,
-4.383750405488668456e+00,
1.041183838455605155e+03,
1.845713207082292362e-01,
-8.767500811543338557e+01,
1.806036499934248477e-12,
2.214855848927702375e+00,
-1.520809860277674365e-14,
},
{
-4.204450201054061722e+02,
1.028235384376999036e+03,
5.274428230768938164e+01,
-8.659953554106452600e+01,
-2.655765547953277128e+00,
2.187874696835101496e+00,
2.236436998677967733e-02,
},
{
-6.252212398690057853e+02,
1.012324772218319481e+03,
7.861807384327492798e+01,
-8.526701821025999095e+01,
-3.963300525286223230e+00,
2.154303405238410019e+00,
3.337591728854644862e-02,
},
{
-8.261576034601389438e+02,
9.902409599968411840e+02,
1.040101038620055078e+02,
-8.341447067868182330e+01,
-5.246603948439398479e+00,
2.107591640488710727e+00,
4.418387051838278817e-02,
},
{
-1.022029066099930674e+03,
9.621194403682394523e+02,
1.287655411388214191e+02,
-8.105323734279339476e+01,
-6.497849501195001487e+00,
2.048024897508514020e+00,
5.472230651327036205e-02,
},
{
-1.211641685052465846e+03,
9.281325125632403115e+02,
1.527334571875565246e+02,
-7.819776416358575943e+01,
-7.709407077082295778e+00,
1.975967064636717785e+00,
6.492694891710981686e-02,
},
{
-1.393839895837377981e+03,
8.884882251942955236e+02,
1.757677462500141417e+02,
-7.486551015708859325e+01,
-8.873889298373859091e+00,
1.891858197375664297e+00,
7.473556013771853745e-02,
},
{
-1.567513552836978079e+03,
8.434291012191183654e+02,
1.977280157156748999e+02,
-7.107684052022742094e+01,
-9.984196548771036461e+00,
1.796211827791390103e+00,
8.408832083611128549e-02,
},
{
-1.731604691388328092e+03,
7.932306528856925070e+02,
2.184804415438809428e+02,
-6.685490204713997286e+01,
-1.103356024518112832e+01,
1.689611826040643638e+00,
9.292819463345221298e-02,
},
{
-1.885113970168549031e+03,
7.381996957668357027e+02,
2.378985834759943714e+02,
-6.222548159932293288e+01,
-1.201558408482835461e+01,
1.572708833236854176e+00,
1.012012758119673062e-01,
},
{
-2.027106754692863205e+03,
6.786724722004977366e+02,
2.558641550778030478e+02,
-5.721684849491838065e+01,
-1.292428301665912294e+01,
1.446216287446767446e+00,
1.088571178928505606e-01,
},
{
-2.156718804888279919e+03,
6.150125956243800829e+02,
2.722677439219889948e+02,
-5.185958178140458585e+01,
-1.375411969957050928e+01,
1.310906067114096629e+00,
1.158490410881200111e-01,
},
{
-2.273161532137167342e+03,
5.476088283877294316e+02,
2.870094775264712439e+02,
-4.618638344835515852e+01,
-1.450003822540497644e+01,
1.167603778546099980e+00,
1.221344167528711361e-01,
}
};
static double array_cof_err_sf_airy_Ai_0[25][8] = {
{
-7.581565600096646048e-14,
8.464119262770638160e-15,
1.776048355580721965e-14,
2.292957491911685431e-15,
6.408089532024329544e-17,
-3.296351422086279063e-17,
-5.204418406513943823e-18,
},
{
6.445078311124797665e-14,
-2.555441568597582686e-16,
-2.310672088638836831e-14,
3.141586897953025071e-15,
6.559204017992397150e-16,
-1.009399872670367619e-16,
6.936472617671105969e-18,
},
{
-9.076712827697169286e-15,
-5.326008334845326232e-14,
1.594129366410051444e-15,
1.899287764637471232e-15,
-4.865689633046311474e-16,
9.573924384057481932e-17,
-6.870189883999111793e-18,
},
{
-7.445399234672754297e-14,
-3.455676485268308021e-14,
-6.119726421164729857e-15,
6.418744913057844914e-16,
4.679292360881883900e-16,
-3.388114418202035782e-17,
-2.678144188017688355e-18,
},
{
-8.235572564961068139e-14,
5.674938631453666454e-15,
1.159095470947548113e-14,
-5.667738470369360012e-15,
-6.818793372402475982e-16,
-4.554761911258803818e-17,
1.605013637783603659e-18,
},
{
8.277230082690807019e-14,
-5.238401521528437849e-14,
-3.636737583806397881e-15,
2.975336199572119715e-15,
-6.194038361919170260e-16,
-9.287449893178732270e-17,
-1.551629375540563082e-18,
},
{
-1.136001093366117975e-13,
-2.312517322928596851e-14,
-5.464583300618408120e-15,
5.795108855101212254e-15,
2.846389451793914261e-16,
-5.674728967691192622e-17,
4.968327527317113418e-18,
},
{
-4.166594292801192756e-15,
-1.282226958614919686e-14,
3.081443918919560247e-15,
2.918720794440894518e-15,
-5.427605151808881958e-17,
4.256698784528687802e-17,
-6.489769606275157989e-18,
},
{
-4.084881381311802138e-14,
-3.880950569760764184e-14,
-9.912500496023475195e-15,
-4.898692332339187455e-15,
3.415016887886168310e-16,
8.058518967642766894e-17,
7.980447646903082002e-19,
},
{
-4.473252577380487211e-14,
-2.243333909429646027e-14,
-3.581466920811311846e-15,
6.248495008823058952e-16,
-4.400749275115642277e-16,
2.160132406756712643e-16,
-2.951643880429752444e-18,
},
{
-3.359574008015631464e-14,
1.303360850990051938e-14,
-6.467613326427417367e-15,
-1.582056038212732020e-16,
1.960148980033934525e-16,
-1.694119497634801929e-16,
-1.606109787150865770e-18,
},
{
-1.687814528537799848e-14,
1.135177043882199356e-13,
2.372635250558367485e-15,
8.643294916420655245e-16,
-1.410186053566983927e-17,
1.915642199026677089e-16,
-1.656457619448825185e-18,
},
{
2.396840301763672744e-13,
-1.179172594223168154e-16,
2.232736525727273256e-15,
1.929118320523228590e-19,
3.019403312379172849e-15,
6.222606860354313609e-29,
-1.063410138164868323e-16,
-3.664196770027701718e-32,
},
{
7.314280586519156082e-15,
4.889326236146986243e-14,
1.152683696557692782e-15,
2.688056678080230020e-15,
1.707563271780509092e-16,
-1.507297744423358870e-16,
-1.271969174721238125e-18,
},
{
2.661324247174286952e-14,
-1.866294885956968454e-14,
6.755438790001480099e-15,
6.057149294717554223e-15,
1.027410502035334128e-16,
-5.824506756206807940e-17,
6.901955151668803851e-19,
},
{
-4.922638109683403022e-14,
-3.486618261244921590e-14,
-5.458021788779333954e-15,
-2.807695019718987587e-15,
-3.774536794172793238e-16,
-1.621199486753905649e-16,
-5.325846636386792289e-19,
},
{
5.650952212676466022e-14,
-3.636899743904118491e-14,
9.897981482180187199e-15,
-6.628841207502938537e-15,
2.621448325393213373e-16,
-5.197877323035776164e-17,
-3.332583377687598758e-18,
},
{
-8.126916668761906690e-14,
1.454650154509980641e-14,
-2.674081987218590334e-15,
2.484165702279820124e-15,
-2.849777209250686016e-16,
-5.040213840802742327e-17,
2.705471851855876093e-18,
},
{
7.669335497268387480e-14,
-1.241958208124971362e-14,
-3.128941776749792384e-16,
3.488527409940112817e-15,
-2.039292096079614592e-16,
-8.950897551984619576e-17,
4.115591782333603971e-18,
},
{
-1.135418021820487684e-13,
1.997946180669302676e-14,
-1.259914426616843856e-14,
4.926388624307379009e-17,
2.553541457387444645e-16,
5.283232165193149184e-17,
-4.646332279177536410e-18,
},
{
3.218824556548318912e-14,
-7.341798983164952245e-15,
-1.135685688783676934e-14,
-6.818685266883929643e-15,
7.053084978105687117e-16,
-8.321504148997691547e-17,
-5.087323606618230606e-18,
},
{
1.223634483044343476e-14,
1.820246420107311470e-14,
2.841270034952757992e-15,
-3.844356401180874236e-17,
3.065502493272521240e-17,
-8.282468848280075828e-17,
4.487991172213003302e-20,
},
{
-4.003819756039253887e-14,
-6.849071467419324787e-15,
8.515429872869008650e-15,
6.263565895370832156e-16,
-4.408533825132244022e-16,
3.752670206160987780e-17,
-4.654410366882600290e-18,
},
{
1.049281505865610532e-13,
1.487128514086913657e-15,
8.310641919503627552e-15,
2.511292270187051360e-15,
-1.414034111205469811e-16,
-1.349405300938841499e-17,
-2.946124069633094821e-18,
},
{
-7.781005248551700468e-14,
-3.577442619969870572e-14,
2.102844501722061472e-14,
3.364999190079895240e-15,
4.615670241948328146e-16,
5.038222811021093286e-17,
5.775001739773064708e-18,
}
};
static double array_point_sf_airy_Ai_0[25] = {
-2.375757432025513367e+02,
-2.375706733916966868e+02,
-2.375656035808420370e+02,
-2.375605337699874440e+02,
-2.375554639591327941e+02,
-2.375503941482781443e+02,
-2.375453243374235512e+02,
-2.375402545265689014e+02,
-2.375351847157143084e+02,
-2.375301149048596585e+02,
-2.375250450940050371e+02,
-2.375199752831504156e+02,
-2.375098356614411443e+02,
-2.374996970201430599e+02,
-2.374946276994940604e+02,
-2.374895583788450040e+02,
-2.374844890581960044e+02,
-2.374794197375469480e+02,
-2.374743504168978916e+02,
-2.374692810962488920e+02,
-2.374642117755998356e+02,
-2.374591424549507792e+02,
-2.374540731343017796e+02,
-2.374490038136527232e+02,
-2.374439344930037237e+02,
};
static double array_cofidx_sf_airy_Ai_0[25] = {
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
7.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
6.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_airy_Ai_0(double x)
{
 int len_glob = 25;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_airy_Ai_0[idx])&&(x<=array_idx_sf_airy_Ai_0[idx+1])){
         double point = array_point_sf_airy_Ai_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_airy_Ai_0[idx];
         eft_tay1v(array_cof_float_sf_airy_Ai_0[idx],array_cof_err_sf_airy_Ai_0[idx],point,x,&res,length);
         return res;
     }
     else if(x<array_idx_sf_airy_Ai_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_airy_Ai_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_airy_Ai(double x)
{
if(x<=-237.4413998326792){
 return accuracy_improve_patch_of_gsl_sf_airy_Ai_0(x);
}
}