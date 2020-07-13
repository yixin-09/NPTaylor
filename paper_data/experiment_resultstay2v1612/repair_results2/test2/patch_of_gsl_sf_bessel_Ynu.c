#include "eft_patch.h"
static double array_idx_sf_bessel_Ynu_0[10] = {
6.047000974755841440e+01,
6.047790466684463695e+01,
6.048579958613085950e+01,
6.049369450541708204e+01,
6.050158942470330459e+01,
6.050948434398953424e+01,
6.051737926327575678e+01,
6.052527418256197933e+01,
6.053316910184820188e+01,
6.054106402113442442e+01,
};
static double array_cof_float_sf_bessel_Ynu_0[9][37] = {
{
-2.969882284679197434e-06,
1.953522500190196485e-06,
2.097181283960195789e-04,
-1.447518677463924349e-05,
-7.081184272856037160e-03,
-3.805938282556846114e-04,
7.195247820727389276e-02,
1.410300681899768788e-16,
2.369520451282911566e-05,
-1.755397679014191769e-05,
-1.196395987722210282e-03,
1.635405985036537512e-04,
2.423710669199351936e-02,
3.226687509440973457e-04,
-8.206639733926408831e-02,
-8.094946969399954189e-05,
6.268833504645277411e-05,
2.728795587700675023e-03,
-4.606226775906231458e-04,
-2.764748158328424882e-02,
1.741596304849051804e-04,
1.534783975022392644e-04,
-1.155988892329929313e-04,
-3.110224209335948724e-03,
4.983583557826267363e-04,
1.050993790581968880e-02,
-1.743907001690110832e-04,
1.171031079028125600e-04,
1.771299338460769011e-03,
-1.873379776527394572e-04,
1.187346293064824997e-04,
-6.212817891180636213e-05,
-4.031939009530671834e-04,
-4.484522960855482048e-05,
1.353641928928404370e-05,
7.247042169760900921e-06,
7.000000000000000000e+00,
},
{
-2.969268027758632620e-06,
1.953394758568149095e-06,
2.096850900610583923e-04,
-1.447843368173987379e-05,
-7.080414586653034463e-03,
-3.805347994178948090e-04,
7.194817175881261517e-02,
-3.201374054512337481e-16,
2.369008237722225669e-05,
-1.755195243023338327e-05,
-1.196196512922887822e-03,
1.635352838322613224e-04,
2.423425172522087126e-02,
3.226132849153610211e-04,
-8.206074237077294609e-02,
-8.093122244897383417e-05,
6.267882255563180112e-05,
2.728315670221008720e-03,
-4.605719617731799287e-04,
-2.764397399247321696e-02,
1.741288052554188068e-04,
1.534423998753324980e-04,
-1.155780109968834274e-04,
-3.109648999514462962e-03,
4.982824052032983888e-04,
1.050850955539436686e-02,
-1.743482309937368659e-04,
1.170791263418433678e-04,
1.770955852390292302e-03,
-1.873039513721508917e-04,
1.187046684000339500e-04,
-6.211414817260689687e-05,
-4.031121431728601902e-04,
-4.483352816608196501e-05,
1.353310753618531333e-05,
7.245090724998424959e-06,
7.000000000000000000e+00,
},
{
-2.968653950987576550e-06,
1.953266991355285651e-06,
2.096520598494685660e-04,
-1.448167874641122387e-05,
-7.079645054612488005e-03,
-3.804757857863963109e-04,
7.194386599749577615e-02,
-3.841657006901460304e-16,
2.368496181296423286e-05,
-1.754992827947151707e-05,
-1.195997090014802762e-03,
1.635299663554116111e-04,
2.423139737556510853e-02,
3.225578345321672294e-04,
-8.205508841462147429e-02,
-8.091298103950347268e-05,
6.266931202062430968e-05,
2.727835884139762773e-03,
-4.605212542178326886e-04,
-2.764046721191909875e-02,
1.740979889344921954e-04,
1.534064142127655966e-04,
-1.155571382358023402e-04,
-3.109073954710038848e-03,
4.982064737662751869e-04,
1.050708155490131038e-02,
-1.743057764468194969e-04,
1.170551520007680885e-04,
1.770612469169306237e-03,
-1.872699351318212065e-04,
1.186747181606519414e-04,
-6.210012207713596212e-05,
-4.030304108559432021e-04,
-4.482183101919439322e-05,
1.352979696102745734e-05,
7.243140017122701989e-06,
7.000000000000000000e+00,
},
{
-2.968040054297458097e-06,
1.953139198582269169e-06,
2.096190377585214614e-04,
-1.448492196958109295e-05,
-7.078875676688960172e-03,
-3.804167873558103358e-04,
7.193956092314213180e-02,
4.180102332722655100e-16,
2.367984281942822286e-05,
-1.754790433791214533e-05,
-1.195797718979489188e-03,
1.635246460753600288e-04,
2.422854364283014150e-02,
3.225023997886661037e-04,
-8.204943547051413155e-02,
-8.089474546316056285e-05,
6.265980344098814859e-05,
2.727356229407766879e-03,
-4.604705549231468092e-04,
-2.763696124134802645e-02,
1.740671815183598308e-04,
1.533704405093678103e-04,
-1.155362709478842593e-04,
-3.108499074858118213e-03,
4.981305614647502924e-04,
1.050565390421586386e-02,
-1.742633365217149759e-04,
1.170311848767049127e-04,
1.770269188755946082e-03,
-1.872359289276267743e-04,
1.186447785834126510e-04,
-6.208610062335842908e-05,
-4.029487039915753536e-04,
-4.481013816585200255e-05,
1.352648756325895226e-05,
7.241190045774611216e-06,
7.000000000000000000e+00,
},
{
-2.967426337619727729e-06,
1.953011380280531349e-06,
2.095860237854893884e-04,
-1.448816335221665884e-05,
-7.078106452837037640e-03,
-3.803578041199523235e-04,
7.193525653557057697e-02,
-1.848114762789310496e-16,
2.367472539598761204e-05,
-1.754588060561643740e-05,
-1.195598399798487041e-03,
1.635193229945426966e-04,
2.422569052681996030e-02,
3.224469806771658258e-04,
-8.204378353815562630e-02,
-8.087651571751800969e-05,
6.265029681629667073e-05,
2.726876705975866781e-03,
-4.604198638879995992e-04,
-2.763345608048622601e-02,
1.740363830043109095e-04,
1.533344787599702594e-04,
-1.155154091312880874e-04,
-3.107924359894166308e-03,
4.980546682921576686e-04,
1.050422660321342731e-02,
-1.742209112118817420e-04,
1.170072249667935524e-04,
1.769926011108362285e-03,
-1.872019327555138443e-04,
1.186148496633940584e-04,
-6.207208380924934691e-05,
-4.028670225690195194e-04,
-4.479844960401544937e-05,
1.352317934233031764e-05,
7.239240810595169068e-06,
7.000000000000000000e+00,
},
{
-2.966812800885888011e-06,
1.952883536480285432e-06,
2.095530179276463374e-04,
-1.449140289522430437e-05,
-7.077337383011320961e-03,
-3.802988360738611276e-04,
7.193095283460002043e-02,
-3.270770986070853353e-16,
2.366960954201628720e-05,
-1.754385708263727861e-05,
-1.195399132453349922e-03,
1.635139971151190741e-04,
2.422283802733866259e-02,
3.223915771927637962e-04,
-8.203813261725069472e-02,
-8.085829180015076465e-05,
6.264079214609998643e-05,
2.726397313794943784e-03,
-4.603691811107976607e-04,
-2.762995172906009683e-02,
1.740055933880470848e-04,
1.532985289594083740e-04,
-1.154945527841381005e-04,
-3.107349809753698257e-03,
4.979787942415762024e-04,
1.050279965176947883e-02,
-1.741785005107839533e-04,
1.169832722681449477e-04,
1.769582936184738902e-03,
-1.871679466113288382e-04,
1.185849313956785750e-04,
-6.205807163277122867e-05,
-4.027853665775476817e-04,
-4.478676533164731451e-05,
1.351987229768980979e-05,
7.237292311225735017e-06,
7.000000000000000000e+00,
},
{
-2.966199444027463528e-06,
1.952755667212214764e-06,
2.095200201822671935e-04,
-1.449464059953480357e-05,
-7.076568467166430634e-03,
-3.802398832120733991e-04,
7.192664982004942642e-02,
1.466401581177087545e-16,
2.366449525688835536e-05,
-1.754183376903085774e-05,
-1.195199916925636634e-03,
1.635086684393607817e-04,
2.421998614419040852e-02,
3.223361893294085403e-04,
-8.203248270750422566e-02,
-8.084007370863466010e-05,
6.263128942995785538e-05,
2.725918052815898707e-03,
-4.603185065901418851e-04,
-2.762644818679613895e-02,
1.739748126659289610e-04,
1.532625911025197255e-04,
-1.154737019045739432e-04,
-3.106775424372253038e-03,
4.979029393062357018e-04,
1.050137304975954505e-02,
-1.741361044118884516e-04,
1.169593267778839570e-04,
1.769239963943276036e-03,
-1.871339704909624941e-04,
1.185550237753507126e-04,
-6.204406409189332304e-05,
-4.027037360064359426e-04,
-4.477508534671101906e-05,
1.351656642878706734e-05,
7.235344547307824392e-06,
7.000000000000000000e+00,
},
{
-2.965586266976006818e-06,
1.952627772507389782e-06,
2.094870305466279257e-04,
-1.449787646609910677e-05,
-7.075799705257004510e-03,
-3.801809455287087506e-04,
7.192234749173789798e-02,
-3.080218845501052717e-17,
2.365938253997817089e-05,
-1.753981066485607070e-05,
-1.195000753196914008e-03,
1.635033369696323831e-04,
2.421713487717944494e-02,
3.222808170800939439e-04,
-8.202683380862123286e-02,
-8.082186144054653127e-05,
6.262178866743818237e-05,
2.725438922989650153e-03,
-4.602678403247951978e-04,
-2.762294545342096341e-02,
1.739440408348654507e-04,
1.532266651841440806e-04,
-1.154528564907482299e-04,
-3.106201203685396515e-03,
4.978271034794925012e-04,
1.049994679705920295e-02,
-1.740937229086650330e-04,
1.169353884931472430e-04,
1.768897094342192222e-03,
-1.871000043903427656e-04,
1.185251267974972737e-04,
-6.203006118459073338e-05,
-4.026221308449653914e-04,
-4.476340964717099376e-05,
1.351326173507294528e-05,
7.233397518483132091e-06,
7.000000000000000000e+00,
},
{
-2.964973269663110231e-06,
1.952499852396167725e-06,
2.094540490180057772e-04,
-1.450111049583289217e-05,
-7.075031097237696916e-03,
-3.801220230185934777e-04,
7.191804584948449652e-02,
2.604812119661625146e-16,
2.365427139066049472e-05,
-1.753778777016701577e-05,
-1.194801641248758413e-03,
1.634980027081381695e-04,
2.421428422611010545e-02,
3.222254604394242036e-04,
-8.202118592030682720e-02,
-8.080365499346485328e-05,
6.261228985809538740e-05,
2.724959924267146647e-03,
-4.602171823132487146e-04,
-2.761944352866131311e-02,
1.739132778908502913e-04,
1.531907511991247299e-04,
-1.154320165407938154e-04,
-3.105627147628734885e-03,
4.977512867544988881e-04,
1.049852089354409886e-02,
-1.740513559945881014e-04,
1.169114574110554223e-04,
1.768554327339732666e-03,
-1.870660483053408210e-04,
1.184952404572086520e-04,
-6.201606290883167837e-05,
-4.025405510824241651e-04,
-4.475173823099318725e-05,
1.350995821599711445e-05,
7.231451224393628299e-06,
7.000000000000000000e+00,
}
};
static double array_cof_err_sf_bessel_Ynu_0[9][37] = {
{
-1.201992669907915468e-22,
-1.363076888639510590e-22,
1.077853591113456505e-20,
5.668813045870544887e-22,
2.053979583409047492e-19,
-5.050098424172104190e-21,
-5.398863829045047343e-19,
-2.324080270438153164e-33,
-1.443071176832425409e-21,
4.195874029126487438e-22,
-1.010926751021503506e-19,
-9.757942766188336137e-21,
9.155536213346366468e-19,
-2.279325951530116006e-20,
-1.650253640261439528e-18,
-5.078542620837874608e-21,
-1.248139864796533426e-21,
-6.150891655008043302e-20,
5.266486754182009199e-21,
-7.064423665046942900e-19,
8.951494837626444676e-21,
1.779457710787268965e-21,
6.676691720224518511e-21,
1.590698492634892409e-19,
4.415179793460859887e-20,
-7.434153486886712077e-19,
9.935871523717713100e-21,
-3.496634885236306490e-21,
-8.035268571145607083e-20,
9.969559911150232869e-21,
4.268723333144562080e-21,
3.873664814896387283e-21,
2.081614951452674891e-20,
4.364419298319961480e-22,
5.864965227911976416e-22,
2.115003579925461353e-22,
},
{
-1.380594175897454178e-22,
1.565991761348335851e-22,
3.006557615689264583e-21,
-6.057571122687311272e-23,
-3.482347815879539254e-19,
-2.288214505635766808e-20,
-1.910768572665991081e-18,
-2.040504842828751887e-32,
1.274843238488928723e-21,
9.586615213718236916e-22,
-6.363815002353406642e-20,
7.733826174292876729e-21,
8.455660715260679541e-19,
2.563915807778312242e-20,
-6.731746086146931972e-18,
-4.987779882139304777e-21,
-6.071880121924054055e-21,
-9.259054281460097992e-20,
1.910832910941631889e-21,
3.382664334731964618e-19,
-9.343921990664862061e-21,
-2.774643204587796849e-21,
-3.384388741081954535e-21,
-1.387804089949905322e-19,
1.459039938272875748e-20,
6.103496213568462844e-19,
-7.739955184691786694e-21,
4.305456746614572062e-21,
3.379149121370372746e-20,
-5.176445455440727305e-21,
-2.993808602019831525e-21,
1.603764443139099950e-21,
1.762436707838937934e-20,
7.803507266599554113e-22,
1.922273633833628256e-22,
-4.005450343084242442e-22,
},
{
-8.622302520256918531e-23,
1.640384520666666283e-23,
8.258637019282547698e-22,
7.625873105735160133e-22,
5.336578811953395989e-20,
3.630641583967342575e-22,
-5.680281361583318122e-18,
1.492844485807373614e-32,
-3.743419337487497507e-23,
1.308849102134745375e-21,
-1.776664933987259795e-20,
-1.273353788154460439e-20,
-6.375416787953466904e-20,
-2.045830301980935124e-20,
6.881777166603692565e-18,
-2.087865318828302943e-21,
2.808925705860614324e-21,
1.538531928954177367e-19,
2.215677981331676800e-21,
-1.657971500257199477e-18,
7.959781782482675543e-21,
-1.176306402391617575e-20,
6.011934773962221559e-21,
1.200546440300759542e-19,
-4.155759703066681081e-20,
-5.139995647863369586e-19,
1.170253278569264129e-20,
4.403314965432064339e-21,
1.220102071495840048e-20,
8.848433766563709701e-21,
3.447729538369078180e-21,
-6.737875207584685124e-21,
-1.337419311135616546e-20,
9.331074833390874567e-22,
4.093267147215247125e-22,
-2.314592773199151695e-22,
},
{
-1.996111015461878705e-22,
-8.398029924551463901e-23,
7.305947587922640594e-21,
-4.550025193884717462e-22,
-3.941947845200955165e-19,
-2.250559766669731566e-20,
-1.734875303696472656e-18,
-1.758385252239234535e-32,
-9.982476398613544153e-23,
-1.152844640101652345e-21,
9.213254198177920966e-20,
-1.302690640208351645e-20,
-1.320022241679996147e-18,
2.533583977940363888e-20,
-4.896479498159487452e-18,
1.499542990911965335e-21,
6.598719155855379493e-21,
4.918703251376041432e-20,
-7.108194719905858691e-21,
9.251903734347156286e-19,
1.318295561838721798e-20,
2.784308559168234729e-21,
2.075614076015393105e-21,
1.693980763265834595e-19,
4.877924683888584734e-20,
2.028783025671315447e-19,
4.961477262287930329e-21,
6.757561925816728215e-21,
5.448647126699265753e-20,
4.864893386723925387e-21,
5.740885672237527401e-21,
4.004000197946061683e-21,
2.003068595057241226e-21,
2.852651002365577207e-21,
-6.670674130513436291e-22,
-1.335988447117469885e-22,
},
{
-1.698804453235958625e-22,
5.351195508134858305e-23,
1.826378255857450713e-21,
2.910428654661632824e-22,
4.149459560450296269e-19,
-4.133949152820344934e-21,
4.437248847112630308e-18,
-1.055638838410515874e-32,
5.625730161339277127e-22,
7.563507531565987394e-22,
-1.010924251400256063e-19,
-3.298349123374745368e-21,
-3.366860368984555862e-19,
2.028731341162719770e-20,
-4.229299726577168993e-18,
-6.534126711718602668e-21,
3.266928035072850937e-21,
-1.875657394186185147e-19,
-1.919023914689371908e-20,
2.164054618941775733e-20,
-2.594791675443708080e-21,
-1.227889403764521162e-20,
5.044505037215964538e-21,
1.457390982941588109e-19,
2.219199545729328287e-20,
1.172432513473302456e-20,
7.687117172223548989e-21,
-6.666435624700965512e-21,
3.497395422444809116e-20,
1.214661008210740171e-20,
-3.199665594832362590e-21,
-5.859965514107772377e-23,
-2.894861265630929327e-23,
-2.895765069328811445e-21,
-7.759424601482639950e-23,
2.391475635234835363e-22,
},
{
6.429245468093776441e-23,
1.500176412378359447e-22,
9.218114240043368794e-22,
5.355192195954638236e-22,
1.763481963673446687e-19,
-6.999846043463661576e-21,
1.257893123398508315e-18,
-1.194856256114285370e-33,
4.404362339037733959e-22,
-1.139287297992601575e-21,
1.022101660456884361e-20,
1.111207384853318024e-20,
-1.592011454966582400e-18,
-1.544325835970229576e-21,
-6.537166452555826262e-18,
-5.560691722055757487e-21,
-5.253725706312064471e-21,
1.912992033837522740e-19,
2.283318339754011570e-20,
-1.578046821897161828e-18,
4.923270819042649706e-22,
1.211282413104902299e-20,
5.890245638269090889e-22,
1.207384123882661075e-19,
-6.999267484279269828e-21,
-8.347057019534746147e-21,
-1.150807773922989195e-21,
-2.394495358285424605e-21,
3.360752647195939251e-20,
-2.486641582527846212e-21,
1.107991546350640056e-21,
-2.791000558202713268e-21,
2.326697427595110392e-20,
3.275288704554344274e-21,
-2.703544854189279136e-22,
-3.205657861513758414e-22,
},
{
-1.635367607231155401e-22,
9.188670940598750576e-23,
1.020140350071964476e-20,
4.563747396544364820e-22,
1.931551544953543875e-19,
-1.995590254653208935e-20,
6.735298707398817251e-18,
-2.703757316914360948e-33,
-1.467294148741794505e-21,
1.225448229087522328e-21,
-3.040441147359952709e-20,
3.797345771470874738e-21,
-1.163698892093463136e-18,
-7.111164530506273369e-21,
-2.104706166025799096e-18,
5.210538216241161734e-21,
5.752112375605758810e-21,
-1.494253195421943960e-19,
-2.715278938364462608e-22,
-7.913562270331169547e-19,
-2.287878195777610909e-22,
1.197416993262809361e-20,
-5.081402027909139012e-21,
2.017339054739693648e-19,
4.780535454645740169e-20,
-8.396331438755073963e-19,
-5.127451604249191162e-21,
-6.519636393470969523e-21,
9.798670536042768767e-20,
4.877333163597773652e-23,
4.710560714473037505e-21,
-3.803691529303376689e-21,
-1.027281378685686153e-20,
3.227072355670319699e-22,
-6.986460816928229276e-22,
2.640428787112377519e-22,
},
{
-7.408197171115276306e-23,
5.750716795209106166e-23,
-3.888581357587501491e-22,
-6.722360017183941923e-22,
-2.145501959068351774e-19,
1.227907688366906548e-21,
-2.396115630814614560e-18,
7.043296474637560664e-34,
1.613174234280989922e-22,
-1.322284974578056329e-21,
1.182884783283863170e-20,
-6.863783278506635392e-21,
1.164599882089966938e-18,
-1.985218475086234478e-20,
-2.815228331805025642e-18,
5.791529099033501684e-21,
-1.277335379034456716e-21,
-1.778279583766179321e-19,
-1.859068019006707727e-20,
-6.181075662636364803e-20,
1.211062729386575920e-21,
-1.038271277741920259e-20,
4.538855404737619409e-21,
1.302034930583176557e-19,
-1.933474619875181820e-20,
-1.647648146666105401e-19,
8.433935975911652223e-21,
-4.768258371199676184e-21,
1.784416285901060738e-20,
9.239691548552918826e-21,
-4.874951116027659716e-21,
4.150752148956111742e-21,
2.037793020505871540e-20,
2.566506461202562049e-21,
-1.719443358812503834e-22,
-1.456788185561375011e-22,
},
{
-1.967686659166566132e-22,
-1.470362498318471372e-22,
7.042183909988985802e-21,
1.137913355305654784e-22,
-2.822146896338050799e-20,
1.556522920901418160e-20,
6.829956309427812508e-18,
4.495123469451542274e-33,
9.989606341892236417e-22,
2.230895375205426428e-22,
-8.812763655777291030e-20,
-5.490484447658853745e-21,
-3.887475546952678331e-19,
-1.636830876614914106e-20,
3.155658219281527219e-18,
6.539723052761306439e-21,
1.639234057634641842e-21,
-1.263735626862212097e-19,
1.293260316426650080e-20,
-5.559490507388746816e-19,
-5.816371699420511776e-21,
-1.048987955643690534e-20,
3.112945871460354229e-21,
8.262002311929951222e-20,
1.836122763353893474e-20,
7.675235859754176984e-19,
-1.208465147920693864e-20,
1.387074608009989192e-21,
6.614801848747485214e-20,
8.497563025853031255e-21,
-5.359941891045294703e-21,
-6.609382469874840989e-21,
2.498737365970389285e-20,
-3.055186157102397539e-21,
-7.569682118737447184e-22,
1.389386211388203400e-22,
}
};
static double array_pointx_sf_bessel_Ynu_0[9] = {
6.047395720720152212e+01,
6.048185212648775178e+01,
6.048974704577396722e+01,
6.049764196506019687e+01,
6.050553688434641941e+01,
6.051343180363264196e+01,
6.052132672291887161e+01,
6.052922164220508705e+01,
6.053711656149131670e+01,
};
static double array_pointy_sf_bessel_Ynu_0[9] = {
9.452659615770723178e+01,
9.453560077666229233e+01,
9.454460531407440271e+01,
9.455360976995892486e+01,
9.456261414433112122e+01,
9.457161843720633954e+01,
9.458062264859989909e+01,
9.458962677852706236e+01,
9.459863082700317705e+01,
};
static double array_cofidx_sf_bessel_Ynu_0[9] = {
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_Ynu_0(double x,double y)
{
 int len_glob = 9;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_bessel_Ynu_0[idx])&&(x<=array_idx_sf_bessel_Ynu_0[idx+1])){
         double pointx = array_pointx_sf_bessel_Ynu_0[idx];
         double pointy = array_pointy_sf_bessel_Ynu_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_Ynu_0[idx];
         eft_tay2v(array_cof_float_sf_bessel_Ynu_0[idx],array_cof_err_sf_bessel_Ynu_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(x<array_idx_sf_bessel_Ynu_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_bessel_Ynu_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_Ynu(double x,double y)
{
if((x<=60.541064021134424)&&(y<=94.63366841790715)){
 return accuracy_improve_patch_of_gsl_sf_bessel_Ynu_0(x,y);
}
}