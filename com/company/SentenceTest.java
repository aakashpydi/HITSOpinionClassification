package com.company;

import junit.framework.TestCase;

public class SentenceTest extends TestCase {

    public void testSentenceConstructor() throws Exception{
        Main.initializePositiveWordSet();
        Main.initializeNegativeWordSet();

        //Here we take two words from positive polar words and three words from negative
        // agreeable from positive polar words, irrationally and blah from negative
        String _sentence_1 = "1 O agreeable yum Agreeable buzzword irrationally bird blah IrratioNally\tIrRationally\t0\t1\t1";
        String[] _words ={"agreeable", "yum", "Agreeable", "buzzword", "irrationally" , "bird" ,"blah", "IrratioNally", "IrRationally"};
        Sentence test_1 = new Sentence(_sentence_1, false);
        assertEquals(1, test_1._sent_id);
        assertEquals("IrRationally",test_1._rootWord);
        assertEquals(2, test_1._positive_polar_count);
        assertEquals(3, test_1._negative_polar_count);
        assertEquals(-1, test_1._root_polarity);
        assertEquals(0, test_1._advMod);
        assertEquals(1, test_1._aComp);
        assertEquals(1, test_1._xComp);
        int i = 0;
        for( String word : test_1._words){
            assertEquals(_words[i], word);
            i++;
        }
        assertEquals(_words[i], test_1._rootWord);

        String _sentence_2 = "12 Agreeable buzzword IrRationally\tAgreeable\t1\t0\t0";
        String[] _words_2 ={"Agreeable", "buzzword", "IrRationally", "Agreeable"};
        Sentence test_2 = new Sentence(_sentence_2, true);
        assertEquals(12, test_2._sent_id);
        assertEquals("Agreeable",test_2._rootWord);
        assertEquals(1, test_2._positive_polar_count);
        assertEquals(1, test_2._negative_polar_count);
        assertEquals(1, test_2._root_polarity);
        assertEquals(1, test_2._advMod);
        assertEquals(0, test_2._aComp);
        assertEquals(0, test_2._xComp);
        i = 0;
        for( String word : test_2._words){
            assertEquals(_words_2[i], word);
            i++;
        }
        assertEquals(_words_2[i], test_2._rootWord);

        String _sentence_3 = "5\tWashington insiders agree\tagree\t0\t0\t0";
        String[] _words_3 ={"Washington", "insiders", "agree", "agree"};
        Sentence test_3 = new Sentence(_sentence_3, true);
        assertEquals(5, test_3._sent_id);
        assertEquals("agree",test_3._rootWord);
        assertEquals(0, test_3._positive_polar_count);
        assertEquals(0, test_3._negative_polar_count);
        assertEquals(0, test_3._root_polarity);
        assertEquals(0, test_3._advMod);
        assertEquals(0, test_3._aComp);
        assertEquals(0, test_3._xComp);
        i = 0;
        for( String word : test_3._words){
            assertEquals(_words_3[i], word);
            i++;
        }
        assertEquals(_words_3[i], test_3._rootWord);


        String _sentence_4 = "7\tO\tAt least they seem so\tseem\t1\t0\t0";
        String[] _words_4 ={"At", "least", "they", "seem", "so", "seem"};
        Sentence test_4 = new Sentence(_sentence_4, false);
        assertEquals(7, test_4._sent_id);
        assertEquals("seem",test_4._rootWord);
        assertEquals(0, test_4._positive_polar_count);
        assertEquals(1, test_4._negative_polar_count);
        assertEquals(0, test_4._root_polarity);
        assertEquals(1, test_4._advMod);
        assertEquals(0, test_4._aComp);
        assertEquals(0, test_4._xComp);
        i = 0;
        for( String word : test_4._words){
            assertEquals(_words_4[i], word);
            i++;
        }
        assertEquals(_words_4[i], test_4._rootWord);
    }

    public void testSetFactProbability() throws Exception {
        Sentence testMe = new Sentence();
        testMe.setFactProbability(0.9345);
        assertEquals(testMe._isFactProbability, 0.9345);
    }

    public void testSetOpinionProbability() throws Exception {
        Sentence testMe = new Sentence();
        testMe.setOpinionProbability(0.9345);
        assertEquals(testMe._isOpinionProbability, 0.9345);
    }

    public void testComputerSentenceSimilarity() throws Exception{
        //General Test Case. Correct values calculated on Excel
        Sentence s1 = new Sentence();
        double[] vec_s1 = {5.0,4.0,5.0,3.0};
        Sentence s2 = new Sentence();
        double[] vec_s2 = {3.0,3.0,3.0,3.0};
        s1.setVectorArray(vec_s1);
        s2.setVectorArray(vec_s2);
        assertEquals(0.9814954576223638, s1.computeSentenceSimilarity(s2));

        //Test Case Taken From train_vec_0. Correct values calculated on Excel
        Sentence s3 = new Sentence();
        double[] vec_s3 = {-0.039970368, 0.16962379, -0.081852213, 0.0089267464, 0.12107661, 0.056631435, 0.034089614, -0.0079668341, -0.062846191, 1.9471427, -0.13758463, 0.05754621, 0.13303871, -0.1358296, -0.042650145, -0.051025953, -0.011708091, 0.86480051, -0.11261898, -0.029016672, -0.015629834, -0.013374166, -0.07720121, -0.00042033274, 0.10387788, 0.087230779, -0.044567924, -0.017781457, -0.052615702, 0.066859215, -0.020671276, 0.17954044, -0.07980746, 0.11217767, 0.038643669, -0.04315567, -0.031615488, 0.12087154, -0.078436852, 0.0061752112, -0.051830828, 0.086882643, 0.018298337, -0.043780569, 0.047185659, 0.028399328, -0.088155687, 0.097243823, 0.037209209, 0.02555313, 0.028652804, 0.020953367, -0.023520092, 0.022682875, -0.083032995, 0.039412003, 0.016253037, -0.086545773, 0.02859127, -0.028692961, -0.0056559057, 0.020754665, -0.052671362, 0.20569269, -0.011901896, -0.033271413, 0.00045153499, -0.060455382, -0.13514642, 0.21606691, -0.01442108, -0.07670372, 0.071487576, 0.084147751, 0.019049337, 0.10895445, 0.051607996, -0.078876875, -0.043707281, 0.11424673, -0.050042067, 0.11216075, -0.098056376, -0.070885442, 0.08679343, -0.083653338, -0.10426459, -0.085485846, 0.053698692, 0.097590961, -0.10368491, 0.0017303746, 0.0085595818, -0.021334043, 0.061141569, -0.090199709, -0.10000399, -0.041710246, 0.081652336, 0.0062582511, 0.016231544, 0.018279433, -0.12352305, -0.05334742, 0.041160032, -0.24260752, 0.10696948, 0.12976955, 0.057955086, 0.030354619, 0.057309475, -0.16708361, -0.0048512924, -0.042731494, -0.078395031, 0.061174348, -0.0026586067, 0.042636212, -0.010119044, -0.0041272962, 0.056401044, -0.072948545, -0.069305792, -0.040175028, -0.086581171, 0.025382495, -0.010443131, 0.002564789, 0.022503084, 0.037567426, -0.053392123, -0.01434492, -0.081692845, -0.083928287, 0.033190172, -0.0046478598, 0.032071281, 0.092726164, 0.13517314, -0.083338164, -0.99708748, 0.085245423, 0.063769586, 0.11112013, 0.028398186, 0.013167222, -0.026345501, 0.079142295, 0.11427795, -0.02743875, -0.09662994, -0.063950345, 0.035084169, 0.023413332, -0.074226171, 0.056020927, -0.032532666, 0.023046166, -0.076868072, -0.071400039, 0.0046419832, 0.011059794, 0.051323041, -0.12494728, -0.056484457, -0.15570642, -0.03869129, -0.070671313, 0.083410628, 0.069965921, 0.01876138, -0.032928299, -0.047898993, -0.085263096, -0.13955911, -0.0020613365, -0.10459236, 0.044679213, 0.061278339, -0.1090822, 0.049556818, -0.13286002, -0.037428167, 0.028125444, -0.16262047, 0.021440247, 0.034753039, -0.0073211268, 0.06658712, -0.043440834, -0.025753142, 0.046377957, -0.011936863, 0.10307848, -0.091693215, 0.13195242, -0.041409735, -0.059321251, -0.11188794, 0.15417536, 0.070146695, -0.019463873, 0.043418694, 0.044122357, 0.14977084, -0.034588087, 0.029678432, -0.087189831, -0.1118603, 0.028235331, -0.019406041, 0.039153036, -0.062896796, -0.043658007, 0.023753932, 0.18549056, -0.068660527, -0.041300308, -0.11530712, 0.064031273, -0.042401876, -0.086391546, -0.13934737, 0.045938749, 0.0044896896, 0.014164098, 0.068337299, 0.13401569, 0.067113377, 0.062951855, -0.097962998, 0.11288231, 0.040845666, 0.054022163, -0.029898755, -0.017301379, -0.12485778, -0.034636136, -0.088532783, 0.027765876, -0.054600164, 0.020744925, 0.0055243731, 0.036074623, 0.011932873, -0.11551217, -0.18921512, -0.087408297, -0.069047458, 0.1502535, 0.0086100018, 0.03690435, -0.06604071, -0.11977027, 0.016736232, 0.14671566, -0.0070350468, -0.051838573, -0.023511538, 0.034861188, 0.10368826, 0.14529516, 0.024247749, 0.091669224, -0.0032584418, -0.019098017, -0.025482543, -0.026389794, 0.15376715, -0.041232828, 0.11868665, 0.097040482, -0.086327076, -0.14005955, -0.14698416, 0.024730666, 0.13264126, 0.012706158, -0.065924384, 0.17198829, 0.23957847, -0.019264368, -0.10227159, -0.10538291, -0.011053578, -0.12510101, 0.10981957, -0.073305108, 0.085451327, -0.064654917, -0.16533342, -0.014758751, -0.057830881, -0.11172125, -0.12522601, 0.10201671, 0.028307959, -0.045162499, 0.061551515, 0.00065895898};
        Sentence s4 = new Sentence();
        double[] vec_s4 = {0.011701468, 0.09989325, -0.10794373, -0.088366404, 0.0128185, -0.062435482, 0.015698515, -0.063916937, 0.0039201584, 2.1196001, -0.058201246, 0.064510971, 0.13169828, -0.052937448, -0.016843, -0.14797574, -0.067317039, 0.82675236, -0.16982144, -0.11385521, -0.057117596, -0.13173097, -0.043496583, 0.027228784, 0.055036765, 0.049687609, -0.078560859, -0.068291873, -0.053840533, -0.027205663, -0.044589341, 0.050950997, 0.0047109178, 0.03858497, 0.024383761, -0.079871923, -0.034259696, 0.020387676, -0.063031048, -0.11255565, 0.0098403161, 0.17224142, 0.02425975, -0.053328756, 0.088906541, 0.0020765641, -0.19218279, -0.019721035, -0.031376891, -0.00382478, -0.0062199803, 0.039289638, -0.049313426, 0.039378956, 0.003371858, 0.011376454, -0.023722271, -0.10214742, -0.021806719, -0.040695943, 0.037120346, 0.0088548101, -0.052590501, 0.16406927, -0.039926834, -0.034697082, 0.026918136, 0.021031111, 0.007978972, 0.050102886, 0.17353825, 0.1031913, 0.14152551, 0.0023746388, 0.11332248, -0.03880354, 0.051836852, -0.062155861, -0.031330865, 0.16677043, -0.048166003, 0.063115828, -0.16248032, -0.026059724, 0.076256663, -0.13914926, -0.15128429, -0.10425223, 0.13804556, 0.051065471, -0.086117029, 0.050893143, -0.057877406, 0.033473942, 0.14238922, -0.077519476, 0.077192783, -0.054974392, 0.03341205, 0.01848422, -0.0097942548, 0.059941068, -0.091541812, -0.10178569, 0.066456303, -0.6284889, 0.075689398, 0.042757072, 0.037894234, -0.082553178, 0.049525771, -0.15402286, 0.036649611, -0.047083393, 0.019924538, -0.034920827, -0.037942365, 0.14179218, 0.0142504, -0.0057407087, 0.037450023, -0.015394464, -0.078714028, 0.010389096, -0.06056061, 0.061575942, -0.041521482, -0.1396275, -0.020471133, -0.031597745, 0.028735671, 0.0077983621, -0.075744279, 0.036582246, 0.07488919, -7.0874892e-05, 0.021636801, 0.01877, 0.059068382, -0.024043407, -1.087485, 0.080339432, 0.11319686, -0.048171081, 0.049287263, -0.048765123, -0.078322507, -0.047475919, 0.14441581, -0.084693864, -0.069512546, -0.02878195, 0.099588417, -0.027180575, -0.16570179, -0.088428676, -0.12160189, 0.070693098, -0.039223727, -0.099758297, -0.10623913, -0.023711534, 0.012906338, -0.1641825, -0.10336348, -0.20022264, 0.039644875, 0.011417161, 0.16000408, 0.011989512, 0.046452358, 0.0087559791, 0.066687092, 0.029687775, -0.11948226, -0.0065705469, 0.0085841715, -0.03790592, -0.025243031, -0.097576216, -0.07941509, -0.068069279, -0.11081024, -0.06662485, -0.087477982, -0.036543105, -0.038185034, 0.043141779, 0.083277643, 0.02972419, 0.029876752, 0.058848996, -0.12549712, 0.10172872, 0.0066426746, 0.12104063, -0.054503746, -0.018363975, -0.14512306, 0.088361293, 0.061025679, -0.040700421, -0.04765353, 0.073966995, 0.15326093, 0.017034724, 0.036767531, 0.026083974, 0.030428192, 0.074026316, 0.0018018875, 0.0542094, -0.036553677, -0.14880963, -0.018386232, 0.23620027, -0.062948979, 0.0097317258, -0.1133187, -0.035314247, 0.054846413, -0.019271731, -0.093245588, 0.070304304, -0.039428256, -0.033216607, -0.020122109, 0.11824793, 0.05164161, 0.074316032, -0.085191138, 0.0588734, 0.051363271, 0.048820429, -0.069959864, -0.10826072, -0.014478177, -0.035696167, -0.09369114, 0.082751498, 0.037870962, 0.073584877, -0.011860215, 0.10319284, 0.048065357, -0.13487361, -0.070189387, -0.158852, -0.06147927, 0.045517579, -0.021793876, -0.041122891, 0.011262093, -0.037737854, 0.046782389, 0.1392975, 0.13700257, 0.0022017488, -0.026919393, 0.049757503, 0.19670761, 0.18801729, -0.067140423, 0.083575532, 0.09981101, 0.0079408167, 0.032245234, 0.036475468, 0.32392785, -0.049188554, 0.15219951, -0.031726696, -0.12627573, -0.099904768, -0.069811068, 0.0026830519, -0.0095500993, 0.11069986, 0.026700804, 0.22103511, 0.2494186, 0.099391192, -0.053014956, -0.067115702, -0.040385194, -0.04635229, 0.043701299, -0.042465601, 0.11060674, -0.029520294, -0.18733634, 0.031621117, 0.052448336, -0.075993821, 0.029528627, 0.041944142, -0.047952391, -0.082398757, 0.052425116, 0.0013526038};
        s3.setVectorArray(vec_s3);
        s4.setVectorArray(vec_s4);
        assertEquals(0.9077168453160825 , s3.computeSentenceSimilarity(s4));
    }

    public void testDeepCopy() throws Exception {
        Sentence makeCopyOfMe = new Sentence("1\tO\tHBO s dramedy Looking three men San Francisco road discovery love happiness themselves\tis\t1\t0\t0",false);
        String[] words = {"HBO","s", "dramedy","Looking", "three", "men", "San", "Francisco", "road", "discovery", "love", "happiness", "themselves","is"};
        Sentence deepCopy = makeCopyOfMe.deepCopy();
        assertNotSame(makeCopyOfMe, deepCopy);
        assertEquals(makeCopyOfMe._sent_id, deepCopy._sent_id);
        assertEquals(makeCopyOfMe._rootWord, deepCopy._rootWord);
        assertEquals(makeCopyOfMe._isFactProbability, deepCopy._isFactProbability);
        assertEquals(makeCopyOfMe._isOpinionProbability, deepCopy._isOpinionProbability);
        assertEquals(makeCopyOfMe._root_polarity, deepCopy._root_polarity);
        assertEquals(makeCopyOfMe._classifiedAs, deepCopy._classifiedAs);
        assertEquals(makeCopyOfMe._negative_polar_count, deepCopy._negative_polar_count);
        assertEquals(makeCopyOfMe._positive_polar_count, deepCopy._positive_polar_count);
        assertEquals(makeCopyOfMe._aComp, deepCopy._aComp);
        assertEquals(makeCopyOfMe._xComp, deepCopy._xComp);
        assertEquals(makeCopyOfMe._advMod, deepCopy._advMod);
        int index = 0;
        for(String _word : deepCopy._words){
            assertEquals(words[index], _word);
            index++;
        }
        assertEquals(words[index], deepCopy._rootWord);
    }

}