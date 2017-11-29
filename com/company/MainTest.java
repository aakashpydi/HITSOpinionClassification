package com.company;

import junit.framework.TestCase;

import java.util.HashSet;

public class MainTest extends TestCase {
    private static final int NUM_OF_WORDS_IN_POS_POLAR = 1176;  //we get this from looking at pos_polar.txt. Total words are 1185
                                                                // There are 9 duplicate words like agreeable, beneficial

    private static final int NUM_OF_WORDS_IN_NEG_POLAR = 1896; //Total words 1902. Duplicates 6.

    public void testInitializePositiveWordSet() throws Exception {
        Main.initializePositiveWordSet();
        HashSet<String> _positiveWordSet = Main.getPositiveWordSet();
        assertEquals(NUM_OF_WORDS_IN_POS_POLAR, _positiveWordSet.size());
    }

    public void testInitializeNegativeWordSet() throws Exception {
        Main.initializeNegativeWordSet();
        HashSet<String> _negativeWordSet = Main.getNegativeWordSet();
        assertEquals(NUM_OF_WORDS_IN_NEG_POLAR, _negativeWordSet.size());
    }

    public void testMainTrainFilesInitialization(){
        Main.initializeTrainingData();
        assertEquals(50, Main._trainingDocuments.size());
    }

    public void testMainTestFilesInitialization(){
        Main.initializeTestData();
        assertEquals(10, Main._testDocuments.size());
    }
}