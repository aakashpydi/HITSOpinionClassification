package com.company;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedList;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class Main {
    private static final String TRAINING_DATA_DIRECTORY = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\train_data\\train_files";        //directory of the training data files
    private static final String TEST_DATA_DIRECTORY = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\test_data\\test_files";              //directory of the test data files

    private static final String TRAIN_VECTORS_DIRECTORY = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\train_data\\train_vectors";      //directory of the train vector files
    private static final String TEST_VECTORS_DIRECTORY = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\test_data\\test_vectors";         //directory of the test vector files

    private static final String POSITIVE_WORDS_FILE = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\pos_polar.txt";                      //file with list of positive polar words
    private static final String NEGATIVE_WORDS_FILE = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\neg_polar.txt";                      //file with list of negative polar words

    private static final String TRAIN_DATA_ARFF_FILE = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\train_data.arff";                   //train data's ARFF file used by Weka classifier
    private static final String TEST_DATA_ARFF_FILE = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\test_data.arff";                     //test data's ARFF file used by Weka classifier

    public static LinkedList<Document> _trainingDocuments = new LinkedList<Document>();         //represents all the training files or documents
    public static LinkedList<Document> _testDocuments = new LinkedList<Document>();             //represents all the test files or documents

    public static HashSet<String> _positiveWordSet = new HashSet<String>();                     //the set of positive polar words
    public static HashSet<String> _negativeWordSet = new HashSet<String>();                     //the set of negative polar words

    /**This function initializes the positive word set
     */
    protected static void initializePositiveWordSet(){
        try{
            BufferedReader br = new BufferedReader(new FileReader(POSITIVE_WORDS_FILE));
            String _wordRead = null;
            while((_wordRead = br.readLine()) != null){
                _positiveWordSet.add(_wordRead.toLowerCase());  //observe that the inserted string is converted to lower case. done to ignore case during comparisons
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    /**This function initializes the negative word set
     */
    protected static void initializeNegativeWordSet(){
        try{
            BufferedReader br = new BufferedReader(new FileReader(NEGATIVE_WORDS_FILE));
            String _wordRead = null;
            while((_wordRead = br.readLine()) != null){
                _negativeWordSet.add(_wordRead.toLowerCase());  //observe that the inserted string is converted to lower case. done to ignore case during comparisons
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    /**Getter method for _positveWordSet
     * @return the positive polar word set
     */
    protected static HashSet<String> getPositiveWordSet(){
        return _positiveWordSet;
    }

    /**Getter method for _negativeWordSet
     * @return the negative polar word set
     */
    protected static HashSet<String> getNegativeWordSet(){
        return _negativeWordSet;
    }

    /** This method initializes the sentence vectors associated with each sentence in the files in the train data set
     */
    private static void initializeTrainVectors(){
        File training_vectors = new File(TRAIN_VECTORS_DIRECTORY);
        for(final File fileEntry : training_vectors.listFiles()){
            int fileId = getNumberInFileName(fileEntry.getName());  //helper method that returns the file_id in the file name. file_id is unique for each data set
            //System.out.println("Train: "+fileEntry.getName() + " + " + getDocumentUsingFileID(fileId, _trainingDocuments)._fileName);
            getDocumentUsingFileID(fileId, _trainingDocuments).initializeDocVector(fileEntry.getAbsolutePath());
            //getDocumentUsingFileID(fileId, _trainingDocuments).printSentencesVectors();
        }
    }

    /**This method initializes the sentence vectors associated with each sentence in the files in the test data set
     */
    private static void initializeTestVectors(){
        File test_vectors = new File(TEST_VECTORS_DIRECTORY);
        for(final File fileEntry : test_vectors.listFiles()){
            int fileId = getNumberInFileName(fileEntry.getName());
            //System.out.println("Test: "+fileEntry.getName() + " + " + getDocumentUsingFileID(fileId, _trainingDocuments)._fileName);
            getDocumentUsingFileID(fileId, _testDocuments).initializeDocVector(fileEntry.getAbsolutePath());
            //getDocumentUsingFileID(fileId, _trainingDocuments).printSentencesVectors();
        }
    }

    /**This method initializes the training documents list which represents the documents in the train data set
     */
    protected static void initializeTrainingData(){
        File trainingFiles = new File(TRAINING_DATA_DIRECTORY);
        //File trainingFiles = new File(ASSERTIONS_DIRECTORY);
        for(final File fileEntry : trainingFiles.listFiles()){
            //System.out.println("Printing Files: "+fileEntry.getName());
            Document toAdd = new Document(fileEntry.getName(), fileEntry.getAbsolutePath(), false);
            _trainingDocuments.add(toAdd);
        }
    }

    /**This method initializes the test document list which represents the documents in the test data set
     */
    protected static void initializeTestData(){
        File testFiles = new File(TEST_DATA_DIRECTORY);
        for(final File fileEntry : testFiles.listFiles()){
            //System.out.println("Printing Files: "+fileEntry.getName());
            Document toAdd = new Document(fileEntry.getName(), fileEntry.getAbsolutePath(), true);
            _testDocuments.add(toAdd);
        }
    }

    /** This method creates an ARFF file representation of the data in the argument documents list
     * IMP: Each data entry in the ARFF file is stored in the ORDER CORRESPONDING TO THE ORDER IN WHICH
     * this._documents and this._documents are stored in the lists.
     *
     * Note: The ARFF file is created for using the Weka Naive Bayes classifier
     *
     * @param ARFF_PATH the path to the ARFF file to be created
     * @param _documents  the document list that needs to be represented as an ARFF file
     */
    private static void createARFF(String ARFF_PATH, LinkedList<Document> _documents){
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(ARFF_PATH));
            bw.write("@RELATION opinion_evaluator\n\n");
            bw.write("@ATTRIBUTE pos_polar_count INTEGER\n");
            bw.write("@ATTRIBUTE neg_polar_count INTEGER\n");
            bw.write("@ATTRIBUTE root_polarity {-1,0,1}\n");
            bw.write("@ATTRIBUTE advMod {0,1}\n");
            bw.write("@ATTRIBUTE aComp {0,1}\n");
            bw.write("@ATTRIBUTE xComp {0,1}\n");
            bw.write("@ATTRIBUTE class {F, O}\n\n");

            bw.write("@DATA\n");
            for(Document d : _documents){
                for(Sentence s : d._sentences){
                    bw.write(s._positive_polar_count+","+s._negative_polar_count+","+s._root_polarity+","+s._advMod+","+s._aComp+","+s._xComp+","+s._classifiedAs+"\n");
                }
            }
            bw.flush();
            bw.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    /**The method builds and runs the naive bayes classifier. The naive bayes classifier
     * is used to initialize the relevant values in each Sentence object
     */
    private static void runNaiveBayesClasifier(){
        try{
            //the following represents the train data set
            BufferedReader br = new BufferedReader(new FileReader(TRAIN_DATA_ARFF_FILE));
            Instances training_data = new Instances(br);
            br.close();
            training_data.setClassIndex(training_data.numAttributes() - 1);                 //class index is the last attribute

            //the following represents the test data set
            BufferedReader br_2 = new BufferedReader(new FileReader(TEST_DATA_ARFF_FILE));
            Instances testing_data = new Instances(br_2);
            br_2.close();
            testing_data.setClassIndex(testing_data.numAttributes()-1);                     //class index is the last attribute

            //Build the NaiveBayes classifier using the TRAIN DATA SET
            NaiveBayes nbClassifier = new NaiveBayes();
            nbClassifier.buildClassifier(training_data);

            //EVALUATE the naive bayes classifier built using the same TRAIN DATA SET
            Evaluation eval = new Evaluation(training_data);
            eval.evaluateModel(nbClassifier, training_data);
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println("Printing Confusion Matrix");
            double[][] confusionMatrix = eval.confusionMatrix();
            for(int i = 0; i < confusionMatrix.length; i++){
                for(int j = 0; j < confusionMatrix[i].length; j++){
                    System.out.print(confusionMatrix[i][j] + "\t");
                }
                System.out.println();
            }

            //INITIALIZE the naive bayes probabilities in the train data set
            // for each class in each sentence in every document using the classifier built above
            int index = 0;
            for(Document d : _trainingDocuments){
                //System.out.println("Document Name: " + d._fileName);
                for(Sentence s : d._sentences){
                    //System.out.println("Sentence ID: "+ s._sent_id);
                    //for(String word : s._words){
                    //  System.out.print(word + " ");
                    //}

                    double[] prediction = nbClassifier.distributionForInstance(training_data.get(index));   //remember from createARFF that corresponding Instance objects are in the order
                                                                                                            //documents and constituent sentences are in
                    for(int i = 0; i < prediction.length; i++){
                        if(i == 0){
                            s.setFactProbability(prediction[i]);
                        }
                        else if(i == 1){
                            s.setOpinionProbability(prediction[i]);
                        }
                        else{
                            System.out.println("ERROR using naive bayes classifier.");
                        }
                        //System.out.println("\nProbability of class "+ data.classAttribute().value(i)+" : "+Double.toString(prediction[i]));
                    }
                    //System.out.println(s._isFactProbability + "  :  "+s._isOpinionProbability);
                    index++;
                    //System.out.println("\n");
                }
            }

            //INITIALIZE the naive bayes probabilities on the test data set
            //for each class in each sentence in every document using the classifier built above
            index = 0;
            for(Document d: _testDocuments){
                for(Sentence s : d._sentences){
                    double[] prediction = nbClassifier.distributionForInstance(testing_data.get(index));
                    for(int i = 0; i < prediction.length; i++){
                        if(i == 0){
                            s.setFactProbability(prediction[i]);
                        }
                        else if(i == 1){
                            s.setOpinionProbability(prediction[i]);
                        }
                        else{
                            System.out.println("ERROR using naive bayes classifier");
                        }
                    }
                    index++;
                }
            }
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    /**The method runs the HITS alogorithm on each of the documents in the argument list
     *
     * @param documents the document list on which to run the HITS algorithm
     * @param iterations    the number of iterations to run the HITS algorithm for
     */
    private static void runHITSAlgorithm(LinkedList<Document> documents, int iterations){
        for(Document d: documents){                     //iterate through all the documents in the document list
            //System.out.println(d._fileName);
            d.initializeSentenceEdges();                //initialize the sentence edges
            for(int i=0; i < iterations; i++){          //update the sentence edges for the number of times given by iterations
                d.updateSentenceEdgesMSE();
            }

            //Code used for evaluation and reporting results
            //d.printSentenceEdges();
            //for(Sentence s : d._sentences){
            //    s.printSentenceInfo();
            //}
            System.out.println("\n\n------------"+d._fileName+"------------");
            //d.calculatePrecisionAt5Comparison();                              //Used on train data to evaluate performance
            //d.calculatePrecisionAt3Comparison();                              //Used on train data to evaluate performance
            d.printSentencesByHubScoresDesc();
            d.printSentencesByOpinionDesc();
            d.printTopHubScoreSentence();
            System.out.println("-------------------\n\n");
            //d.printSentenceEdges();
            //System.out.println("\n");
        }
        //Document.printTrainPrecisionAtFiveSummaryPerformance();               //Used on train data to evaluate performance
        //Document.printTrainPrecisionAtThreeSummaryPerformance();              //Used on train data to evaluate performance
    }

    /**Print the file names, the corresponding positive polar count and negative polar count
     * in each of the files in the argument list
     * @param toPrint
     */
    private static void printDocuments(LinkedList<Document> toPrint){
        System.out.println("--- PRINTING DOCUMENTS ---");
        int i = 1;
        for( Document fileEntry : toPrint){
            System.out.println(i + ". " + fileEntry._fileName + "\tPosPolarCount: "+fileEntry._docPosPolarCount+",\tNegPolarCount: "+fileEntry._docNegPolarCount);
            i++;
        }
        System.out.println("--- FINISHED PRINTING DOCUMENTS ---\n");
    }

    /**Print the sentences in descending order of the probability of being an opinion  for each of the files
     *
     * @param toPrint the list of files
     */
    private static void printOpinionatedSentences(LinkedList<Document> toPrint){
        for(Document file : toPrint){
            System.out.println("---PRINTING FOR "+file._fileName+"---");
            file.printSentencesByOpinionDesc();
            System.out.println("---Finished Printing---");
        }
    }

    /**Helper method that returns a DEEP copy of the list passed as an argument
     * @param toCopy    the list to copy
     * @return  DEEP copy of argument list
     */
    private static LinkedList<Document> deepCopyDocs(LinkedList<Document> toCopy){
        LinkedList<Document> duplicate = new LinkedList<Document>();
        for(Document toAdd: toCopy){
            duplicate.add(toAdd.deepCopy());
        }
        return duplicate;
    }

    /**Returns the file_id in the file name
     * Example files: "train_0.tsv" has fileID = 0, "train_1" has fileID =1
     * @param fileName
     * @return  the file ID
     */
    public static int getNumberInFileName(String fileName){
        String result = "";
        for(int i=0; i < fileName.length();i++){
            if(fileName.charAt(i) >= '0' && fileName.charAt(i) <= '9'){
                result += fileName.charAt(i);
            }
        }
        return Integer.parseInt(result);
    }

    /**Helper method that returns the requested Document from the Document list
     * @param fileID
     * @param toGetFrom
     * @return
     */
    private static Document getDocumentUsingFileID(int fileID, LinkedList<Document> toGetFrom){
        for(Document toCheck : toGetFrom){
            if(toCheck._fileId == fileID){
                return toCheck;
            }
        }
        return null;
    }

    public static void main(String[] args) {
        initializePositiveWordSet();
        initializeNegativeWordSet();

        initializeTrainingData();
        initializeTestData();

        //LinkedList<Document> _trainingDocsDuplicate = deepCopyDocs(_trainingDocuments);
        //printDocuments(_trainingDocuments);
        //Collections.sort(_trainingDocsDuplicate, new SortDocumentDescByPosPolarCountDesc());
        //printDocuments(_trainingDocsDuplicate);
        //Collections.sort(_trainingDocsDuplicate, new SortDocumentDescByNegPolarCountDesc());
        //printDocuments(_trainingDocsDuplicate);

        //LinkedList<Document> _testDocsDuplicate = deepCopyDocs(_testDocuments);
        //Collections.sort(_testDocsDuplicate, new SortDocumentDescByPosPolarCountDesc());
        //printDocuments(_testDocsDuplicate);
        //Collections.sort(_testDocsDuplicate, new SortDocumentDescByNegPolarCountDesc());
        //printDocuments(_testDocsDuplicate);

        //data is stored the order in which they are present in the corresponding lists
        createARFF(TRAIN_DATA_ARFF_FILE, _trainingDocuments);
        createARFF(TEST_DATA_ARFF_FILE, _testDocuments);
        runNaiveBayesClasifier();

        initializeTrainVectors();
        initializeTestVectors();

        //printOpinionatedSentences(_testDocuments);
        //Document toCheck = getDocumentUsingFileID(5, _testDocuments);
        // Sentence s = toCheck.getSentenceFromID(57);
        //s.printSentenceInfo();
        //runHITSAlgorithm(_trainingDocuments, 10000);
        runHITSAlgorithm(_testDocuments, 10000);
    }
}

/**Class used to sort the document in descending order based on total positive polar word counts
 */
class SortDocumentDescByPosPolarCountDesc implements Comparator<Document> {
    @Override
    public int compare(Document o1, Document o2) {
        if(o1._docPosPolarCount < o2._docPosPolarCount){
            return 1;
        }
        else if(o1._docPosPolarCount > o2._docPosPolarCount){
            return -1;
        }
        else{
            return 0;
        }
    }
}

/**Class used to sort the document in descending order based on total negative polar word counts
 */
class SortDocumentDescByNegPolarCountDesc implements Comparator<Document> {
    @Override
    public int compare(Document o1, Document o2) {
        if(o1._docNegPolarCount < o2._docNegPolarCount){
            return 1;
        }
        else if(o1._docNegPolarCount > o2._docNegPolarCount){
            return -1;
        }
        else{
            return 0;
        }
    }
}