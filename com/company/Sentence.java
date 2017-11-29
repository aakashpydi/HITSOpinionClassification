package com.company;

import java.util.LinkedList;

/**
 * This class represents an individual sentence in a given file.
 */
class Sentence{
    private static final int VECTOR_ELEMENTS_COUNT = 300;   //number of elements in the sentence vector. vectors stored in files <file_name>_vectors.tsv

    LinkedList<String> _words;  //the words in the Sentence are stored as a LinkedList of strings
    String _rootWord;           //the "root word" of the sentence
    char _classifiedAs;         // Indicates what the sentence is Classified as. 'O' --> Opinion. 'F' --> Fact. '?' --> Unknown

    int _sent_id;               //the sentence_id. this value is UNIQUE within each article/file
    int _positive_polar_count;  //the number of positive polar words found in the sentence. The value is an integer greater than or equal to 0. The total positive polar word set is found in --> pos_polar.txt
    int _negative_polar_count;  //the number of negative polar words found in the sentence. The value is an integer greater than or equal to 0. The total negative polar word set is found in --> neg_polar.txt
    int _root_polarity;         //the "polarity" of the root word. Can be {-1,0,1}. -1 --> indicates root word is a negative polar word. +1 --> indicates root word is a positive polar word. 0 --> neither negative or positive

    double[] _vectorArray;      //the sentence vector. retrieved from <file_name>_vectors.tsv

    //Based on Stanford typed dependencies representation. Link: https://nlp.stanford.edu/pubs/dependencies-coling08.pdf
    int _advMod;                // indicates if advMod is present. can be {0,1}
    int _aComp;                 // indicates if aComp is present. can be {0,1}
    int _xComp;                 // indicates if xComp is present. can be {0,1}

    double _isFactProbability;      //the probability that this sentence is a "Fact" (initialized by the Naive Bayes classifier built in Main)
    double _isOpinionProbability;   //the probability that this sentence is an "Opinion" (initialized by the Naive Bayes classifier built in Main)

    //used in implementing the ITERATIVE HyperLink Induced Topic Search (HITS) algorithm.
    double _hubScore;           //the hub score of this sentence.
    double _authorityScore;     //the authority score of this sentence.
    double _topSupportingAuthorityScore; //used to help identify the corresponding top supporting authorities for a sentence

    /**
     * Default constructor.
     */
    public Sentence(){
    }

    /**
     * A Constructor that takes a single line from a train/test file and initializes a representative Sentence object.
     *
     * @param _toExtractData a single line from a train/test file which represents a unique sentence in the file.
     *                       Format: <sentence_id> <classified_as> <sequence of sentence words> <root_word of sentence> <adv_mod_pres> <acomp_pres> <x_comp_pres>
     *                       Note: <classified_as> is missing in test file sentences.
     * @param isUnclassified    indicates if the sentence is classified as either opinion or fact.
     */
    public Sentence(String _toExtractData, boolean isUnclassified){
        String[] _sentenceTokens = _toExtractData.split("\\s|\\t"); //tokenizes the sentence with space or tab as the delimitter.
        _words = new LinkedList<String>();
        _vectorArray = new double[VECTOR_ELEMENTS_COUNT];
        _topSupportingAuthorityScore = 0;
        for(int i = 0; i < _sentenceTokens.length; i++){                //iterates through all the tokens in the argument sentence
            if(i == 0){                                                 //first token is the sentence ID
                this._sent_id = Integer.parseInt(_sentenceTokens[i]);
            }
            else if(i == 1){
                if(isUnclassified){                                     //IF this is an unclassified sentence- then second token is NOT the classified_as tag. In this case second tag will be
                                                                        //FIRST WORD in the sequence of words of the sentence
                    this._classifiedAs = "?".charAt(0);                 //setting classified as to ? which indicates the sentence is unclassified

                    //check if the word being added is either a positive polar word or negative polar word- and update corresponding count if so.
                    //all words compared in lower case (so no case sensitivity)
                    if(Main._positiveWordSet.contains(_sentenceTokens[i].toLowerCase())){   //we initialize _positiveWordSet in Main.
                        this._positive_polar_count++;
                    }
                    if(Main._negativeWordSet.contains(_sentenceTokens[i].toLowerCase())){   //we initialize _negativeWordSet in Main
                        this._negative_polar_count++;
                    }
                    _words.add(_sentenceTokens[i]);
                }
                else{
                    this._classifiedAs = _sentenceTokens[i].charAt(0);  //IF this is a classified sentence- then the second token IS the classified_as tag
                }
            }
            else if(i == _sentenceTokens.length - 4){                   //the FOURTH last token in the sentence is the root word by the format
                this._rootWord = _sentenceTokens[i];
                //set root polarity using the word sets initialized in main. observe there is no case sensitivity because all words compared in lower case.
                if(Main._positiveWordSet.contains(_sentenceTokens[i].toLowerCase())){
                    this._root_polarity = 1;
                }
                else if(Main._negativeWordSet.contains(_sentenceTokens[i].toLowerCase())){
                    this._root_polarity = -1;
                }
                else{
                    this._root_polarity = 0;
                }
            }
            else if(i == _sentenceTokens.length - 3){                   //the THIRD last token in the sentence is the advMod presence indicator as given by the format
                this._advMod = Integer.parseInt(_sentenceTokens[i]);
            }
            else if(i == _sentenceTokens.length - 2){                   //the SECOND last token in the sentence is the aComp presence indicator as given by the format
                this._aComp = Integer.parseInt(_sentenceTokens[i]);
            }
            else if(i == _sentenceTokens.length - 1){                   //the LAST token in the sentence is the xComp presence indicator as given by the format.
                this._xComp = Integer.parseInt(_sentenceTokens[i]);
            }
            else{                                                       //all remaining tokens are the sequence of words in the sentence
                if(Main._positiveWordSet.contains(_sentenceTokens[i].toLowerCase())){
                    this._positive_polar_count++;
                }
                if(Main._negativeWordSet.contains(_sentenceTokens[i].toLowerCase())){
                    this._negative_polar_count++;
                }
                _words.add(_sentenceTokens[i]);
            }
        }
    }

    /**Setter method for _isFactProbability
     *
     * @param prob
     */
    public void setFactProbability(double prob){
        this._isFactProbability = prob;
    }

    /**Setter method for _isOpinionProbability
     *
     * @param prob
     */
    public void setOpinionProbability(double prob){
        this._isOpinionProbability = prob;
    }

    /**Setter method for _vectorArray
     *
     * @param toSet
     */
    public void setVectorArray(double[] toSet){
        this._vectorArray = toSet;
    }

    /**Getter method for _vectorArray
     *
     * @return
     */
    public double[] getVectorArray(){
        return this._vectorArray;
    }


    /**Computes the cosine similarity between two Sentences
     *
     * @param _sent_2
     * @return
     */
    public double computeSentenceSimilarity(Sentence _sent_2){
        //find dot product of the two vectors (summation of multiplied values of corresponding elements)
        double numerator = 0;
        for(int i = 0; i < this._vectorArray.length; i++){  //both vectors will have the same length
            if(Double.isNaN(this._vectorArray[i]) || Double.isNaN(_sent_2._vectorArray[i])){    //ignore NaN values that might be in the vector
                continue;
            }
            else{
                numerator+= this._vectorArray[i]*_sent_2._vectorArray[i];
            }
        }


        double denominator = 0;
        //find the squared sum of each vector (summation of squared values of each element)
        double squaredSum_1 = 0;
        double squaredSum_2 = 0;
        for(int i = 0; i< this._vectorArray.length; i++){
            if(Double.isNaN(this._vectorArray[i])){     //ignore NaN values that might be in the vector
                continue;
            }
            else{
                squaredSum_1 += (this._vectorArray[i]*this._vectorArray[i]);
            }
        }
        for(int i = 0; i < this._vectorArray.length; i++){
            if(Double.isNaN(_sent_2._vectorArray[i])){      //ignore NaN values that might be in the vector
                continue;
            }
            else{
                squaredSum_2 += (_sent_2._vectorArray[i]*_sent_2._vectorArray[i]);
            }
        }
        denominator = Math.sqrt(squaredSum_1)*Math.sqrt(squaredSum_2);
        if(denominator == 0.0){ //if the denominator is zero we return a Cosine similarity of 1.0
                                //we assume we reach this case only when denominator gets infinitely small before the numerator
            return 1.0;
        }
        else{
            return numerator/denominator;
        }

    }

    /** Creates a NEW Sentence object that is a DEEP copy of the current object.
     *
     * @return The NEW Sentence object
     */
    public Sentence deepCopy(){
        Sentence toReturn = new Sentence();
        LinkedList<String> _wordsCopy = new LinkedList<String>();
        for(String word : this._words){
            _wordsCopy.add(word);
        }
        toReturn._words = _wordsCopy;
        toReturn._rootWord = this._rootWord;
        toReturn._classifiedAs = this._classifiedAs;
        toReturn._sent_id = this._sent_id;
        toReturn._positive_polar_count = this._positive_polar_count;
        toReturn._negative_polar_count = this._negative_polar_count;
        toReturn._root_polarity = this._root_polarity;
        toReturn._advMod = this._advMod;
        toReturn._aComp = this._aComp;
        toReturn._xComp = this._xComp;
        toReturn._isFactProbability = this._isFactProbability;
        toReturn._isOpinionProbability = this._isOpinionProbability;
        toReturn._authorityScore = this._authorityScore;
        toReturn._hubScore = this._hubScore;
        toReturn._vectorArray = new double[this._vectorArray.length];
        for(int i = 0; i < this._vectorArray.length; i++){
            toReturn._vectorArray[i] = this._vectorArray[i];
        }
        toReturn._topSupportingAuthorityScore = this._topSupportingAuthorityScore;
        return toReturn;
    }

    /** Prints sentence to STDOUT.
     */
    public void printSentence(){
        System.out.print("(Sent Id: "+this._sent_id + ") (Class:"+this._classifiedAs +"). (isOpinion:" + this._isOpinionProbability+") (Hub Score: "+this._hubScore+") (Authority Score: "+this._authorityScore+") ");
        for(String word: this._words){
            System.out.print(word + " ");
        }
        System.out.println();
    }

    /** Prints Sentence to STDOUT with supporting authority score included
     */
    public void printSupportingAuthoritySentence(){
        System.out.print( "(Sent Id: "+this._sent_id + "(Support Metric Score: "+this._topSupportingAuthorityScore+") ");
        for(String word: this._words){
            System.out.print(word + " ");
        }
        System.out.println();
    }

    /**Prints the Sentence Vector values to STDOUT
     */
    public void printSentenceVector(){
        System.out.print(this._sent_id+" [");
        for(int i = 0; i < this._vectorArray.length; i++){
            System.out.print(_vectorArray[i]+", ");
        }
        System.out.println("\n");
    }

    /**Prints sentence information to STDOUT
     */
    public void printSentenceInfo(){
        System.out.println("Sentence ID: " + this._sent_id);
        System.out.println("Hub Score: "+this._hubScore);
        System.out.println("Authority Score: "+this._authorityScore);
//        System.out.println("AdvMod - AComp - xComp: " + _advMod+" - "+_aComp+" - "+_xComp);
//        System.out.println("Root Word: "+ _rootWord);
//        System.out.println("Root Polarity: "+ _root_polarity);
//        System.out.println("Classified as: "+ _classifiedAs);
//        System.out.println("Positive Polar Words Count: " + this._positive_polar_count);
//        System.out.println("Negative Polar Words Count: " + this._negative_polar_count);
    }
}