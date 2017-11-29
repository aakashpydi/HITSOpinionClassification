package com.company;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;

/** This class represents each train/test document.
 *  Note: The following classes are declared after this class within this file: Edge, SortSentencesByOpinionDesc, SortSentencesByHubScores, SortSentencsByTopSupportingAuthorityScores
 */
public class Document{
    private static final double EPSILON_CONVERGENCE_FACTOR_MSE = 0.001;             //Used to terminate the ITERATIVE HUBS algorithms
    private static final int NUMBER_OF_TOP_SUPPORTING_AUTHORITIES_TO_PRINT = 3;     //the number of supporting authorities to print. Used when printing results of iterative HUBS
    private static final int NUMBER_OF_SENTENCES_TO_PRINT = 4;                      //indicates number of sentences to print by some of the member functions
    private static final double NUMBER_OF_DOCS_IN_TRAIN = 50.0;                     //the number of files in the train set

    //static helper variables used to calculate performance of classifiers on train data
    private static double _totalHubPrecisionAtFive = 0.0;
    private static double _totalNaivePrecisionAtFive = 0.0;
    private static double _totalHubPrecisionAtThree = 0.0;
    private static double _totalNaivePrecisionAtThree = 0.0;

    String _fileName;                   //the file name
    int _fileId;                        //the unique id of the file in this directory. Extracted from file name. Eg: test_1.tsv _fileId = 1
    LinkedList<Sentence> _sentences;    //represents the Sentences in this file
    LinkedList<Edge> _sentenceEdges;    //represents the directed edges between the sentences in this file. Used in ITERATIVE HUBS
    private boolean _hasConverged;      //indicates if the scores associated with iterative HUBS have converged


    int _docPosPolarCount = 0;          //the total number of positive polar words in this document
    int _docNegPolarCount = 0;          //the total number of negative polar words in this document

    /** Default constructor
     */
    public Document(){
    }

    /** Simple constructor that only initializes the file name and the file id
     * @param fileName the name of the file
     */
    public Document(String fileName) {
        this._fileName = fileName;
        this._fileId = Main.getNumberInFileName(fileName);
        this._hasConverged = false;
        this._sentences = null;
        this._sentenceEdges = null;
    }

    /** The constructor that initializes each Document object using the information in each file.
     *
     * @param fileName the name of the file
     * @param fileAbsolutePath  the absolute path of the file
     * @param isTestFile    indicates if the file is a test file (the sentences in a test file have no classification)
     */
    public Document(String fileName, String fileAbsolutePath, boolean isTestFile){
        _hasConverged = false;
        this._fileName = fileName;
        _fileId = Main.getNumberInFileName(fileName);                                               //helper method defined in Main that returns the numbers in the file name
        this._sentences = new LinkedList<Sentence>();
        this._sentenceEdges = new LinkedList<Edge>();
        try{
            BufferedReader fileReader = new BufferedReader(new FileReader(fileAbsolutePath));
            String lineRead = fileReader.readLine();                                                //discard first line as it only has the column titles
            while((lineRead = fileReader.readLine()) != null){
                //System.out.println(lineRead);
                Sentence toAdd = new Sentence(lineRead, isTestFile);                                //create a corresponding Sentence object for each sentence in the file
                this._docPosPolarCount += toAdd._positive_polar_count;                              //the total sum of positive polar words in the doc is the sum of polar polar words in each sentence
                this._docNegPolarCount += toAdd._negative_polar_count;                              //the total sum of negative polar words in the doc is the sum of the negative polar words in each sentence
                //toAdd.printSentenceInfo();
                _sentences.add(toAdd);
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    /**The function that initializes the vectors associated with each sentence in the current Document object.
     * Note that each line in the corresponding vectors file is stored in the following format: sentence_id [<val_1> <val_2> ... <val_n>]
     *
     * @param vectorAbsolutePath the absolute path to the file that contains the sentence vectors for the current Document object
     */
    public void initializeDocVector(String vectorAbsolutePath){
        try{
            BufferedReader vectorReader = new BufferedReader(new FileReader(vectorAbsolutePath));
            String lineRead = vectorReader.readLine();                                              //discard the first line as it only has the column titles
            while((lineRead = vectorReader.readLine()) != null){
                //System.out.println(lineRead);
                String[] lineTokens = lineRead.split(",\\s|\\t\\[|\\]");                      //tokenize the line that was read

                int sentenceId = Integer.parseInt(lineTokens[0]);                                   //by the given format the first token is the sentence ID
                Sentence toInitialize = getSentenceFromID(sentenceId);                              //use helper method to get corresponding Sentence object from the extracted sentence ID
                double[] _vectorArrayToInitialize = toInitialize.getVectorArray();
                int i = 1;
                while(i < lineTokens.length){
                    _vectorArrayToInitialize[i-1] = Double.parseDouble(lineTokens[i]);
                    i++;
                }
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    /** Helper function that returns the requested Sentence object in _sentences using the sentence ID
     *
     * @param sentID the unique sentence ID in a given document
     * @return
     */
    public Sentence getSentenceFromID(int sentID){
        for(Sentence s : _sentences){
            if(s._sent_id == sentID) {
                return s;
            }
        }
        return null;
    }

    /**This function initializes the directed graph represented by _edges. For implementing HITS we need an edge from
     * each Sentence to every other Sentence. The authority score is initialized to the probability the sentence is a fact as given by the naive bayes classifier
     * The hub score is initialized to the probability the sentence is an opinion as given by the naive bayes classifier
     *
     */
    public void initializeSentenceEdges(){
        //iterating through all sentences and setting initial values of authority score and hub score to corresponding probabilities given by the naive bayes classifier
        for(Sentence s : _sentences){
            s._authorityScore = s._isFactProbability;
            s._hubScore = s._isOpinionProbability;
        }

        //creating the directed graph. An edge is created from each Sentence to every other Sentence
        for(Sentence s_1 : _sentences){
            for(Sentence s_2 : _sentences){
                if(s_1 == s_2){                         //we don't want to create an edge from a given Sentence to itself
                    continue;
                }
                else{
                    //Weight of each edge given by: W_i = (Sim(s_i, s_j))^2 * (Hub_Score_s_i)^3 * (1 + 1/d)
                    // where i --> is the from Sentence, j is the to Sentence and d is the absolute value of distance between the sentences
                    double weight = s_1.computeSentenceSimilarity(s_2)*s_1.computeSentenceSimilarity(s_2);
                    weight *= (s_1._hubScore*s_1._hubScore*s_1._hubScore);
                    weight *= (1.0 + (1.0/(Math.abs(s_1._sent_id - s_2._sent_id))));
                    //System.out.println("IN INIIALIZE SENTENCES: WEIGHT" +weight);
                    Edge toAdd = new Edge(s_1, s_2, weight);
                    _sentenceEdges.add(toAdd);
                }
            }
        }
    }

    /** This function represents a single iteration of the iterative HUBS algorithm.
     * For each sentence- new hub score and authority score values are calculated- the scores are updated
     * ONLY IF the mean square error of the authority scores and hub scores is greater than a defined threshold MSE value
     */
    public void updateSentenceEdgesMSE(){
        if(this._hasConverged){                     //hasConverged indicates if the values associated with iterative HUBS have converged
            return;
        }

        LinkedList<Sentence> sentencesCopy = new LinkedList<Sentence>();    //create a copy of the sentences in the current Document
                                                                            //use this copy to represent next iteration sentences (with new hub and authority scores)
        for(Sentence s: this._sentences){
            sentencesCopy.add(s.deepCopy());
        }

        //compute the new authority score for each sentence
        for(Sentence from : this._sentences) {
            double _newAuthorityScore = 0.0;
            //iterate through all the edges in the directed graph
            for (Edge edge : this._sentenceEdges) {
                if (edge._to == from) {                                         //identifies all the Edges that are directed INTO the current Sentence
                    _newAuthorityScore += (edge._edgeWeight * edge._from._hubScore);
                }
            }
            getSentenceFromSentID(sentencesCopy, from._sent_id)._authorityScore = _newAuthorityScore;   //store the new authority score in the corresponding duplicate
                                                                                                        // sentence in sentencesCopy
        }

        //compute the new hub score for each sentence (using the NEW AUTHORITY SCORES)
        for(Sentence from : _sentences){
            double _newHubScore = 0.0;
            for(Edge edge : _sentenceEdges){
                if(edge._from == from){                                         //identifies all Edges that are directed OUT of the current Sentence
                    _newHubScore += (edge._edgeWeight * getSentenceFromSentID(sentencesCopy, edge._to._sent_id)._authorityScore);   //observe NEW AUTHORITY scores used
                    //_newHubScore += (edge._edgeWeight * edge._to._authorityScore);                                                //observe NEW authority scores NOT USED.
                }
            }
            getSentenceFromSentID(sentencesCopy, from._sent_id)._hubScore = _newHubScore;   //store the new hub score in the corresponding duplicate
                                                                                            //sentence in sentenceCopy
        }

        //normalize the NEW authority and hub scores
        double ssAuthority = 0.0;                       //sum of squares of NEW authority scores
        double ssHub = 0.0;                             //sum of squares of NEW hub scores
        for(Sentence s : sentencesCopy){
            ssAuthority += (s._authorityScore*s._authorityScore);
            ssHub += (s._hubScore*s._hubScore);
        }
        for(Sentence s: sentencesCopy){                 //iterate through sentences copy to NORMALIZE NEW SCORES
            s._hubScore = s._hubScore/Math.sqrt(ssHub);
            s._authorityScore = s._authorityScore/Math.sqrt(ssAuthority);
        }


        //calculate the mean square errors using the old and new authority and hub scores
        double mse_authority = 0.0;
        double mse_hub = 0.0;
        for(Sentence s : _sentences){       //loop calculates squared errors for new authority and hub scores
            mse_authority += Math.pow((s._hubScore - getSentenceFromSentID(sentencesCopy, s._sent_id)._hubScore), 2) ;
            mse_hub += Math.pow((s._authorityScore - getSentenceFromSentID(sentencesCopy, s._sent_id)._authorityScore), 2);
        }
        mse_authority /= _sentences.size() -1;  //divide by (n-1) to get the MSE
        mse_hub /= _sentences.size() -1;        //divide by (n-1) to get the MSE

        //double ave_mse = (mse_authority+mse_hub)/2;
        //System.out.println("------------------MSE AUTHORITY----------------"+mse_authority);
        //System.out.println("------------------MSE HUB----------------"+mse_hub);
        //System.out.println("-----------------------MSE AVE------------------"+ave_mse);

        //we check if the hub values and authority values have converged using the evaluated MSE values and a
        //predefined threshold MSE value
        if(mse_authority < EPSILON_CONVERGENCE_FACTOR_MSE && mse_hub < EPSILON_CONVERGENCE_FACTOR_MSE){
            _hasConverged = true;                   //indicates the values have converged
            return;                                 //return from function as values have converged
        }

        //this code only executes WHEN VALUES HAVEN'T CONVERGED
        for(Sentence s: _sentences){                //update the authority and hub scores to the evaluated NEW authority and hub scores respectively
            s._authorityScore = getSentenceFromSentID(sentencesCopy, s._sent_id)._authorityScore;
            s._hubScore = getSentenceFromSentID(sentencesCopy, s._sent_id)._hubScore;
        }

        updateEdgeWeights();    //finally update the edge weights
    }

    /**This function updates the edge weights.
     * Weight of each edge given by: W_i = (Sim(s_i, s_j))^2 * (Hub_Score_s_i)^3 * (1 + 1/d)
     */
    public void updateEdgeWeights(){
        for(Edge edge : this._sentenceEdges){
            double weight = edge._from.computeSentenceSimilarity(edge._to)*edge._from.computeSentenceSimilarity(edge._to);
            weight *= (edge._from._hubScore*edge._from._hubScore*edge._from._hubScore);
            weight *= (1.0 + (1.0/(Math.abs(edge._from._sent_id - edge._to._sent_id))));
            edge._edgeWeight = weight;
        }
    }

    /** Creates a NEW Document object that is a deep copy of the current object
     *
     * @return the NEW Document object
     */
    public Document deepCopy(){
        Document toReturn = new Document();
        toReturn._fileName = this._fileName;
        toReturn._docNegPolarCount = this._docNegPolarCount;
        toReturn._docPosPolarCount = this._docPosPolarCount;
        toReturn._hasConverged = this._hasConverged;

        LinkedList<Sentence> _sentencesCopy = new LinkedList<Sentence>();
        for(Sentence toAdd : this._sentences){
            _sentencesCopy.add(toAdd.deepCopy());
        }
        toReturn._sentences = _sentencesCopy;

        LinkedList<Edge> _edgesCopy = new LinkedList<Edge>();
        for(Edge toAdd : this._sentenceEdges){
            _edgesCopy.add(toAdd.deepCopy());
        }
        toReturn._sentenceEdges = _edgesCopy;
        return toReturn;
    }

    /**Uses the SortSentencesByOpinionDesc Comparator defined at the end of this file
     * to print sentences in descending order based on their opinion score
     *
     */
    public void printSentencesByOpinionDesc(){
        LinkedList<Sentence> _sentencesCopy = new LinkedList<Sentence>();
        for(Sentence _toCopy : this._sentences){
            _sentencesCopy.add(_toCopy.deepCopy());
        }
        System.out.println("(Order by Naive Bayes Score Desc) Printing Sentences for: "+this._fileName);
        Collections.sort(_sentencesCopy, new SortSentencesByOpinionDesc());
        int count = 0;
        for(Sentence s : _sentencesCopy){
            if(count >= NUMBER_OF_SENTENCES_TO_PRINT){
                break;
            }
            s.printSentence();
            count++;
        }
        System.out.println("---- (Naive Bayes) Finished Printing ----");
    }

    /**Uses the SortSentencesByHubScores Comparator defined at the end of this file
     * to print sentences in descending order based on their hub scores
     *
     */
    public void printSentencesByHubScoresDesc(){
        LinkedList<Sentence> _sentencesCopy = new LinkedList<Sentence>();
        for(Sentence _toCopy : this._sentences){
            _sentencesCopy.add(_toCopy.deepCopy());
        }
        Collections.sort(_sentencesCopy, new SortSentencesByHubScores());
        System.out.println("(Order by HUB Score Desc) Printing Sentences for: "+this._fileName);
        int count = 0;
        for(Sentence s : _sentencesCopy){
            if(count >= NUMBER_OF_SENTENCES_TO_PRINT){
                break;
            }
            count++;
            s.printSentence();
        }
        System.out.println("---- (HUB Score) Finished Printing ----");
    }

    /**Prints the most opinionated sentence (by hub score) and corresponding top two authorities
     *
     */
    public void printTopHubScoreSentence(){
        Collections.sort(_sentences, new SortSentencesByHubScores());
        System.out.println("\n\n---- Printing Top Opinion (By Hub Score) Sentence and Corresponding Top Authorities:-----");
        Sentence top_sentence = _sentences.get(0);
        System.out.println("Printing most opinionated sentence.");
        top_sentence.printSentence();
        System.out.println();
        LinkedList<Sentence> _potentialTopAuthorities = new LinkedList<Sentence>();
        for(Edge edge : _sentenceEdges){
            if(edge._from == top_sentence){
                double metric = edge._from.computeSentenceSimilarity(edge._to)*edge._to._authorityScore;
                edge._to._topSupportingAuthorityScore = metric;
                //System.out.println("From - To - Metric Score");
                //System.out.println(edge._from._sent_id +"\t"+edge._to._sent_id+"\t"+metric);
                //System.out.println(edge._from.computeSentenceSimilarity(edge._to)+"\t"+edge._to._authorityScore+"\n\n");
                _potentialTopAuthorities.add(edge._to);
            }
        }
        Collections.sort(_potentialTopAuthorities, new SortSentencsByTopSupportingAuthorityScores());
        System.out.println("Printing Corresponding Top Authorities.");
        for(int i = 0; i < NUMBER_OF_TOP_SUPPORTING_AUTHORITIES_TO_PRINT ; i++){
            System.out.print("("+(i+1)+".)"+"\t");
            _potentialTopAuthorities.get(i).printSupportingAuthoritySentence();
        }
        System.out.println("---- Finished Printing Top Hub Score Sentence and Corresponding Top Authorities:-----");
    }

    /** Prints the sentence vector of each sentence
     */
   public void printSentencesVectors(){
        System.out.println("Printing sentence vectors for: "+this._fileName);
        for(Sentence s : this._sentences){
            s.printSentenceVector();
        }
    }

    /**Prints the all the edges in the directed graph represented by the current document
     */
    public void printSentenceEdges(){
        int count = 1;
        for(Edge edge : this._sentenceEdges){
            System.out.println(count + ". From:"+edge._from._sent_id+"\t:\tTo:"+edge._to._sent_id+"\tWeight:"+edge._edgeWeight );
            edge._from.printSentenceInfo();
            count++;
        }
    }

    /**Used for evaluation on train data
     */
    public void calculatePrecisionAt3Comparison(){
        LinkedList<Sentence> _sentencesCopy = new LinkedList<Sentence>();
        for(Sentence _toCopy : this._sentences){
            _sentencesCopy.add(_toCopy.deepCopy());
        }
        Collections.sort(_sentencesCopy, new SortSentencesByHubScores());
        System.out.println("Printing Precision Comparison for: "+this._fileName);
        double _correctylClassified = 0;
        int  _topThree = 0;
        for(Sentence s : _sentencesCopy){
            if(_topThree < 3){
                if(s._classifiedAs == 'O'){
                    _correctylClassified++;
                }
                _topThree++;
            }
            else{
                break;
            }
        }
        System.out.print("(HUB Score P@3): "+(_correctylClassified/3.0) + "\t");
        _totalHubPrecisionAtThree += (_correctylClassified/3.0);

        Collections.sort(_sentencesCopy, new SortSentencesByOpinionDesc());
        _topThree = 0;
        _correctylClassified = 0.0;
        for(Sentence s: _sentencesCopy){
            if(_topThree < 3){
                if(s._classifiedAs == 'O' && s._isOpinionProbability > 0.50){
                    _correctylClassified++;
                }
                _topThree++;
            }
            else{
                break;
            }
        }
        System.out.print("(Naive Bayes P@3): "+(_correctylClassified/3.0) + "\n");
        _totalNaivePrecisionAtThree += (_correctylClassified/3.0);
    }

    /**Used for evaluation on train data
     */
    public void calculatePrecisionAt5Comparison(){
        LinkedList<Sentence> _sentencesCopy = new LinkedList<Sentence>();
        for(Sentence _toCopy : this._sentences){
            _sentencesCopy.add(_toCopy.deepCopy());
        }
        Collections.sort(_sentencesCopy, new SortSentencesByHubScores());
        System.out.println("Printing Precision Comparison for: "+this._fileName);
        double _correctylClassified = 0;
        int  _topFive = 0;
        for(Sentence s : _sentencesCopy){
            if(_topFive < 5){
                if(s._classifiedAs == 'O'){
                    _correctylClassified++;
                }
                _topFive++;
            }
            else{
                break;
            }
        }
        System.out.print("(HUB Score P@5): "+(_correctylClassified/5.0) + "\t");
        _totalHubPrecisionAtFive += (_correctylClassified/5.0);

        Collections.sort(_sentencesCopy, new SortSentencesByOpinionDesc());
        _topFive = 0;
        _correctylClassified = 0.0;
        for(Sentence s: _sentencesCopy){
            if(_topFive < 5){
                if(s._classifiedAs == 'O' && s._isOpinionProbability > 0.50){
                    _correctylClassified++;
                }
                _topFive++;
            }
            else{
                break;
            }
        }
        System.out.print("(Naive Bayes P@5): "+(_correctylClassified/5.0) + "\n");
        _totalNaivePrecisionAtFive += (_correctylClassified/5.0);
    }

    /**Helper function that returns the Sentence object corresponding to the corresponding sentence id passed as argument
     *
     * @param sentences the sentences list (which should have sentences with unique sentence ids)
     * @param sent_id   the sentence id of the requested Sentence object
     * @return
     */
    public static Sentence getSentenceFromSentID(LinkedList<Sentence> sentences, int sent_id){
        for(Sentence s : sentences){
            if(s._sent_id == sent_id){
                return s;
            }
        }
        return null;
    }

    /**Used for evaluation on train data
     */
    public static void printTrainPrecisionAtThreeSummaryPerformance(){
        System.out.println("\n\n---- SUMMARY PERFORMANCE P@3---");
        System.out.println("Naive Bayes: " + (_totalNaivePrecisionAtThree/NUMBER_OF_DOCS_IN_TRAIN));
        System.out.println("HUB Scores: " + (_totalHubPrecisionAtThree/NUMBER_OF_DOCS_IN_TRAIN));
    }

    /**Used for evaluation on train data
     */
    public static void printTrainPrecisionAtFiveSummaryPerformance(){
        System.out.println("\n\n---- SUMMARY PERFORMANCE P@5---");
        System.out.println("Naive Bayes: " + (_totalNaivePrecisionAtFive/NUMBER_OF_DOCS_IN_TRAIN));
        System.out.println("HUB Scores: " + (_totalHubPrecisionAtFive/NUMBER_OF_DOCS_IN_TRAIN));
    }
}

/**This class represents a single edge in the directed graph.
 */
class Edge {
    Sentence _from;
    Sentence _to;
    double _edgeWeight;

    /** A constructor that initializes an edge in the directed graph
     *
     * @param from  the "from" sentence (s_i)
     * @param to    the "to" sentence (s_j)
     * @param edgeWeight    the edge weight
     */
    public Edge(Sentence from, Sentence to, double edgeWeight){
        this._from = from;
        this._to = to;
        this._edgeWeight = edgeWeight;
    }

    /**Returns a NEW edge that is a deep copy of current object
     * @return
     */
    public Edge deepCopy(){
        Edge toReturn = new Edge(this._from, this._to, this._edgeWeight);
        return toReturn;
    }
}

/**Class that helps sort sentences in descending order by the naive bayes opinion probability score
 */
class SortSentencesByOpinionDesc implements Comparator<Sentence>{
    @Override
    public int compare(Sentence o1, Sentence o2) {
        if(o1._isOpinionProbability < o2._isOpinionProbability){
            return 1;
        }
        else if(o1._isOpinionProbability > o2._isOpinionProbability){
            return -1;
        }
        else{
            return 0;
        }
    }
}

/**Class that helps sort sentences in descending order by the hub score probability score
 */
class SortSentencesByHubScores implements Comparator<Sentence>{
    @Override
    public int compare(Sentence o1, Sentence o2) {
        if(o1._hubScore < o2._hubScore){
            return 1;
        }
        else if(o1._hubScore > o2._hubScore){
            return -1;
        }
        else{
            return 0;
        }
    }
}

/**Class that helps sort sentences in descending order by the top supporting authority score
 */
class SortSentencsByTopSupportingAuthorityScores implements Comparator<Sentence>{
    @Override
    public int compare(Sentence o1, Sentence o2) {
        if(o1._topSupportingAuthorityScore < o2._topSupportingAuthorityScore){
            return 1;
        }
        else if(o1._topSupportingAuthorityScore > o2._topSupportingAuthorityScore){
            return -1;
        }
        else{
            return 0;
        }
    }
}