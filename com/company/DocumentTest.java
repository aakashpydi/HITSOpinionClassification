package com.company;

import junit.framework.TestCase;

public class DocumentTest extends TestCase {
    private static final String TEST_DOCUMENT_ABSOLUTE_PATH = "C:\\Users\\Aakash Pydi\\IdeaProjects\\Project2_CS473\\project2\\train_data\\train_files\\train_0.tsv";

    public void testDocumentConstructor(){
        Document test_1 = new Document("train_0.tsv", TEST_DOCUMENT_ABSOLUTE_PATH, false);
        assertEquals(60 , test_1._sentences.size());
        assertEquals("train_0.tsv" , test_1._fileName);
        assertEquals(0, test_1._fileId);
    }

    public void testGetSentenceFromId(){
        Document test_1 = new Document("train_0.tsv", TEST_DOCUMENT_ABSOLUTE_PATH, false);
        assertEquals(12, test_1.getSentenceFromID(12)._sent_id);

    }

    public void testGetNumberInFileName() throws Exception {
        Document test_1 = new Document("test_0.tsv");
        assertEquals(0, Main.getNumberInFileName(test_1._fileName));

        Document test_2 = new Document("test_9.tsv");
        assertEquals(9, Main.getNumberInFileName(test_2._fileName));

        Document test_3 = new Document("train_32.tsv");
        assertEquals(32, Main.getNumberInFileName(test_3._fileName));
    }

    public void testInitializeSentenceEdges(){
        Document test_1 = new Document("train_0.tsv", TEST_DOCUMENT_ABSOLUTE_PATH, false);
        test_1.initializeSentenceEdges();
        assertEquals(3540, test_1._sentenceEdges.size()); //60 sentences. :. (60*60)-60
    }

}