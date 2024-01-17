import math
import nltk
import os
import sys
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_contents = {}

    for filename in os.listdir(directory):  # Get a list of all files in the directory
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)  # Construct the full file path using os.path.join

            with open(file_path, "r") as file:
                file_contents[filename] = file.read()  # Add an entry to the file_contents dictionary
    return file_contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Tokenize the document into words
    words = word_tokenize(document)

    # Filter out punctuation and stopwords
    filtered_words = [word.lower() for word in words if word.lower() not in string.punctuation and word.lower() not in stopwords.words("english")]

    return filtered_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}  # Dictionary to store IDF values

    total_documents = len(documents)  # Total number of documents

    # Count the number of documents in which each word appears
    word_documents_count = {}
    for document in documents.values():
        for word in set(document):
            word_documents_count[word] = word_documents_count.get(word, 0) + 1

    # Calculate IDF for each word
    for word, document_count in word_documents_count.items():
        idfs[word] = math.log(total_documents / document_count)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_scores = {}  # Dictionary to store file scores

    # Calculate the tf-idf score for each file
    for filename, words in files.items():
        score = 0
        for word in query:
            if word in words:
                score += words.count(word) * idfs[word]
        file_scores[filename] = score

    # Sort the files based on the scores in descending order
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

    # Get the top n files
    top_n_files = [filename for filename, _ in sorted_files[:n]]

    return top_n_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = {}  # Dictionary to store sentence scores

    # Calculate the IDF score and query term density for each sentence
    for sentence, words in sentences.items():

        # Calculate the relevance of the sentence to the query based on the IDF values of the matching words
        idf_score = sum(idfs[word] for word in query if word in words)

        # Calculate the proportion of words in the sentence that are also in the query
        query_density = sum(word in query for word in words) / len(words)
        sentence_scores[sentence] = (idf_score, query_density)

    # Sort the sentences based on the scores in descending order
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    # Get the top n sentences
    top_n_sentences = [sentence for sentence, _ in sorted_sentences[:n]]

    return top_n_sentences


if __name__ == "__main__":
    main()
