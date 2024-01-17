import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP Conj S | NP VP Conj S Adv
NP -> Det N | Det Adj N | Det Adj Adj N | Det N Conj NP | Det N P NP
NP -> P NP | NP P N | Det Adj N Conj NP | N | Det Adj Adj Adj N
VP -> V NP | V | V NP P | V Adv | Adv V | V P NP | V NP P NP
VP -> V P | Adv V NP | VP Conj VP | V NP Adv NP Conj VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Tokenize the sentence into a list of words
    words = nltk.word_tokenize(sentence)

    # Filter out words that do not contain at least one alphabetic character
    processed_words = [word.lower() for word in words if any(char.isalpha() for char in word)]
    return processed_words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []

    # Check if the current tree node has the label 'NP' and doesn't contain any other subtrees with the label 'NP'
    if tree.label() == 'NP' and not any(subtree.label() == 'NP' for subtree in tree.subtrees() if subtree != tree):
        chunks.append(tree)

    # Recursively calls itself on each subtree
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            chunks.extend(np_chunk(subtree))  # Search for more noun phrase chunks within the subtree
    return chunks


if __name__ == "__main__":
    main()
