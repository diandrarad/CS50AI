import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    page_probability = {}

    if page not in corpus or len(corpus[page]) == 0:
        # If the page has no outgoing links, choose randomly among all pages
        page_probability = {p: 1 / num_pages for p in corpus}
    else:
        # Calculate the probability of choosing a link from the current page
        link_probability = damping_factor / len(corpus[page])

        # Calculate the probability of choosing a random page from all pages
        random_probability = (1 - damping_factor) / num_pages

        # Assign probabilities to the linked pages
        for p in corpus:
            page_probability[p] = random_probability

        # Assign additional probability to the linked pages from the current page
        for link in corpus[page]:
            page_probability[link] += link_probability

    return page_probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}

    # Generate the first sample by choosing a page at random
    current_page = random.choice(list(corpus.keys()))
    page_rank[current_page] += 1 / n

    # Generate remaining samples based on transition model
    for _ in range(n - 1):
        probabilities = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
        page_rank[next_page] += 1 / n
        current_page = next_page

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    initial_rank = 1 / N # The initial PageRank value assumed for each page
    page_rank = {page: initial_rank for page in corpus} # Each page in the corpus is assigned the initial rank value
    convergence_threshold = 0.001 # The level of change in PageRank values that indicates convergence
    has_converged = False

    # The while loop continues until the PageRank values have converged
    while not has_converged:

        # Calculate the new ranks for all pages in the corpus
        new_page_rank = {}
        for page in corpus:
            rank = (1 - damping_factor) / N # The probability of choosing the page at random divided by N

            # Iterate over each linking page and its associated links
            for linking_page, links in corpus.items():
                if page in links:
                    # The PageRank of the linking page contributes to the rank calculation of the current page
                    rank += damping_factor * page_rank[linking_page] / len(links)

            # Store the updated calculated PageRank value for the current page
            new_page_rank[page] = rank

        # Check if all page differences are smaller than the convergence threshold, has_converged becomes True, and the while loop terminates
        has_converged = all(
            abs(new_page_rank[page] - page_rank[page]) < convergence_threshold
            for page in corpus
        )

        page_rank = new_page_rank # Updated with the new PageRank values stored in new_page_rank

    return page_rank


if __name__ == "__main__":
    main()
