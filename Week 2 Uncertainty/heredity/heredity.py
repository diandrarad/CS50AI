import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set `have_trait` does not have the trait.
    """
    # Initialize joint probability to 1
    joint_prob = 1.0

    # Iterate over each person
    for person in people:
        # Determine the number of genes the person has
        num_genes = 2 if person in two_genes else 1 if person in one_gene else 0

        # Get the person's parents
        mother = people[person]['mother']
        father = people[person]['father']

        # Calculate the probability of having the gene based on parents
        if mother is None and father is None:
            prob_gene = PROBS["gene"][num_genes]
        else:
            # Probability of inheriting gene from mother
            prob_inherit_mother = PROBS["mutation"]
            if mother in one_gene:
                prob_inherit_mother = 0.5
            elif mother in two_genes:
                prob_inherit_mother = 1 - PROBS["mutation"]

            # Probability of inheriting gene from father
            prob_inherit_father = PROBS["mutation"]
            if father in one_gene:
                prob_inherit_father = 0.5
            elif father in two_genes:
                prob_inherit_father = 1 - PROBS["mutation"]

            # Calculate the probability of having the gene based on parents
            if num_genes == 0:
                prob_gene = (1 - prob_inherit_mother) * (1 - prob_inherit_father)
            elif num_genes == 1:
                prob_gene = (1 - prob_inherit_mother) * prob_inherit_father + prob_inherit_mother * (1 - prob_inherit_father)
            else:
                prob_gene = prob_inherit_mother * prob_inherit_father

        # Calculate the probability of having the trait
        prob_gene *= PROBS["trait"][num_genes][person in have_trait]

        # Multiply the individual probabilities to obtain the joint probability
        joint_prob *= prob_gene

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `one_gene` and `have_trait`, respectively.
    """
    # Iterate over each person in probabilities
    for person in probabilities:
        num_genes = 2 if person in two_genes else 1 if person in one_gene else 0

        # Update the "gene" distribution
        probabilities[person]["gene"][num_genes] += p

        # Update the "trait" distribution
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Iterate over each person
    for person in probabilities:

        # Normalize the "gene" distribution
        gene_distribution = probabilities[person]["gene"]
        for gene_count in gene_distribution:
            gene_distribution[gene_count] /= sum(gene_distribution.values())

        # Normalize the "trait" distribution
        trait_distribution = probabilities[person]["trait"]
        for trait_value in trait_distribution:
            trait_distribution[trait_value] /= sum(trait_distribution.values())


if __name__ == "__main__":
    main()
