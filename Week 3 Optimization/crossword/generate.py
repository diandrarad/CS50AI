import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }


    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters


    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()


    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        bbox = draw.textbbox((0, 0), letters[i][j], font=font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)


    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())


    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:
            length = var.length
            domain = self.domains[var].copy()  # Make a copy to iterate over while removing values

            # Check each value in the domain and compare its length with the variable's length
            for value in domain:
                if len(value) != length:
                    self.domains[var].remove(value)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False

        # Check if there is an overlap between variables x and y
        overlap = self.crossword.overlaps[x, y]

        if overlap is not None:
            i, j = overlap
            domain_x = self.domains[x].copy()

            # Iterate over each value value_x in the domain of x
            for value_x in domain_x:

                has_corresponding_value = any(
                    value_x[i] == value_y[j] for value_y in self.domains[y]
                )  # Keep value_x in the domain of x if there is at least one corresponding value

                # Otherwise, remove value_x from the domain of x
                if not has_corresponding_value:
                    self.domains[x].remove(value_x)
                    revised = True

        return revised


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = self.crossword.overlaps.keys()  # Set arcs to be all the arcs in the problem

        queue = list(arcs)

        while queue:  # Continue until the queue is empty
            x, y = queue.pop(0)  # Extract an arc (x, y) from the front of the queue

            if self.revise(x, y):  # Call the revise function to make x arc consistent with y
                if len(self.domains[x]) == 0:  # Check if the domain of x is empty
                    return False  # It's impossible to solve the problem with an empty domain

                # Add all the neighbors of x except y to the queue to ensure their consistency with x
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))

        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        variables = self.crossword.variables

        # Check if all crossword variables present as a key in the assignment dictionary
        return all(variable in assignment for variable in variables)


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check for word uniqueness
        values = set(assignment.values())
        if len(values) != len(assignment):
            return False

        # Check for conflicts
        for variable1, word1 in assignment.items():
            for variable2, word2 in assignment.items():
                if variable1 != variable2:
                    overlap = self.crossword.overlaps[variable1, variable2]
                    if overlap:
                        i, j = overlap
                        if word1[i] != word2[j]:
                            return False

        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        domain = list(self.domains[var])  # All the values in the domain of the current variable
        count_values = []  # Create a list to store tuples of values and their corresponding counts

        for value in domain:
            count = 0

            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]

                    # Count the number of the neighbor's domain values that aren't equal to the current value
                    if overlap is not None:
                        count += sum(1 for val in self.domains[neighbor] if val != value)

            count_values.append((value, count))  # Store the value along with its count as a tuple

        count_values.sort(key=lambda x: x[1])  # Sort the count_values list based on the count
        domain = [value for value, _ in count_values]  # Update the domain with the value from the count_values list

        return domain


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Sort the unassigned variables based on the number of remaining values in their domain (minimum remaining value heuristic)
        unassigned_variables = [var for var in self.domains.keys() if var not in assignment]
        unassigned_variables.sort(key=lambda var: len(self.domains[var]))

        # Extract the tied variables, which are the variables with the same number of remaining values
        min_remaining_values = len(self.domains[unassigned_variables[0]])
        tied_variables = [var for var in unassigned_variables if len(self.domains[var]) == min_remaining_values]

        # If there is only one tied variable, return it as the result
        if len(tied_variables) == 1:
            return tied_variables[0]

        # Sort the tied variables based on the number of neighbors (degree heuristic)
        tied_variables.sort(key=lambda var: len(self.crossword.neighbors(var)), reverse=True)

        return tied_variables[0]


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment  # Return the assignment as it is a satisfactory solution

        var = self.select_unassigned_variable(assignment)

        # Iterate over the selected variable's domain's values ordered by the order_domain_values function
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value  # Assign the value to the variable, add it to the assignment

            if self.consistent(assignment):
                result = self.backtrack(assignment)  # Recursively call the backtrack function

                if result is not None:
                    return result  # Terminate early and return a valid assignment as soon as one is found

            del assignment[var]  # Remove the current value for the variable and continue with the next value

        return None  # No satisfying assignment is possible


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
