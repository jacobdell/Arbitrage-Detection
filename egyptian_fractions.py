
def ceiling(x, y):
    """
    Compute the ceiling of y / x using integer arithmetic.

    This returns the smallest integer n such that n >= y / x.
    It is used in the greedy Egyptian fraction algorithm to
    determine the next unit fraction denominator.
    """
    if y % x == 0:
        return y // x
    return y // x + 1


def subtract(x, y, n):
    """
    Subtract the unit fraction 1/n from the fraction x/y.

    Given x/y and a positive integer n, this function returns
    integers (a, b) such that:

        a / b = x / y - 1 / n

    The result is not simplified; simplification is unnecessary
    for correctness of the Egyptian fraction algorithm.
    """
    a = (x * n) - y
    b = y * n
    return a, b


def egyptian_fraction(x, y):
    """
    Compute the Egyptian fraction decomposition of x/y.

    This function uses the greedy algorithm to express a positive
    rational number x/y as a sum of distinct unit fractions.
    It returns a list of denominators [n1, n2, ..., nk] such that:

        x / y = 1/n1 + 1/n2 + ... + 1/nk

    The recursion terminates when the numerator becomes zero.
    """
    if x == 0:
        return []
    greedy = ceiling(x, y)
    x, y = subtract(x, y, greedy)
    return [greedy] + egyptian_fraction(x, y)
