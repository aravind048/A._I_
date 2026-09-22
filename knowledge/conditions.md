# Python Conditions

## What is a condition?

A condition is an expression that evaluates to True or False.

For example:

x > 5

If x is 10, the condition is True.

## if and else

Python can use if and else to select between different actions.

For example:

x = 10

if x > 5:
    print("A")
else:
    print("B")

Output:

A

## False condition

If the condition is False, the else block is executed.

For example:

x = 3

if x > 5:
    print("A")
else:
    print("B")

Output:

B

## Equality comparison

The == operator checks whether two values are equal.

For example:

x = 10

if x == 10:
    print("Yes")
else:
    print("No")

Output:

Yes

## Common misconception

A common beginner mistake is confusing:

=

and:

==

The = operator is used for assignment.

The == operator is used to compare values.

## Worked example

x = 10

if x == 10:
    print("Yes")

Output:

Yes