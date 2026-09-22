# Python Loops

## What is a loop?

A loop repeats a block of code.

For example:

for i in range(3):
    print(i)

Output:

0
1
2

## range()

range(3) produces the sequence:

0, 1, 2

The ending value 3 is not included.

## Repeating an action

A loop can repeat the same action multiple times.

For example:

for i in range(2):
    print("Hello")

Output:

Hello
Hello

## Accumulating values

A variable can be updated during a loop.

For example:

total = 0

for i in range(1, 4):
    total += i

print(total)

Output:

6

The loop adds:

1 + 2 + 3

Therefore the final value of total is 6.

## Common misconception

A common beginner mistake is assuming that range(3) produces:

1, 2, 3

In Python, range(3) produces:

0, 1, 2

## Worked example

for i in range(3):
    print(i)

Output:

0
1
2