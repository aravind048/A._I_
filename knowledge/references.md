# Python References

## What is a reference?

A Python variable refers to an object. The variable name
is associated with that object.

## Assignment and references

When we write:

x = [1, 2, 3]
y = x

Python does not automatically create a new list.
Both x and y refer to the same list object.

## Mutable objects

Lists are mutable objects. Their contents can be changed
after the list is created.

For example:

y.append(4)

changes the list object.

## Common misconception

A common misconception is that:

y = x

creates a completely independent copy of the list.

It does not. Both variables can refer to the same list.

## Worked example

x = [1, 2, 3]
y = x
y.append(4)

print(x)

Output:

[1, 2, 3, 4]

Both x and y refer to the same list.