# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:08:33 2018

@author: Leo
"""
# Indentation
for x in array:
    if x < 0:
        print("Negative")
    elif x > 0:
        print("Positive")
        
        
# Variables and argument passing
a = 0                    # assign 0 to variable a
a = b = c = 0            # assign 0 to variable a, b, c at same time 
a = 1; b = 2; c = 2      # Use semicolon to separate multiple statement
a, b, c = 1, 2, 3        # equavelent to previous line
a, b = b, a              # swap the value for a and b
print(a, b)

# Numbers
int1 = 123456789
int1**5
type(int1)

float1 = 3.1415    
float2 = 5.69e-6         # use scitific notation
type(float1)
type(float2)

# Strings
s1 = 'Hello, world'           # Single quotations
s2 = "Hello, world"           # Double quotations
type(s1)
s1 == s2

c = """This is a longer string that 
spans multiple lines"""
c

a = "The first string "
b = "and the second string"
a + b

# Booleans
x = True
y = False
x and y
x or y

# None
x = None
y = 3
x is None
y is not None

type(x)
type(y)


# print statement
letter = 'c'
print(letter )                              # print letter
number = 42
print("the number is:", number)             # prints "the number is: 42"


# List
[1, 2, 3]                  # List with three items with the same type
[4, 5.67, "Python"]        # List with three items but they are different types
[45]                       # List with one item
[]                         # Empty list


# indexing and slicing a list
L = [12, 90, 4, 5.67, "Python"]                          
L[0]                       # Select the first element
L[2]                       # Select the third element
L[1:4]                     # Select from the second to the fourth elements                         
L[:3]                      # Select from the first element to the third element                             
L[2:]                      # Select from the third element to the end
L[::-1]                    # reverse the list                                  

# List practice
L = [3, 7, 2, 56, 23, 21]
L.append(21)               # Append 21 at the end of L
L
L.extend([3, 8])           # Append 3 and 8 at the end of L
L
L.insert(4, 100)           # Insert 100 as the fifth element   
L
L.remove(56)               # Remove 56 from L
L
L.pop()                    # Remove the first element
L
L.pop(3)                   # Remove the fourth element
L
L.index(23)                # Return the index of 23
L.count(3)                 # Count how many 3 in L
L.reverse()                # Reverse L
L
L.sort()                   # Sort L in ascending order
L

# Tuple
tup1 = (1, 2, 3)              # Create a tuple variable called tup1 
tup1
tup2 = 1, 2, 3               # Equavalent to tup1 = (1, 2, 3)
tup2
type(tup1)
type(tup2)

# nested tuple
nested_tup = (4, 5, 6), (7, 8)   # Create a tuple variable called nested_tup
nested_tup

tuple('hello')             
tuple([2, 3, 4, 5])
tup1
tup1[1]
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)                # Count how many 2 in a
a.index(2)                # Return the index of the first 2 in a  

# set
set([2, 2, 2, 1, 3, 3])
{2, 2, 2, 1, 3, 3}

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
a.union(b)
a | b
a.intersection(b)
a & b

# dictionary
# Dictionary with three string keys and items
{'x':34, 'y':12, 'z':7}   # or dict(x = 34, y = 12, z = 7)  
# Dictionary with two integer keys and items
{1 :2, 2 : 3}             # or dict([1, 2], [2, 3])       
 # Empty dictionary                        
{}                        # or use dict() to creat a empty dictionary                                                             

# Modify a dict
d = {'x':34, 'y':12, 'z':7}                            
d['x']                   # Return the value for the key 'x'                                                
d['y'] = 0               # Change the value for the key 'y'   
d                                                                      
d['a'] = 78              # Insert 'a': 78 into d
d                             

# examples of how to use dict methods
d
0 in d
d.items()
d.keys()
d.values()
d.get('a')
d.pop('a')
d
d.popitem()
d


                             

# if else statement
x = 4
if x < 0:
    print('The input is negative')
elif x == 0:
    print('The input is equal to zero')
else:
    print('The input is positive')
    
# for loop
L1 = [3, 6, 2, 8, 78]
total = 0
for val in L1:
    total += val                       # calculate the sum of L1
total
sum(L1)  

total = 0  
for i in range(len(L)):                # Iterate a list by index:
    total += L1[i]
    
d = {'a': 3, 'b': 1, 'c': None, 'd': 6, None: '3'} 

for key, value in d.items():
    if not key or not value:          # If key or value is None 
       print(key, value)              # Print out key - value pairs


# while statement
L = [3, 6, 2, 8, 78]
n = len(L)
i = 0
while i < n:
    print(L[i])
    i += 1 



# break statement
# get the sum of the list, if the value is none, stop the loop
L2 = [11, 3, 135, None, 9]
result = 0
for i in range(len(L2)):
    if L2[i] is None:
        print('L2[', i, '] is not a valid number')
        break
    result += L2[i]

# Continue statment
result = 0
for value in L2:
    if value is None:
        continue
    result += value
result 

# pass statement

result = 0
# We can also skip the none value and continue to loop
for value in L2:
    if value is None:
        pass
    else:
        result += value
result

# Define function
def my_add(x, y):
    x + y  # this function doesn't return anything
my_add(1, 5)    


def multiply(x, y):
    return x*y
multiply(4, 8)