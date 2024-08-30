"""Question1: Write a Python program to calculate the length of a string using 2 ways"""   #Levi Ackerman

# Method 1: Using len() function
string = input("Enter a string: ")
print("The length of the string is:", len(string))

# Method 2: Using loop
string = input("Enter a string: ")
count = 0
for char in string:
    count += 1
print("The length of the string is: ", count)

##############################################################################################
"""Question2: Write a Python program to get a string made of the first 2 and last 2 characters of a given string. 
If the string length is less than 2, return the empty string instead"""

string = input("Enter a string: ")

if len(string) < 2:
    print("Empty String")
else:
    print(string[:2] + string[-2:])

##############################################################################################
"""Question3: Write a Python program to add 'ing' at the end of a given string (length should be at least 3).
 If the given string already ends with 'ing', add 'ly' instead.
  If the string length of the given string is less than 3, leave it unchanged."""

string = input("Enter a string: ")

if len(string) < 3:
    print(string)
elif string[-3:] == "ing":
    print(string + "ly")
else:
    print(string + "ing")

##############################################################################################
"""Question4: Write a Python function that takes a list of words and return the longest word and the length of the longest one"""

def longest_word(words):
    longest = ""
    for word in words:
        if len(word) > len(longest):
            longest = word

    print("The longest word is: (" + longest + ")", "and its length is: ",len(longest))

longest_word(["Levi", "Ackerman", "Mikasa", "Armin", "Eren", "Historia", "Exercises"])

##############################################################################################
"""Question5:Write a Python program to change a given string to a newly string where the first and last chars have been exchanged using 2 ways"""

# Method 1: Using string concatenation
string = "abca"
new_char = "e"
new_string = new_char + string[1:-1] + new_char
print("New string is:", new_string)

# Method 2: Using slicing & replace() method
string = "abca"
new_char = "e"
new_string = string.replace(string[0], new_char).replace(string[-1], new_char)
print("New string:", new_string)

##############################################################################################
"""Question6: Write a Python program to remove characters that have odd index values in a given string"""

string = input("Enter a string: ")
new_string = ""
for i in range(len(string)):
    if i % 2 == 0:
        new_string += string[i]
print("The new string is:", new_string)

#Another method
string = input("Enter a string: ")
new_string = "".join(char for index, char in enumerate(string) if index % 2 == 0)
print("The new string is:", new_string)
enumerate_list = list(enumerate(string))
print(enumerate_list)

##############################################################################################
"""Question7: You have a list of your favourite marvel super heros."""

heros=['spider man','thor','hulk','iron man','captain america']

"""1. Length of the list"""
print("Length of the list is:", len(heros))

"""2. Add 'black panther' at the end of this list"""
heros.append("black panther")
print("Updated list:", heros)

"""3. Remove 'black panther' from the list then add it after 'hulk' """
heros.remove("black panther")
heros.insert(3, "black panther")
print("Updated list:", heros)

"""4. Remove thor and hulk from the list and replace them with doctor strange in a one line-code."""
heros[1:3] = ["doctor strange"]
print("Updated list:", heros)

"""5. Sort the heros list in alphabetical order (Hint. Use dir() functions to list down all functions available in list)"""
heros.sort()
print("Sorted list:", heros)

##############################################################################################
"""Question8: Write a Python script that takes input from the user and displays that input back in upper and lower cases"""

string = input("Enter a string: ")
print("Upper case:", string.upper())
print("Lower case:", string.lower())

##############################################################################################
"""Question9: Write a Python function to reverse a string if its length is a multiple of 4"""

def reverse_string(string):
    if len(string) % 4 == 0:
        return string[::-1]
    return string

string = input("Enter a string: ")
print("Reversed string is:", reverse_string(string))

##############################################################################################
"""Question10: Write a Python program to remove a newline in Python"""

print("FirstLine", end=" ")
print("SecondLine")

##############################################################################################
"""Question11: Write a Python program to add prefix text to all of the lines in a string"""

string = """Line1
Line2
Line3"""
prefix = "Prefix"
new_string = "\n".join(prefix + line for line in string.split("\n"))
print(new_string)

##############################################################################################
"""Question12: Write a Python program to print the following numbers up to 2 decimal places"""

number = float(input("Enter a number: "))
print("Original Number:", number)
print("Formatted Number: {:.2f}".format(number))

##############################################################################################
"""Question13: Write a Python program to print the following numbers up to 2 decimal places with a sign"""

number = float(input("Enter a number: "))
print("Original Number:", number)
print("Formatted Number with sign: {:+.2f}".format(number))

##############################################################################################
"""Question14: Write a Python program to display a number with a comma separator"""

number = int(input("Enter a number: "))
print("Original Number:", number)
print("Formatted Number with comma separator: {:,}".format(number))

##############################################################################################
"""Question15: Write a Python program to reverse a string using 2 ways"""

# Method 1: Using slicing
string = input("Enter a string: ")
print("Reversed string is:", string[::-1])

# Method 2: Using loop
string = input("Enter a string: ")
reversed_string = ""
for char in string:
    reversed_string = char + reversed_string
print("Reversed string is:", reversed_string)

##############################################################################################
"""Question16: Write a Python remove spaces from a given string"""

string = input("Enter a string: ")
print("Original string:", string)
print("String without spaces:", string.replace(" ", ""))

##############################################################################################
"""Question17: Write a Python program to swap first and last element of any list."""

list1 = [12, 35, 9, 56, 24]
print("Original list:", list1)
list1[0], list1[-1] = list1[-1], list1[0]
print("List after swapping first and last element:", list1)

##############################################################################################
"""Question18: Given a list in Python and provided the positions of the elements, write a program to swap the two elements in the list."""
list1 = [23, 65, 19, 90]
pos1, pos2 = 1, 3
print("Original list:", list1)
list1[pos1], list1[pos2] = list1[pos2], list1[pos1]
print("List after swapping elements at positions", pos1, "and", pos2, ":", list1)

##############################################################################################
"""Question19: search for the all ways to know the length of the list"""

"""1. Using len() method"""
list1 = [12, 35, 9, 56, 24]
print("The length of the list is:",len(list1))

"""2. Using len() method with str() method
This method first converts the list into a string then counts the commas in it.
Then it adds 1 to the count because in a list the number of commas is always one less than the number of elements."""

list1 = [12, 35, 9, 56, 24]
print("The length of the list is:", str(list1).count(',') + 1)

"""3. Using for loop"""
list1 = [12, 35, 9, 56, 24]
count = 0
for _ in list1:
    count += 1
print("The length of the list is:",count)

"""4. Using list comprehension: This method uses a list comprehension to create a new list of 1s with the same length as the original list, and then sums the new list."""
list1 = [12, 35, 9, 56, 24]
print("The length of the list is:",sum([1 for _ in list1]))

"""5. Using reduce() method from the `functools` module: This method uses the `reduce()` function to apply a function that increments a counter to all elements in the list."""
from functools import reduce
list1 = [12, 35, 9, 56, 24]
print("The length of the list is:",reduce(lambda count, _: count + 1, list1, 0))

"""6. Using the enumerate() method: This method uses the `enumerate()` function to create an iterator that yields the index and value of each element in the list, and then counts the number of elements."""
list1 = [12, 35, 9, 56, 24]
print("The length of the list is:",sum(1 for _ in enumerate(list1)))

"""7. Using the `map()` function: This method uses the `map()` function to apply a function that increments a counter to all elements in the list."""
list1 = [12, 35, 9, 56, 24]
print("The length of the list is:",sum(map(lambda _: 1, list1)))

"""8. Using the `zip()` function: This method uses the `zip()` function to create an iterator that pairs each element in the list with a placeholder value, and then counts the number of elements."""
list1 = [12, 35, 9, 56, 24]
print("The length of the list is:",sum(1 for _, _ in zip(list1, [0] * len(list1))))

##############################################################################################
"""Question20: write a Python code to find the Maximum number of list of numbers."""

list1 = [12, 35, 9, 56, 24]
print("Maximum number in the list is:", max(list1))

##############################################################################################
"""Question21: write a Python code to find the Minimum number of list of numbers."""

list1 = [12, 35, 9, 56, 24]
print("Minimum number in the list is:", min(list1))

##############################################################################################
"""Question22: search for if an elem is existing in list."""

list1 = ["Levi", "Ackerman", "Mikasa", "Armin", "Eren", "Historia", "Exercises"]
elem = "Eren"
if elem in list1:
    print(elem, "is in the list")
else:
    print(elem, "is not in the list")

##############################################################################################
"""Question23: clear python list using different ways."""

"""1. Using clear() method"""
list1 = [12, 35, 9, 56, 24]
list1.clear()
print("List after clearing:", list1)

"""2. Using reassignment"""
list1 = [12, 35, 9, 56, 24]
list1 = []
print("List after clearing:", list1)

"""3. Using slicing"""
list1 = [12, 35, 9, 56, 24]
list1[:] = []
print("List after clearing:", list1)

"""4. Using pop() method"""
list1 = [12, 35, 9, 56, 24]
while list1:
    list1.pop()
print("List after clearing:", list1)

"""5. Using remove() method"""
list1 = [12, 35, 9, 56, 24]
for element in list1:
    list1.remove(element)
print("List after clearing:", list1)

##############################################################################################
"""Question24: remove duplicated elements from a list."""

list1 = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
list1 = list(set(list1))
print("List after removing duplicates:", list1)

#Another method
list1 = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
list2 = []
[list2.append(element) for element in list1 if element not in list2]
print("List after removing duplicates:", list2)

#Another method
list1 = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
list2 = []
for element in list1:
    if element not in list2:
        list2.append(element)
print("List after removing duplicates:", list2)

##############################################################################################
"""Question25: Write a python program to count unique values inside a list using different ways."""

list1 = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
unique_values = set(list1)
print("Number of unique values in the list:", len(unique_values))

#Another method
list1 = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
unique_values = []
[unique_values.append(element) for element in list1 if element not in unique_values]
print("Number of unique values in the list:", len(unique_values))

#Another method
list1 = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
unique_values = []
for element in list1:
    if element not in unique_values:
        unique_values.append(element)
print("Number of unique values in the list:", len(unique_values))

##############################################################################################
"""Question26: Write a python program to Extract all elements with Frequency greater than K."""

test_list = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
K = 3
freq = {}
for i in test_list:
    if i in freq:
        freq[i] += 1
    else:
        freq[i] = 1
output_list = [key for key, value in freq.items() if value > K]
print("Elements with frequency greater than", K, "are:", output_list)

##############################################################################################
"""Question27: Write a python program to find the Strongest Neighbour."""

list1 = [1, 2, 2, 3, 4, 5]
strongest_neighbour = max(list1)
print("Strongest neighbour is:", strongest_neighbour)

##############################################################################################
"""Question28: Write a python program to print all Possible Combinations from the three Digits."""

from itertools import permutations
input_list = [1, 2, 3]
permutations_list = permutations(input_list)
print("All possible combinations are:", end=" ")
for permutation in permutations_list:
    print(' '.join(map(str, permutation)), end=" ## ")

#Another method
digits = [1, 2, 3]
n = len(digits)
print("All possible combinations are:", end=" ")
for i in range(n):
    for j in range(n):
        if i != j:
            for k in range(n):
                if i != k and j != k:
                    print(digits[i], digits[j], digits[k], end=" ## ")

##############################################################################################
"""Question29: Write a python program to find all the Combinations in the list with the given condition."""

test_list = [1, 2, 3]
output_list = []
for i in range(len(test_list)):
    for j in range(i, len(test_list)):
        output_list.append(test_list[i:j + 1])
print("All possible combinations are:", output_list)

##############################################################################################
"""Question30: Write a python program to get all unique combinations of two Lists (List_1 = ["a","b"] List_2 = [1,2] Unique_combination = [[('a',1),('b',2)],[('a',2),('b',1)]] )."""

from itertools import product
list1 = ["a", "b"]
list2 = [1, 2]
unique_combinations = list(product(list1, list2))   #product() function returns the Cartesian product of the input iterables.
print("Unique combinations are:", unique_combinations)

#Another method
list1 = ["a", "b"]
list2 = [1, 2]
unique_combinations = [[(i, j) for j in list2] for i in list1]
print("Unique combinations are:", unique_combinations)

##############################################################################################
"""Question31: Write a Python program that finds all pairs of elements in a list whose sum is equal to a given value."""

list1 = [2, 4, 3, 5, 7, 8, 9]
sum_value = 7
pairs = []
for i in range(len(list1)):
    for j in range(i + 1, len(list1)):
        if list1[i] + list1[j] == sum_value:
            pairs.append((list1[i], list1[j]))
print("Pairs with sum equal to", sum_value, "are:", pairs)

##############################################################################################
"""Question32: I have a string variable called s='maine 200 banana khaye'. This of course is a wrong statement, the correct statement is 'maine 10 samosa khaye'. Replace incorrect words in original strong with new ones and print the new string. Also try to do this in one line."""

print(' '.join((s := 'maine 200 banana khaye'.replace('200', '10').replace('banana', 'samosa')).split()[i] for i in [0, 2, 1, 3]))

##############################################################################################
"""Question33: The same as question 7 it's repeated."""