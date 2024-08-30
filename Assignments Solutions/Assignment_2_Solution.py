"""Question1: Write a Python function to perform linear search on a list"""

def linear_search(lst, target):
    for i in range(len(lst)):
        if lst[i] == target:
            return i
    return "Not Found"

# Test the function
print(linear_search([1, 2, 3, 4, 5], 3))  # 2
print(linear_search([1, 2, 3, 4, 5], 6))  # Not Found

##############################################################################################
"""Question2: Modify the function to return the index of all occurrences of the search element."""

def linear_search_indices(lst, target):
    indices = []
    for i in range(len(lst)):
        if lst[i] == target:
            indices.append(i)
    return "Not Found" if not indices else indices

# Test the function
print("The indices are:",linear_search_indices([1, 2, 3, 4, 3], 3))  # [2, 4]
print(linear_search_indices([1, 2, 3, 4, 5], 6))  # Not Found

##############################################################################################
"""Question3: Create a program to find the maximum element in a list using linear search."""


def find_max(lst):
    max = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max:
            max = lst[i]
    return max

# Test the function
print("The maximum number is:",find_max([1, 2, 3, 4, 5]))  # 5
print("The maximum number is:",find_max([-20,-4,-9]))  # -4

##############################################################################################
"""Question4: Implement binary search iteratively in Python."""

def binary_search(lst, target):
    left, right = 0, len(lst) - 1
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return "Not Found"

# Test the function
print(binary_search([1, 2, 3, 4, 5], 3))  # 2
print(binary_search([1, 2, 3, 4, 5], 6))  # Not Found

##############################################################################################
"""Question5: Implement binary search recursively in Python."""

def binary_search_recursive(lst, target, left, right):
    if left > right:
        return "Not Found"
    mid = (left + right) // 2
    if lst[mid] == target:
        return mid
    elif lst[mid] < target:
        return binary_search_recursive(lst, target, mid + 1, right)
    else:
        return binary_search_recursive(lst, target, left, mid - 1)

# Test the function
print(binary_search_recursive([1, 2, 3, 4, 5], 3, 0, 4))  # 2
print(binary_search_recursive([1, 2, 3, 4, 5], 6, 0, 4))  # Not Found

##############################################################################################
"""Question6: Write a program to check if a list is sorted in ascending order using binary search."""

def is_sorted(lst):
    left, right = 0, len(lst) - 1
    while left < right:
        mid = (left + right) // 2
        if lst[mid] > lst[mid + 1]:
            return "Not Sorted"
        if lst[mid] > lst[left]:
            left = mid
        else:
            right = mid
    return "Sorted"

# Test the function
print(is_sorted([1, 2, 3, 4, 5]))  # Sorted
print(is_sorted([1, 2, 3, 5, 4]))  # Not Sorted

##############################################################################################
"""Question7: Write a Python program to calculate the factorial of a number using a for loop."""

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Test the function
print("The factorial is:",factorial(5))  # 120
print("The factorial is:",factorial(0))  # 1

##############################################################################################
"""Question8: Create a program to generate the Fibonacci sequence up to a certain number of terms using a for loop."""

def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib

# Test the function
print("The Fibonacci sequence is:",fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

##############################################################################################
"""Question9: Write a function to compute the sum of all elements in a list using a for loop."""

def sum_elements(lst):
    sum = 0
    for i in lst:
        sum += i
    return sum

# Test the function
print("The sum of elements is:",sum_elements([1, 2, 3, 4, 5]))  # 15
print("The sum of elements is:",sum_elements([-1, -2, -3, -4, -5, 4]))  # -11

##############################################################################################
"""Question10: Write a Python function to check if a number is prime."""

def is_prime(n):
    if n < 2:
        return "Not Prime"
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return "Not Prime"
    return "Prime"

# Test the function
print("The number is:",is_prime(5))  # Prime
print("The number is:",is_prime(10))  # Not Prime

##############################################################################################
"""Question11: Create a function to reverse a string using recursion."""

def reverse_string(s):
    if len(s) == 0:
        return s
    return reverse_string(s[1:]) + s[0]

# Test the function
print("The reversed string is:",reverse_string("hello"))  # olleh
print("The reversed string is:",reverse_string("python"))  # nohtyp

##############################################################################################
"""Question12: Write a program that takes a list of integers and returns the largest number using a function."""

def largest_number(lst):
    max = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max:
            max = lst[i]
    return max

# Test the function
print("The largest number is:",largest_number([1, 2, 3, 4, 5]))  # 5
print("The largest number is:",largest_number([-20,-4,-9]))  # -4

##############################################################################################
"""Question13: Implement selection sort using a function and demonstrate its usage."""

def selection_sort(lst):
    for i in range(len(lst)):
        min_index = i
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[min_index]:
                min_index = j
        lst[i], lst[min_index] = lst[min_index], lst[i]
    return lst

# Test the function
print("The sorted list is:",selection_sort([64, 25, 12, 22, 11]))  # [11, 12, 22, 25, 64]
print("The sorted list is:",selection_sort([-5, -2, -8, -7, -1]))  # [-8, -7, -5, -2, -1]

##############################################################################################
"""Question14: Create a program that uses a function to find the intersection of two lists."""

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2)) if set(lst1) & set(lst2) else "No Intersection"

# Test the function
print("The intersection is:",intersection([1, 2, 3, 4, 5], [4, 5, 6, 7, 8]))  # [4, 5]
print("The intersection is:",intersection([1, 2, 3], [4, 5, 6]))  # No Intersection