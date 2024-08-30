"""Problem 1: Set Operations"""
# Define two sets, A and B, with at least 5 elements each

A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}


# Write a function to find the union of sets A and B.

def union(A, B):
    return A.union(B)

#### Test the function ####
print(union(A, B))  # {1, 2, 3, 4, 5, 6, 7, 8}


# Write a function to find the intersection of sets A and B.

def intersection(A, B):
    return A.intersection(B)

#### Test the function ####
print(intersection(A, B))  # {4, 5}


# Write a function to find the difference between sets A and B.

def difference(A, B):
    return A.difference(B)

#### Test the function ####
print(difference(A, B))  # {1, 2, 3}


# Write a function to check if set A is a subset of set B.

def is_subset(A, B):
    return A.issubset(B)

##ANOHER WAY##
def issubset(A, B):
    return A <= B

#### Test the function ####
print(is_subset(A, B))  # False
print(issubset(A, B))  # False


# Write a function to find the symmetric difference between sets A and B.

def symmetric_difference(A, B):
    return A.symmetric_difference(B)

##ANOHER WAY##
def symmetricdifference(A, B):
    return A ^ B

#### Test the function ####
print(symmetric_difference(A, B))  # {1, 2, 3, 6, 7, 8}
print(symmetricdifference(A, B))  # {1, 2, 3, 6, 7, 8}


# Write a function to remove duplicates from a list and convert it into a set.

def remove_duplicates(lst):
    return set(lst)

#### Test the function ####
print(remove_duplicates([1, 2, 3, 3, 4, 5, 5]))  # {1, 2, 3, 4, 5}


# Write a function to determine if two sets have any elements in common.

def has_common_element(A, B):
    return not A.isdisjoint(B)

#### Test the function ####
print(has_common_element(A, B))  # True


# Write a function to check if a set is empty.

def is_empty(A):
    return not bool(A)

#### Test the function ####
print(is_empty(set()))  # True
print(is_empty({1, 2, 3}))  # False


# Write a function to remove a specific element from a set.

def remove_element(A, element):
    A.discard(element)
    return A

#### Test the function ####
print(remove_element(A, 3))  # {1, 2, 4, 5}


# Write a function to check if two sets are disjoint (have no elements in common).

def is_disjoint(A, B):
    return A.isdisjoint(B)

#### Test the function ####
print(is_disjoint(A, B))  # False


# Write a function to find the Cartesian product of two sets.

def cartesian_product(A, B):
    return {(a, b) for a in A for b in B}

#### Test the function ####
print(cartesian_product({1, 2}, {'a', 'b'}))  # {(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')}


# Write a function to find the power set of a set.

def power_set(A):
    from itertools import chain, combinations
    return set(chain.from_iterable(combinations(A, r) for r in range(len(A)+1)))

#### Test the function ####
print(power_set({1, 2, 3}))  # {(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)}


# Write a function to find the maximum and minimum elements of a set.

def max_min_element(A):
    return max(A), min(A)

#### Test the function ####
print(max_min_element({1, 2, 3, 4, 5}))  # (5, 1)


# Write a function to calculate the Jaccard similarity coefficient between two sets.

def jaccard_similarity(A, B):
    return len(A.intersection(B)) / len(A.union(B))

#### Test the function ####
print(jaccard_similarity({1, 2, 3}, {2, 3, 4}))  # 0.5

##############################################################################################
"""Problem 2: Dictionary Manipulation"""
# Create a dictionary representing the population of fifteen countries.

population = {
    'China': 1439323776,
    'India': 1380004385,
    'United States': 331002651,
    'Indonesia': 273523615,
    'Pakistan': 220892340,
    'Brazil': 212559417,
    'Nigeria': 206139589,
    'Bangladesh': 164689383,
    'Russia': 145934462,
    'Mexico': 128932753,
    'Japan': 126476461,
    'Ethiopia': 114963588,
    'Philippines': 109581078,
    'Egypt': 102334404,
    'Vietnam': 97338579
}


# Write a function to find the country with the smallest population.

def smallest_population(population):
    return min(population, key=population.get)

#### Test the function ####
print(smallest_population(population))  # Vietnam


# Write a function to calculate the total population.

def total_population(population):
    return sum(population.values())

#### Test the function ####
print(total_population(population))  # 5053696481


# Write a function to find the top 5 most populous countries.

def top_5_countries(population):
    return dict(sorted(population.items(), key=lambda x: x[1], reverse=True)[:5])

#### Test the function ####
print(top_5_countries(population))  # {'China': 1439323776, 'India': 1380004385, 'United States': 331002651, 'Indonesia': 273523615, 'Pakistan': 220892340}


# Write a function to add a new country and its population to the dictionary.

def add_country(population, country, population_count):
    population[country] = population_count
    return population

#### Test the function ####
print(add_country(population, 'Germany', 83783942))  # {'China': 1439323776, 'India': 1380004385, 'United States': 331002651, 'Indonesia': 273523615, 'Pakistan': 220892340, 'Brazil': 212559417, 'Nigeria': 206139589, 'Bangladesh': 164689383, 'Russia': 145934462, 'Mexico': 128932753, 'Japan': 126476461, 'Ethiopia': 114963588, 'Philippines': 109581078, 'Egypt': 102334404, 'Vietnam': 97338579, 'Germany': 83783942}


# Write a function to remove a country from the dictionary.

def remove_country(population, country):
    population.pop(country, None)
    return population

#### Test the function ####
print(remove_country(population, 'Japan'))  # {'China': 1439323776, 'India': 1380004385, 'United States': 331002651, 'Indonesia': 273523615, 'Pakistan': 220892340, 'Brazil': 212559417, 'Nigeria': 206139589, 'Bangladesh': 164689383, 'Russia': 145934462, 'Mexico': 128932753, 'Ethiopia': 114963588, 'Philippines': 109581078, 'Egypt': 102334404, 'Vietnam': 97338579}
print(remove_country(population, 'England'))  # {'China': 1439323776, 'India': 1380004385, 'United States': 331002651, 'Indonesia': 273523615, 'Pakistan': 220892340, 'Brazil': 212559417, 'Nigeria': 206139589, 'Bangladesh': 164689383, 'Russia': 145934462, 'Mexico': 128932753, 'Japan': 126476461, 'Ethiopia': 114963588, 'Philippines': 109581078, 'Egypt': 102334404, 'Vietnam': 97338579}

# Write a function to update the population of a specific country.

def update_population(population, country, new_population):
    population[country] = new_population
    return population

#### Test the function ####
print(update_population(population, 'China', 2000000000))  # {'China': 2000000000, 'India': 1380004385, 'United States': 331002651, 'Indonesia': 273523615, 'Pakistan': 220892340, 'Brazil': 212559417, 'Nigeria': 206139589, 'Bangladesh': 164689383, 'Russia': 145934462, 'Mexico': 128932753, 'Japan': 126476461, 'Ethiopia': 114963588, 'Philippines': 109581078, 'Egypt': 102334404, 'Vietnam': 97338579}


# Write a function to check if a country exists in the dictionary.

def country_exists(population, country):
    return country in population

#### Test the function ####
print(country_exists(population, 'India'))  # True
print(country_exists(population, 'Germany'))  # False


# Write a function to get the population of a specific country.

def get_population(population, country):
    return population.get(country)

#### Test the function ####
print(get_population(population, 'India'))  # 1380004385
print(get_population(population, 'England'))  # None


# Write a function to clear all entries from the dictionary.

def clear_population(population):
    population.clear()
    return population

#### Test the function ####
print(clear_population(population))  # {}


# Write a function to merge two dictionaries.

def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}

#### Test the function ####
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
print(merge_dicts(dict1, dict2))  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}


# Write a function to sort a dictionary by its values.

def sort_by_values(d):
    return dict(sorted(d.items(), key=lambda x: x[1]))

#### Test the function ####
d = {'a': 3, 'b': 1, 'c': 2}
print(sort_by_values(d))  # {'b': 1, 'c': 2, 'a': 3}


# Write a function to check if a dictionary is empty.

def is_dict_empty(d):
    return not bool(d)

#### Test the function ####
print(is_dict_empty({}))  # True
print(is_dict_empty({'a': 1}))  # False


# Write a function to find the country with the highest population density.

def highest_population_density(population, land_area):
    pop_density = {country: population[country] / land_area[country]
                   for country in population if country in land_area and population[country] != 0 and land_area[country] != 0}
    if not pop_density:
        return None
    return max(pop_density, key=pop_density.get)

#### Test the function ####
population = {
    'China': 1439323776,
    'India': 1380004385,
    'United States': 331002651,
    'Indonesia': 273523615,
    'Pakistan': 220892340,
    'Brazil': 212559417,
    'Nigeria': 206139589,
    'Bangladesh': 164689383,
    'Russia': 145934462,
    'Mexico': 128932753,
    'Japan': 126476461,
    'Ethiopia': 114963588,
    'Philippines': 109581078,
    'Egypt': 102334404,
    'Vietnam': 97338579
}

land_area = {
    'China': 9640011,
    'India': 3287263,
    'United States': 9629091,
    'Indonesia': 1904569,
    'Pakistan': 881913,
    'Brazil': 8515767,
    'Nigeria': 923768,
    'Bangladesh': 147570,
    'Russia': 17098242,
    'Mexico': 1964375,
    'Japan': 377975,
    'Ethiopia': 1104300,
    'Philippines': 300000,
    'Egypt': 1002450,
    'Vietnam': 331212
}
print(highest_population_density(population, land_area))  # Expected output: 'Bangladesh'


# Write a function to create a dictionary from two lists, one containing keys and the other containing values.

def create_dict(keys, values):
    return dict(zip(keys, values))

#### Test the function ####
keys = ['a', 'b', 'c']
values = [1, 2, 3]
print(create_dict(keys, values))  # {'a': 1, 'b': 2, 'c': 3}

##############################################################################################
"""Problem 3: Tuple Operations"""
# Create a tuple containing the names of 10 programming languages.

languages = ('Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'PHP', 'Swift', 'Go', 'Rust', 'Kotlin')


# Write a function to reverse the elements of a tuple.

def reverse_tuple(t):
    return t[::-1]

#### Test the function ####
print(reverse_tuple(languages))  # ('Kotlin', 'Rust', 'Go', 'Swift', 'PHP', 'Ruby', 'JavaScript', 'C++', 'Java', 'Python')


# Write a function to find the index of a specific element in a tuple.

def find_index(t, element):
    return t.index(element)

#### Test the function ####
print(find_index(languages, 'Python'))  # 0


# Write a function to convert a tuple to a list.

def tuple_to_list(t):
    return list(t)

#### Test the function ####
print(tuple_to_list(languages))  # ['Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'PHP', 'Swift', 'Go', 'Rust', 'Kotlin']


# Write a function to count the occurrences of a specific element in a tuple.

def count_occurrences(t, element):
    return t.count(element)

#### Test the function ####
print(count_occurrences(languages, 'Python'))  # 1
print(count_occurrences(languages, 'C'))  # 0


# Write a function to find the length of a tuple.

def tuple_length(t):
    return len(t)

#### Test the function ####
print(tuple_length(languages))  # 10


# Write a function to check if two tuples are equal.

def are_tuples_equal(t1, t2):
    return t1 == t2

#### Test the function ####
print(are_tuples_equal(languages, ('Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'PHP', 'Swift', 'Go', 'Rust', 'Kotlin')))  # True
print(are_tuples_equal(languages, ('Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'PHP', 'Swift', 'Go', 'Rust', 'Kotlin', 'Scala')))  # False


# Write a function to slice a tuple.

def slice_tuple(t, start, end):
    return t[start:end]

#### Test the function ####
print(slice_tuple(languages, 2, 5))  # ('C++', 'JavaScript', 'Ruby')


# Write a function to convert a tuple to a string.

def tuple_to_string(t):
    return ' '.join(t)

#### Test the function ####
print(tuple_to_string(languages))  # Python Java C++ JavaScript Ruby PHP Swift Go Rust Kotlin


# Write a function to sort a tuple of tuples by the second element.

def sort_by_second_element(t):
    return tuple(sorted(t, key=lambda x: x[1]))

#### Test the function ####
t = ((1, 3), (3, 2), (2, 1))
print(sort_by_second_element(t))  # ((2, 1), (3, 2), (1, 3))


# Write a function to zip two tuples.

def zip_tuples(t1, t2):
    return tuple(zip(t1, t2))

#### Test the function ####
t1 = (1, 2, 3)
t2 = ('a', 'b', 'c')
print(zip_tuples(t1, t2))  # ((1, 'a'), (2, 'b'), (3, 'c'))


# Write a function to convert a tuple of strings to a single string.

def tuple_of_strings_to_string(t):
    return ''.join(t)

#### Test the function ####
t = ('Hello', 'World')
print(tuple_of_strings_to_string(t))  # HelloWorld


# Write a function to find the average value of a tuple of numbers.

def average_tuple(t):
    return sum(t) / len(t)

#### Test the function ####
t = (1, 2, 3, 4, 5)
print(average_tuple(t))  # 3.0


# Write a function to flatten a tuple of tuples into a single tuple.

def flatten_tuple(t):
    return tuple(x for sub in t for x in sub)

#### Test the function ####
t = ((1, 2), (3, 4), (5, 6))
print(flatten_tuple(t))  # (1, 2, 3, 4, 5, 6)


# Write a function to find the index of the first occurrence of a tuple within another tuple.

def find_tuple(t1, t2):
    t1_str = ' '.join(map(str, t1))
    t2_str = ' '.join(map(str, t2))
    pos = t1_str.find(t2_str)
    return pos // 2 if pos != -1 else -1

#### Test the function ####
t1 = (1, 2, 3, 4, 5)
t2 = (3, 4)
print(find_tuple(t1, t2))  # 2
t2 = (6, 7)
print(find_tuple(t1, t2))  # -1



