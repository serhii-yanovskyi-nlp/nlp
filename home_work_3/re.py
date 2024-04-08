import re


# Write a Python program that matches a word containing 'z', not the start or end of the word

def check_z(text):
    pat = r'\b\w*z\w*\b'
    matches = re.findall(pat, text)
    return matches


check_z("Zebras are grazing in the zoo, while listening to jazz music.")


# Write a Python program to remove leading zeros from an IP address

def remove_leading_zeros(ip_address):
    pattern = r'\b0+(\d+)\b'
    result = re.sub(pattern, r'\1', ip_address)
    return result


remove_leading_zeros("192.008.020.006")


# Write a Python program to find the occurrence and position of substrings within a string

def find_substring(string, substring):
    pattern = re.compile(f'({re.escape(substring)})')
    matches = pattern.finditer(string)
    result = []
    for match in matches:
        start = match.start(1)
        end = match.end(1)
        result.append((substring, start, end))
    return result


results = find_substring("hello world, hello Bob, hello python", "hello")
for result in results:
    print(f"substring: {result[0]}, start: {result[1]}, end: {result[2]}")


# Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format

def convert_date(date):
    pattern = r'(\d{4})-(\d{2})-(\d{2})'
    converted = re.sub(pattern, r'\3-\2-\1', date)
    return converted


converted_date = convert_date("2022-04-10")
print(converted_date)


# Write a Python program to find all three, four, and five character words in a string

def find_words(string):
    pattern = r'\b\w{3,5}\b'
    words = re.findall(pattern, string)
    return words


words = find_words("This is a sample text with some words of various lengths like cat, dog, python, fish, and birds")
print(words)


# Write a Python program to convert a camel-case string to a snake-case string

def camel(string):
    words = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)
    snake = '_'.join(words).lower()
    return snake


camel = camel("ThisIsALongerCamelCaseStringWithMultipleWords")
print(camel)


# Write a Python program to find all adverbs and their positions in a given sentence

def find_adverbs(text):
    pattern = r'\b\w+ly\b'
    matches = re.finditer(pattern, text)
    adverbs = []
    for match in matches:
        adverb = match.group(0)
        pos = match.span()
        adverbs.append((adverb, pos))
    return adverbs


adverbs = find_adverbs("He quickly ran to the store and carefully picked up the package.")
for adverb, pos in adverbs:
    print(f"{adverb}: {pos}")


# Write a Python program to concatenate the consecutive numbers in a given string.

def concatenate(text):
    pattern = r'\b(\d+)\s+(\d+)\b'
    matches = re.finditer(pattern, text)
    result = text
    for match in matches:
        start, end = match.span()
        num1, num2 = match.groups()
        concat = num1 + num2
        result = result[:start] + concat + result[end:]
    return result


concat = concatenate(
    "Enter at 1 20 Kearny Street. The security desk can direct you to floor 1 6. Please have your identification ready.")

print(concat)
