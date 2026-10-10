"""Fixed prompt set: 4 natural-language + 4 code prompts (deterministic, offline)."""

PROMPTS = [
    {"id": "nl1", "category": "natural_language",
     "text": "Explain in one paragraph why the sky appears blue during the day."},
    {"id": "nl2", "category": "natural_language",
     "text": "Summarize the following in one sentence: Artificial neural networks are computing systems inspired by biological brains."},
    {"id": "nl3", "category": "natural_language",
     "text": "What is photosynthesis? Answer in two sentences for a high-school student."},
    {"id": "nl4", "category": "natural_language",
     "text": "List three causes of the seasons on Earth and briefly explain each."},
    {"id": "code1", "category": "code",
     "text": "Write a Python function to compute the factorial of n:\ndef factorial(n):"},
    {"id": "code2", "category": "code",
     "text": "Write a Python function to check if a string is a palindrome:\ndef is_palindrome(s):"},
    {"id": "code3", "category": "code",
     "text": "Write a Python function that returns the Fibonacci sequence up to n terms:\ndef fibonacci(n):"},
    {"id": "code4", "category": "code",
     "text": "Write a Python function to sort a list using bubble sort:\ndef bubble_sort(arr):"},
]
