from evaluation import evaluate_model

test_titles = [
    "Batman",
    "Superman",
    "The Avengers",
    "Spider-Man",
    "Iron Man"
]

print("=== BASELINE ===")
print(evaluate_model("baseline", test_titles))

print("\n=== HYBRID ===")
print(evaluate_model("hybrid", test_titles))
