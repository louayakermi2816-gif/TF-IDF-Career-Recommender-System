import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # ============================================
    #   Welcome Message
    # ============================================
    print("============================================")
    print("   Welcome to the Tech Stack Recommender")
    print("============================================\n")

    # ============================================
    # Task 6: Data Pipeline Foundation
    # ============================================

    # Load job roles from external CSV (54 roles!)
    df = pd.read_csv("job_roles_expanded.csv")

    # Data is loaded and ready.

    # ============================================
    # Task 7: Vectorization with sklearn
    # ============================================

    # Step 1: Extract the skills column as a plain Python list
    skills_list = df["skills"].tolist()

    # Step 2: Create a TfidfVectorizer instance
    vectorizer = TfidfVectorizer()

    # Step 3: Fit and transform in one step
    job_vectors = vectorizer.fit_transform(skills_list)

    # Step 4: Vectorization complete

    # ============================================
    # Task 8-11: User Input, Scoring & Results
    # (wrapped in a loop so the user can try again)
    # ============================================

    while True:
        # Step 1: Ask the user for their skills
        raw = input("\nEnter your skills (use underscores for multi-word skills,\ne.g. python, deep_learning, sql): ")

        # --- Input Validation ---
        # Check if the user typed nothing or only spaces/commas
        cleaned_check = raw.replace(",", " ").strip()
        if not cleaned_check:
            print("\n[!] You didn't enter any skills. Please type at least 3 skills.")
            continue

        # Step 2: Clean the input
        # 2a - lowercase everything
        step1 = raw.lower()
        # 2b - replace commas with spaces
        step2 = step1.replace(",", " ")
        # 2c - collapse extra whitespace
        cleaned_input = " ".join(step2.split())

        # Step 3: Transform using the ALREADY fitted vectorizer
        user_vector = vectorizer.transform([cleaned_input])

        # --- Zero Vector Detection ---
        # If none of the user's skills exist in the vocabulary,
        # the vector will be all zeros and cosine similarity = 0 for everything
        if user_vector.nnz == 0:
            print("\n[!] None of your skills matched our database.")
            print("    Make sure you're using underscores for multi-word skills")
            print("    (e.g. machine_learning, not machine learning).")
            continue

        # Step 4: Compute cosine similarity
        scores = cosine_similarity(user_vector, job_vectors)
        scores = scores.flatten()

        # Step 5: Pair scores with job role names
        scores_dict = dict(zip(df["job_role"], scores))

        # Step 6: Sort by score, highest first
        ranked_results = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

        # Step 7: Slice and display the top 3
        top_3 = ranked_results[:3]

        print("\n--- Top 3 Recommendations ---")
        medals = ["[1st]", "[2nd]", "[3rd]"]
        for i, (job, score) in enumerate(top_3):
            print(f"{medals[i]} {job:<25} -> {score:.3f}")

        # --- Ask to try again ---
        again = input("\nWould you like to search again? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\nThank you for using the Tech Stack Recommender!")
            break


if __name__ == "__main__":
    main()
