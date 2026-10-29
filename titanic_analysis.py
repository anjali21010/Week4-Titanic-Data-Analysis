
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

print("=" * 50)
print("TITANIC DATA ANALYSIS PROJECT")
print("=" * 50)

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 1. Display dataset information
print("\n1. DATASET OVERVIEW")
print("Total passengers in training data:", len(train))
print("\nFirst 5 rows:")
print(train.head())

# 2. Check missing values
print("\n2. MISSING VALUES")
print(train.isnull().sum())

# 3. Survival analysis
print("\n3. SURVIVAL ANALYSIS")
survival_counts = train["Survived"].value_counts().sort_index()
print("Did not survive:", survival_counts.get(0, 0))
print("Survived:", survival_counts.get(1, 0))
print("Survival percentage:", round(train["Survived"].mean() * 100, 2), "%")

# 4. Survival by gender
print("\n4. SURVIVAL BY GENDER")
gender_survival = train.groupby("Sex")["Survived"].mean() * 100
print(gender_survival.round(2))

# 5. Survival by passenger class
print("\n5. SURVIVAL BY PASSENGER CLASS")
class_survival = train.groupby("Pclass")["Survived"].mean() * 100
print(class_survival.round(2))

# 6. Survival by age group
print("\n6. SURVIVAL BY AGE GROUP")
train["AgeGroup"] = pd.cut(
    train["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teenager", "Young Adult", "Adult", "Senior"]
)
age_survival = train.groupby("AgeGroup", observed=False)["Survived"].mean() * 100
print(age_survival.round(2))

# 7. Graph 1: Survival counts
plt.figure(figsize=(7, 5))
survival_counts.plot(kind="bar", rot=0)
plt.title("Titanic Passenger Survival Counts")
plt.xlabel("Survival (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("outputs/survival_counts.png")
plt.close()

# 8. Graph 2: Survival by gender
plt.figure(figsize=(7, 5))
gender_survival.plot(kind="bar", rot=0)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()
plt.savefig("outputs/survival_by_gender.png")
plt.close()

# 9. Graph 3: Survival by passenger class
plt.figure(figsize=(7, 5))
class_survival.plot(kind="bar", rot=0)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()
plt.savefig("outputs/survival_by_class.png")
plt.close()

# 10. Save analysis results
results = pd.DataFrame({
    "Analysis": [
        "Total training passengers",
        "Survival percentage",
        "Female survival rate (%)",
        "Male survival rate (%)",
        "Class 1 survival rate (%)",
        "Class 2 survival rate (%)",
        "Class 3 survival rate (%)"
    ],
    "Result": [
        len(train),
        round(train["Survived"].mean() * 100, 2),
        round(gender_survival.get("female", 0), 2),
        round(gender_survival.get("male", 0), 2),
        round(class_survival.get(1, 0), 2),
        round(class_survival.get(2, 0), 2),
        round(class_survival.get(3, 0), 2)
    ]
})
results.to_csv("outputs/analysis_results.csv", index=False)

# 11. Save a short report
with open("outputs/analysis_report.txt", "w", encoding="utf-8") as report:
    report.write("TITANIC DATA ANALYSIS REPORT\n")
    report.write("=" * 35 + "\n\n")
    report.write(f"Training passengers: {len(train)}\n")
    report.write(f"Overall survival rate: {train['Survived'].mean() * 100:.2f}%\n\n")
    report.write("Survival rate by gender (%):\n")
    report.write(gender_survival.round(2).to_string() + "\n\n")
    report.write("Survival rate by passenger class (%):\n")
    report.write(class_survival.round(2).to_string() + "\n\n")
    report.write("Note: These are associations in the dataset, not proof of causation.\n")

print("\nPROJECT ANALYSIS COMPLETED SUCCESSFULLY!")
print("Check the outputs folder for graphs and report.")
