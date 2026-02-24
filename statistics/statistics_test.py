import pandas as pd
import ast
import matplotlib.pyplot as plt

# Decoded order in labels_10 (index positions):
LABELS_ORDER = [
    "B440 Respiration functions",
    "B140 Attention functions",
    "D840-D859 Work and employment",
    "B1300 Energy level",
    "D550 Eating",
    "D450 Walking",
    "B455 Exercise tolerance functions",
    "B530 Weight maintenance functions",
    "B152 Emotional functions",
    "None"
]


DISPLAY_ORDER = [
    "B1300 Energy level",
    "B140 Attention functions",
    "B152 Emotional functions",
    "B440 Respiration functions",
    "B455 Exercise tolerance functions",
    "B530 Weight maintenance functions",
    "D450 Walking",
    "D550 Eating",
    "D840-D859 Work and employment"
]


df = pd.read_csv('combined_test_new_INS_fixed_FP.csv')

# Convert labels_10 from string to list
df['labels_10_list'] = df['labels_10'].apply(lambda x: ast.literal_eval(x))

# Total sentences
total_sentences = len(df)
print(f"Total sentences: {total_sentences}\n")

# Build a mapping: category name - index position in labels_10
label_index_map = {cat: idx for idx, cat in enumerate(LABELS_ORDER)}

# Count and percentage
category_counts = {}
for cat in DISPLAY_ORDER:
    idx = label_index_map[cat]
    count = df['labels_10_list'].apply(lambda x: x[idx] == 1).sum()
    percentage = count / total_sentences * 100
    category_counts[cat] = (count, percentage)

# Display stats
print("Category Counts and Percentages (Custom Order):\n")
for cat in DISPLAY_ORDER:
    count, percentage = category_counts[cat]
    print(f"{cat:<50} | {count:>5} sentences | {percentage:5.2f}%")

# Save statistics to CSV
stats_df = pd.DataFrame([
    {"Category": cat, "Count": count, "Percentage": percentage}
    for cat, (count, percentage) in category_counts.items()
])
stats_df.to_csv('test_data_category_stats.csv', index=False)
print("\n Statistics saved to test_data_category_stats.csv")

# Plot histogram
plt.figure(figsize=(10, 5))
plt.bar(stats_df['Category'], stats_df['Count'])
plt.xticks(rotation=90)
plt.ylabel('Number of Sentences')
plt.title('Category Distribution in Test Set (Custom Order)')
plt.tight_layout()
plt.savefig('test_data_category_distribution_custom_order.png')
print("Histogram saved as test_data_category_distribution_custom_order.png")
plt.show()
