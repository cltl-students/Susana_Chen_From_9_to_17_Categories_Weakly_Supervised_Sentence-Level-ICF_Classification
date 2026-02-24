import pandas as pd
import ast
import matplotlib.pyplot as plt

# Decoded order
LABELS_ORDER_18 = [
    "B1300 Energy level",
    "B140 Attention functions",
    "B152 Emotional functions",
    "B440 Respiration functions",
    "B455 Exercise tolerance functions",
    "B530 Weight maintenance functions",
    "D450 Walking",
    "D550 Eating",
    "D840-D859 Work and employment",
    "B280 Sensations of pain",
    "B134 Sleep functions",
    "D760 Family relationships",
    "B164 Higher-level cognitive functions",
    "D465 Moving around using equipment",
    "D410 Changing basic body position",
    "B230 Hearing functions",
    "D240 Handling stress and other psychological demands",
    "None"
]

DISPLAY_ORDER = [
    "B1300 Energy level","B140 Attention functions","B152 Emotional functions",
    "B440 Respiration functions","B455 Exercise tolerance functions",
    "B530 Weight maintenance functions","D450 Walking","D550 Eating",
    "D840-D859 Work and employment","B280 Sensations of pain","B134 Sleep functions",
    "D760 Family relationships","B164 Higher-level cognitive functions",
    "D465 Moving around using equipment","D410 Changing basic body position",
    "B230 Hearing functions","D240 Handling stress and other psychological demands"
]


# df = pd.read_csv('train_encoded_noteidcleaned.csv')
# df = pd.read_csv('ant_VUMC2023_temp0.1_def_fewshot_cleaned_filtered_encoded.csv')
# df = pd.read_csv('ant_VUMC2023_temp0.1_def_fewshot_cleaned_filtered_encoded.csv')
# df = pd.read_csv('train_jenia_murat_encoded.csv')
# df = pd.read_csv('train_AMC2023.csv')
# df = pd.read_csv('train_combined.csv')
# df = pd.read_csv('dev_aug_ai.csv')
df = pd.read_csv('train_aug_ai_shuffled.csv')
df = pd.read_csv('test_gpt_predictions_encoded_reorder_newgolds_updated.csv')

# Clean NoteID (remove trailing .0, keep as string)
# if 'NoteID' in df.columns:
#     df['NoteID'] = df['NoteID'].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
# else:
#     raise KeyError("Expected a 'NoteID' column in the file.")
if 'NotitieID' in df.columns:
    df['NotitieID'] = df['NotitieID'].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
else:
    raise KeyError("Expected a 'NotitieID' column in the file.")


df['labels_18_list'] = df['labels_18'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [0]*18)


total_sentences = len(df)
total_notes = df['NotitieID'].nunique()
print(f"Total sentences: {total_sentences}")
print(f"Total notes:     {total_notes}\n")

# Expand into a matrix aligned with LABELS_ORDER_18
labels_matrix = pd.DataFrame(df['labels_18_list'].tolist(), columns=LABELS_ORDER_18)
labels_with_note = pd.concat([df[['NotitieID']], labels_matrix], axis=1)

# Sentence-level counts
sent_counts = labels_matrix.sum(axis=0).astype(int)
sent_pct = (sent_counts / total_sentences * 100).round(2)

# Note-level counts
# A note counts for a category if ANY sentence in that note has that category
notes_max = labels_with_note.groupby('NotitieID', as_index=False)[LABELS_ORDER_18].max()
note_counts = notes_max[LABELS_ORDER_18].sum(axis=0).astype(int)
note_pct = (note_counts / total_notes * 100).round(2)

# Build stats table (include 'None' row now)
# First the 17 categories in your preferred order
rows = []
for cat in DISPLAY_ORDER:
    rows.append({
        "Category": cat,
        "Sentence Count": sent_counts[cat],
        "Sentence %": sent_pct[cat],
        "Note Count": note_counts[cat],
        "Note %": note_pct[cat],
    })

# Add 'None' row to the CSV (but we’ll exclude it from plots)
none_cat = "None"
rows.append({
    "Category": none_cat,
    "Sentence Count": sent_counts[none_cat],
    "Sentence %": sent_pct[none_cat],
    "Note Count": note_counts[none_cat],
    "Note %": note_pct[none_cat],
})

stats_df = pd.DataFrame(rows)

# Append totals so they also appear in the CSV
totals_rows = pd.DataFrame([
    {"Category": "TOTAL_SENTENCES", "Sentence Count": total_sentences, "Sentence %": 100.0,
     "Note Count": "", "Note %": ""},
    {"Category": "TOTAL_NOTES", "Sentence Count": "", "Sentence %": "",
     "Note Count": total_notes, "Note %": 100.0}
])
stats_df = pd.concat([stats_df, totals_rows], ignore_index=True)


print("Category counts by SENTENCE and by NOTE (including 'None'):\n")
print(stats_df.to_string(index=False))


out_csv = 'test_18_statistics_including_None.csv'
stats_df.to_csv(out_csv, index=False)
print(f"\nStatistics saved to {out_csv}")


# Exclude 'None' and the two total rows from plots
plot_df = stats_df[~stats_df['Category'].isin(['None','TOTAL_SENTENCES','TOTAL_NOTES'])]

# Sentences plot
plt.figure(figsize=(12, 6))
plt.bar(plot_df["Category"], plot_df["Sentence Count"].astype(float))
plt.xticks(rotation=90, ha='right')
plt.title("Sentence Count per Category (excluding 'None')")
plt.ylabel("Number of Sentences")
plt.tight_layout()
plt.savefig("test_18_sent_stat_excNone.png")
print("Saved: test_18_sent_stat_excNone.png")
plt.show()

# Notes plot
plt.figure(figsize=(12, 6))
plt.bar(plot_df["Category"], plot_df["Note Count"].astype(float))
plt.xticks(rotation=90, ha='right')
plt.title("Note Count per Category (excluding 'None')")
plt.ylabel("Number of Notes")
plt.tight_layout()
plt.savefig("test_18_note_stats_excNone.png")
print("Saved: test_18_note_stats_excNone.png")
plt.show()
