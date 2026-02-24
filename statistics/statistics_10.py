import pandas as pd
import ast
import matplotlib.pyplot as plt


INPUT_CSV = 'combined_test_new_INS_fixed_FP.csv'
OUT_CSV   = 'test_stats_with_notes.csv'
OUT_SENT_PNG = 'test_category_distribution_sentences.png'
OUT_NOTE_PNG = 'test_category_distribution_notes.png'

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
    "D840-D859 Work and employment",
    "None"
]


df = pd.read_csv(INPUT_CSV)

# Ensure NoteID is a clean string (so 104003.0 -> "104003")
# if 'NoteID' not in df.columns:
#     raise KeyError("Expected a 'NoteID' column in the file.")
# df['NoteID'] = df['NoteID'].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
if 'NotitieID' not in df.columns:
    raise KeyError("Expected a 'NotitieID' column in the file.")
df['NotitieID'] = df['NotitieID'].apply(lambda x: str(int(x)) if pd.notnull(x) else "")

# Convert labels_10 from string to list (length 10)
df['labels_10_list'] = df['labels_10'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [0]*10)


total_sentences = len(df)
total_notes = df['NotitieID'].nunique()
print(f"Total sentences: {total_sentences}")
print(f"Total notes:     {total_notes}\n")

# Build a matrix of the 10 labels as columns
labels_matrix = pd.DataFrame(df['labels_10_list'].tolist(), columns=LABELS_ORDER)

# Attach NoteID to compute note-level presence (a note counts for a category if ANY sentence in that note has it)
labels_with_note = pd.concat([df[['NotitieID']], labels_matrix], axis=1)
notes_max = labels_with_note.groupby('NotitieID', as_index=False)[LABELS_ORDER].max()

# Sentence-level counts
sent_counts = labels_matrix.sum(axis=0).astype(int)
sent_pct = (sent_counts / total_sentences * 100).round(2)

# Note-level counts
note_counts = notes_max[LABELS_ORDER].sum(axis=0).astype(int)
note_pct = (note_counts / total_notes * 100).round(2)

# Build tidy table in your DISPLAY_ORDER (exclude 'None')
rows = []
for cat in DISPLAY_ORDER:
    rows.append({
        "Category": cat,
        "Sentence Count": sent_counts[cat],
        "Sentence %": sent_pct[cat],
        "Note Count": note_counts[cat],
        "Note %": note_pct[cat],
    })
stats_df = pd.DataFrame(rows)

# Append totals at the bottom
totals_rows = pd.DataFrame([
    {"Category": "TOTAL_SENTENCES", "Sentence Count": total_sentences, "Sentence %": 100.0,
     "Note Count": "", "Note %": ""},
    {"Category": "TOTAL_NOTES", "Sentence Count": "", "Sentence %": "",
     "Note Count": total_notes, "Note %": 100.0}
])
stats_df = pd.concat([stats_df, totals_rows], ignore_index=True)


stats_df.to_csv(OUT_CSV, index=False)
print(f"Statistics (with note counts/%) saved to {OUT_CSV}")

# Plots (exclude the 2 total rows)
plot_df = stats_df.iloc[:-2]

# Sentence-level plot
plt.figure(figsize=(11, 5))
plt.bar(plot_df['Category'], plot_df['Sentence Count'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Number of Sentences')
plt.title('Category Distribution (Sentence Level)')
plt.tight_layout()
plt.savefig(OUT_SENT_PNG)
print(f"Saved: {OUT_SENT_PNG}")
plt.show()

# Note-level plot
plt.figure(figsize=(11, 5))
plt.bar(plot_df['Category'], plot_df['Note Count'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Number of Notes')
plt.title('Category Distribution (Note Level)')
plt.tight_layout()
plt.savefig(OUT_NOTE_PNG)
print(f"Saved: {OUT_NOTE_PNG}")
plt.show()
