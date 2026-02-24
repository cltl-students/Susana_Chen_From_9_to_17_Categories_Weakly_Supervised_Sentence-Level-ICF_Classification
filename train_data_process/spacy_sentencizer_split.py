import pandas as pd
import csv
import spacy

nlp = spacy.load("nl_core_news_lg", disable=["tagger", "parser", "ner"])

if not nlp.has_pipe("sentencizer"):
    nlp.add_pipe("sentencizer")

input_csv = "newcats_notities_2023_shuf.csv"
output_csv = "newcats_sentences_2023.csv"
sep = ";"

df = pd.read_csv(input_csv, sep=sep, header=None, engine='python', quoting=csv.QUOTE_MINIMAL, encoding='utf-8', on_bad_lines='warn')

note_col = 8 # column with the full note
note_id_col = 1 # columnn with the note indexes

with open(output_csv, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["note_id", "sentence_index", "sentence"])

    for row_idx, row in df.iterrows():
        note_id = str(row[note_id_col])
        raw_note = str(row[note_col]) if not pd.isna(row[note_col]) else ""
        note_text = raw_note.replace("\\n", " ").strip("[]")

        doc = nlp(note_text)

        for sentence_idx, sent in enumerate(doc.sents, start=1):
            text = sent.text.strip()
            writer.writerow([note_id, sentence_idx, text])

print("Finished sentence splitting and assigned note ids and sentence ids.")
