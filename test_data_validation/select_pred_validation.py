import pandas as pd
import ast

# df_comb = pd.read_csv("test_gpt_predictions_encoded.csv")
df_comb = pd.read_csv("../test_aug_ai_pred_reorder_newgolds_updated.csv")

# Convert the string representation to actual list
# df_comb['pred_medroberta_18cats'] = df_comb['pred_medroberta_18cats'].apply(ast.literal_eval)
df_comb['labels_18'] = df_comb['labels_18'].apply(ast.literal_eval)

# validation condition
def needs_validation(row):
    pred = row['pred_gpt_18cats']
    return any(pred[i] == 1 for i in range(9, 17))
# def needs_validation(row):
#     pred = row['labels_18']
#     return any(pred[i] == 1 for i in range(17, 18))

df_validation = df_comb[df_comb.apply(needs_validation, axis=1)].copy()

print(f"Selected {len(df_validation)} sentences for medical validation.")

# df_validation.to_csv('test_gpt_predictions_encoded_for_medical_validation.csv', index=False)
df_validation.to_csv('test_aug_ai_none_for_validation_labels.csv', index=False)
