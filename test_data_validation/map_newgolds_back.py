import pandas as pd

# df_test = pd.read_csv('test_aug_ai_pred_reorder.csv')
# df_new_gold = pd.read_csv('medroberta_validated_with_gold18_final.csv')
# df_test = pd.read_csv('test_aug_ai_pred_reorder_newgolds.csv')
# df_new_gold = pd.read_csv('gpt_unique_pred_with_gold_18_encoded.csv')
# df_test = pd.read_csv('test_gpt_predictions_encoded_reorder.csv')
df_test = pd.read_csv('test_gpt_predictions_encoded_reorder_mednewgolds.csv')
df_new_gold = pd.read_csv('gpt_unique_pred_with_gold_18_encoded.csv')

# define key columns for matching
key_cols = ['pad_sen_id', 'NotitieID', 'text_raw']

# merge gold labels into test set
df_updated = df_test.merge(
    df_new_gold[key_cols + ['gold_18_labels']],
    on=key_cols,
    how='left',
    suffixes=('', '_new_gold')
)

# replace labels_18 if new gold label exists
df_updated['labels_18'] = df_updated.apply(
    lambda row: row['gold_18_labels'] if pd.notna(row['gold_18_labels']) else row['labels_18'],
    axis=1
)

# drop helper column after replacement
df_updated = df_updated.drop(columns=['gold_18_labels'])


# df_updated.to_csv('test_aug_ai_pred_reorder_newgolds.csv', index=False)
# df_updated.to_csv('test_aug_ai_pred_reorder_newgolds_updated.csv', index=False)
df_updated.to_csv('test_gpt_predictions_encoded_reorder_newgolds_updated.csv', index=False)

print("Updated 'labels_18' in test_set_updated.csv")
