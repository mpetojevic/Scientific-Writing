import pandas as pd



chat_df = pd.read_csv('PredictAndExplain.csv')
chat_df['ChatGPT_Predicted_Label'] = chat_df['ChatGPT_Predicted_Label'].str.lower()

matches = chat_df['gold_Label'] == chat_df['ChatGPT_Predicted_Label']

percentage = (matches.sum() / len(chat_df)) * 100

print(f"Percentage of rows where the two columns are the same: {percentage}%")

correct_entailment = ((chat_df['gold_Label'] == 'entailment') & (chat_df['gold_Label'] == chat_df['ChatGPT_Predicted_Label'])).sum()
correct_contradiction = ((chat_df['gold_Label'] == 'contradiction') & (chat_df['gold_Label'] == chat_df['ChatGPT_Predicted_Label'])).sum()
correct_neutral = ((chat_df['gold_Label'] == 'neutral') & (chat_df['gold_Label'] == chat_df['ChatGPT_Predicted_Label'])).sum()

# Calculate the total number of instances for each category
total_entailment = (chat_df['gold_Label'] == 'entailment').sum()
total_contradiction = (chat_df['gold_Label'] == 'contradiction').sum()
total_neutral = (chat_df['gold_Label'] == 'neutral').sum()


# Calculate the accuracy for each category
accuracy_entailment = (correct_entailment / total_entailment) * 100 if total_entailment > 0 else 0
accuracy_contradiction = (correct_contradiction / total_contradiction) * 100 if total_contradiction > 0 else 0
accuracy_neutral = (correct_neutral / total_neutral) * 100 if total_neutral > 0 else 0

print(f"Accuracy for entailment: {accuracy_entailment}%")
print(f"Accuracy for contradiction: {accuracy_contradiction}%")
print(f"Accuracy for neutral: {accuracy_neutral}%")