import  pandas as pd
df = pd.read_csv('predictlabels.csv')

#df['ChatGPT_Explanation'] = df['ChatGPT_Explanation'].str.lower()

#df.to_csv('predictlabels.csv', index=False)

matches = df['gold_label'] == df['ChatGPT_Explanation']

percentage = (matches.sum() / len(df)) * 100

print(f"Percentage of rows where the two columns are the same: {percentage}%")

correct_entailment = ((df['gold_label'] == 'entailment') & (df['gold_label'] == df['ChatGPT_Explanation'])).sum()
correct_contradiction = ((df['gold_label'] == 'contradiction') & (df['gold_label'] == df['ChatGPT_Explanation'])).sum()
correct_neutral = ((df['gold_label'] == 'neutral') & (df['gold_label'] == df['ChatGPT_Explanation'])).sum()

# Calculate the total number of instances for each category
total_entailment = (df['gold_label'] == 'entailment').sum()
total_contradiction = (df['gold_label'] == 'contradiction').sum()
total_neutral = (df['gold_label'] == 'neutral').sum()

# Calculate the accuracy for each category
accuracy_entailment = (correct_entailment / total_entailment) * 100 if total_entailment > 0 else 0
accuracy_contradiction = (correct_contradiction / total_contradiction) * 100 if total_contradiction > 0 else 0
accuracy_neutral = (correct_neutral / total_neutral) * 100 if total_neutral > 0 else 0

print(f"Accuracy for entailment: {accuracy_entailment}%")
print(f"Accuracy for contradiction: {accuracy_contradiction}%")
print(f"Accuracy for neutral: {accuracy_neutral}%")