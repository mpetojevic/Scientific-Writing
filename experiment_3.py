import openai
import pandas as pd
from difflib import SequenceMatcher
import time

# Set your OpenAI API key
openai.api_key = "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"

# Load eSNLI dataset
esnli_dataset_path = "esnli_train_1.csv"
esnli_df = pd.read_csv(esnli_dataset_path).head(20)

# Create a new DataFrame to store the results
results_df = pd.DataFrame(columns=["pairID", "gold_Label", "esnli_Explanation", "ChatGPT_Predicted_Label", "ChatGPT_Explanation"])

request_count = 0


# Function to process a batch of rows
def process_batch(batch_df, request_counter):
    for index, row in batch_df.iterrows():
        # Prepare input for ChatGPT
        label_prompt = (f"Predict the entailment label for the given premise and hypothesis"
                        f" (entailment, contradiction or neutral)."
                        f"\nPremise: {row['Sentence1']}\nHypothesis: {row['Sentence2']}")

        try:
            # Generate label using ChatGPT
            predicted_label = openai.completions.create(
                model="gpt-3.5-turbo-instruct-0914",
                prompt=label_prompt,
                max_tokens=50,
                temperature=0.7
            )

            request_counter += 1

            # Extract the generated label from the response
            predicted_label = predicted_label.choices[0].text.strip()

            explanation_prompt = (f"Explain why the relationship between the following premise and hypothesis is {predicted_label}. Explanation in max. 1 sentence in natural language"
                                  f"\nPremise: {row['Sentence1']}\nHypothesis: {row['Sentence2']}")
            explanation = openai.completions.create(
                model="gpt-3.5-turbo-instruct-0914",
                prompt=explanation_prompt,
                max_tokens=50,
                temperature=0.7
            )

            explanation = explanation.choices[0].text.strip()
            # Append results to the new DataFrame
            results_df.loc[len(results_df)] = {
                "pairID": row['pairID'],
                "gold_Label": row['gold_label'],
                "eSNLI_Explanation": row['Explanation_1'],
                "ChatGPT_Predicted_Label": predicted_label,
                "ChatGPT_Explanation": explanation
            }

            print("Explanation: ", explanation)
            request_counter += 1

        except openai.NotFoundError:
            print("Model not found or other API error. Skipping this row.")
        except openai.RateLimitError:
            print(f"Rate limit reached after {request_counter} requests. Pausing for 60 seconds.")
            time.sleep(60)

    return request_counter


batch_size = 20


for start in range(0, len(esnli_df), batch_size):
    end = start + batch_size
    batch_df = esnli_df[start:end]
    process_batch(batch_df, request_count)
    results_df.to_csv("intermediate_results.csv", index=False)


results_df.to_csv("PredictAndExplain.csv", index=False)
print("Results saved successfully.")
