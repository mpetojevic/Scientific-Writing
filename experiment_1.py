import openai
import pandas as pd
from difflib import SequenceMatcher
import time

# Set your OpenAI API key

# Load eSNLI dataset
esnli_dataset_path = "esnli_train_1.csv"
esnli_df = pd.read_csv(esnli_dataset_path).head(100)


results_df = pd.DataFrame(columns=["pairID", "gold_Label", "ChatGPT_Explanation", "Similarity"])

request_count = 0


# Function to process a batch of rows
def process_batch(batch_df, request_counter):
    for index, row in batch_df.iterrows():
        # Prepare input for ChatGPT
        input_text = (
            "Based on premise and hypothesis, \n"
            "find the entailment label between them: entailment, contradiction or neutral."
            "Your answer must be one word only.\n"
            f"Premise: {row['Sentence1']} Hypothesis: {row['Sentence2']} \n"
            f"Marked Words Sentence 1: {row['Sentence1_marked_1']} Marked Words Sentence 1: {row['Sentence2_marked_1']}"
        )

        try:
            # Generate label using ChatGPT
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct-0914",
                prompt=input_text,
                max_tokens=50,
                temperature=0.7
            )

            request_counter += 1

            # Extract the generated label from the response
            chatgpt_explanation = response.choices[0].text.strip()

            print("ChatGPT predicted label:", chatgpt_explanation)

            # Append results to the new DataFrame
            results_df.loc[len(results_df)] = {
                "pairID": row['pairID'],
                "gold_Label": row['gold_label'],
                "ChatGPT_Explanation": chatgpt_explanation,
            }
        except openai.NotFoundError:
            print("Model not found or other API error. Skipping this row.")
        except openai.RateLimitError:
            print(f"Rate limit reached after {request_counter} requests. Pausing for 60 seconds.")
            time.sleep(60)

    return request_counter



batch_size = 100
for start in range(0, len(esnli_df), batch_size):
    end = start + batch_size
    batch_df = esnli_df[start:end]
    process_batch(batch_df, request_count)
    results_df.to_csv("intermediate_results.csv", index=False)  # Save intermediate results


results_df.to_csv("PredictLabels.csv", index=False)
print("Results saved successfully.")
