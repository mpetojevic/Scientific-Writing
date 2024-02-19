import openai
import pandas as pd
from difflib import SequenceMatcher
import time

# Set your OpenAI API key

# Load eSNLI dataset
esnli_dataset_path = "esnli_train_1.csv"
esnli_df = pd.read_csv(esnli_dataset_path).head(100)

# Create a new DataFrame to store the results
results_df = pd.DataFrame(columns=["pairID", "gold_Label", "eSNLI_Explanation", "ChatGPT_Explanation", "Similarity"])

request_count = 0


# Function to process a batch of rows
def process_batch(batch_df, request_counter):
    for index, row in batch_df.iterrows():
        # Prepare input for ChatGPT
        input_text = (
            "Based on premise, hypothesis, gold explanation and marked words in sentences provide an explanation of the "
            "relationship between two sentences.\n"
            "Explanation can be at most 1 sentence, should be short and precise, not repeating gold label, "
            "hypothesis or the premise. Your explanation mustn't mention marked words! You should just explain"
            "what situation could be going on based on the two sentences.\n"
            f"Premise: {row['Sentence1']} Hypothesis: {row['Sentence2']} Gold Explanation: {row['gold_label']}\n"
            f"Marked Words Sentence 1: {row['Sentence1_marked_1']} Marked Words Sentence 1: {row['Sentence2_marked_1']}"
        )

        try:
            # Generate explanation using ChatGPT
            response = openai.completions.create(
                model="gpt-3.5-turbo-instruct-0914",
                prompt=input_text,
                max_tokens=50,
                temperature=0.7
            )

            request_counter += 1

            # Extract the generated explanation from the response
            chatgpt_explanation = response.choices[0].text.strip()

            print("ChatGPT Explanation:", chatgpt_explanation)

            # Calculate similarity using SequenceMatcher
            similarity_ratio = SequenceMatcher(None, row['Explanation_1'], chatgpt_explanation).ratio()

            # Append results to the new DataFrame
            results_df.loc[len(results_df)] = {
                "pairID": row['pairID'],
                "gold_Label": row['gold_label'],
                "eSNLI_Explanation": row['Explanation_1'],
                "ChatGPT_Explanation": chatgpt_explanation,
                "Similarity": similarity_ratio
            }
        except openai.NotFoundError:
            print("Model not found or other API error. Skipping this row.")
        except openai.RateLimitError:
            print(f"Rate limit reached after {request_counter} requests. Pausing for 60 seconds.")
            time.sleep(60)

    return request_counter


# Process the dataset in batches
batch_size = 100  # Set the batch size as per your requirement
for start in range(0, len(esnli_df), batch_size):
    end = start + batch_size
    batch_df = esnli_df[start:end]
    process_batch(batch_df, request_count)
    results_df.to_csv("intermediate_results.csv", index=False)  # Save intermediate results

# Save the final results to a new CSV file
results_df.to_csv("PremiseAgnostics.csv", index=False)
print("Results saved successfully.")
