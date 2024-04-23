import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from rouge import Rouge 
from extract_information2 import read_reports_from_folder



def abstractive_summary(report, model, tokenizer):
    inputs = tokenizer.encode(report, return_tensors='pt', truncation=True, max_length=512)
    summary_ids = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    return tokenizer.decode(summary_ids[0])

def main():
    folder_path = "C:/Users/prith/OneDrive/Desktop/NLP Git/beth_train"  # Update this with the path to your folder containing text files

    print("Starting to extract information from files...")

    all_reports = read_reports_from_folder(folder_path)

    print(f"Finished extracting information from {len(all_reports)} files.")
    
    rouge = Rouge()

    # Load PEGASUS
    model_name = 'google/pegasus-pubmed'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Initialize a list to hold all the data
    data = []

    # Generate summary for each report and save it to a text file
    for i, report in enumerate(all_reports):
        summary = abstractive_summary(report, model, tokenizer)
        
        with open(f'C:/Users/prith/OneDrive/Desktop/NLP Git/Pegasus_summary/summary_{i}.txt', 'w') as f:
            f.write(summary)
        try:
            scores = rouge.get_scores(summary, report)
            data.append([f'summary_{i}.txt', scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']])        
        except ValueError as e:
            print(f"Error calculating ROUGE score for file {i}: {e}")

    # Convert the data list into a DataFrame
    df = pd.DataFrame(data, columns=['File Name', 'ROUGE - 1', 'ROUGE - 2','ROUGE - L'])

    # Write the DataFrame to an Excel file
    df.to_excel('summary_scores_pegasus.xlsx', index=False)

if __name__ == "__main__":
    main()