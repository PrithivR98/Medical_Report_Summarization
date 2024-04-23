import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from extract_information2 import read_reports_from_folder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
from rouge import Rouge
import pandas as pd
def extractive_summary(text, model, tokenizer):
    sentences = sent_tokenize(text)
    inputs = tokenizer.batch_encode_plus(sentences, return_tensors='tf', max_length=512, truncation=True, padding='longest')
    outputs = model(inputs['input_ids'])
    sentence_embeddings = tf.reduce_mean(outputs[0], axis=1).numpy()
    similarity_matrix = cosine_similarity(sentence_embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    sentence_scores = np.sum(similarity_matrix, axis=0)
    ranked_sentence = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Calculate the number of sentences to return
    num_sentences = int(len(sentences) * 0.6)
    
    return ' '.join([s for (_, s) in ranked_sentence[:num_sentences]])

def main():
    folder_path = "C:/Users/prith/OneDrive/Desktop/NLP Git/beth_train"  # Update this with the path to your folder containing text files

    print("Starting to extract information from files...")

    all_reports = read_reports_from_folder(folder_path)

    print(f"Finished extracting information from {len(all_reports)} files.")
    
    rouge = Rouge()
    
    data = []

    # Load BioBERT
    model = TFBertModel.from_pretrained("monologg/biobert_v1.1_pubmed")
    tokenizer = BertTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")

    # Generate summary for each report and save it to a text file
    for i, report in enumerate(all_reports):
        summary = extractive_summary(report, model, tokenizer)
        with open(f'C:/Users/prith/OneDrive/Desktop/NLP Git/BERT_summary/summary_{i}.txt', 'w') as f:
            f.write(summary)
        try:
            scores = rouge.get_scores(summary, report)
            data.append([f'summary_{i}.txt', scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']])        
        except ValueError as e:
            print(f"Error calculating ROUGE score for file {i}: {e}")
            
    df = pd.DataFrame(data, columns=['File Name', 'ROUGE-1 Score', 'ROUGE-2 Score', 'ROUGE-L Score'])
    df.to_excel('summary_Scores.xlsx', index=False)

if __name__ == "__main__":
    main()