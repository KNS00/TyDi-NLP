import sys
from datasets import load_dataset
import nltk
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Loading TyDi
dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"]
validation_set = dataset["validation"]

train_set_dict = {}
val_set_dict = {}
languages = sys.argv[1:]


def check_for_stopwords(word_list, language):
    """
    Including stopwords gave better results. Decided to discard this function for now.
    """
    #stop_words = set(stopwords.words(language))

    #punctuations = string.punctuation + "،''``"
    #return [word for word in word_list if word.lower() not in stop_words and word not in punctuations]
    return word_list

def process_dataset(dataset, lang):
    return dataset.map(
        lambda example: {
            'question_words': check_for_stopwords(
                nltk.word_tokenize(example['question_text']), lang),
            'answer_words': check_for_stopwords(
                nltk.word_tokenize(example['annotations']['answer_text'][0]), lang),
            'doc_words': check_for_stopwords(
                nltk.word_tokenize(example['question_text'], lang) + [' [SEP] '] + nltk.word_tokenize(example['document_plaintext']), lang),
            'doc_text_words': check_for_stopwords(
                nltk.word_tokenize(example['document_plaintext']), lang),
            'answerable': 1 if example['annotations']['answer_text'][0] else 0
        }
    )

test_set_dict = {}
def separate_val_set(val_set_dict, languages, test_size=0.1, random_state=42):
    """
    Splits val_set_dict for each language into validation and test sets.

    Parameters:
        val_set_dict (dict): Original dataset with structure {lang: Dataset}.
        languages (list): List of language keys to process (e.g., ['english', 'french']).
        test_size (float): Proportion of data to assign to the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Updated val_set_dict and test_set_dict for all languages.
    """
    # Initialize dictionaries for validation and test sets
    validation_set_dict = {}
    test_set_dict = {}

    for lang in languages:
        print(f"Processing language: {lang}")

        data = val_set_dict[lang] 
        data_list = [dict(row) for row in data]

        validation_data, test_data = train_test_split(data_list, test_size=test_size, random_state=random_state)

        validation_set_dict[lang] = {key: [row[key] for row in validation_data] for key in data[0].keys()}
        test_set_dict[lang] = {key: [row[key] for row in test_data] for key in data[0].keys()}

        print(f"Validation set size for {lang}: {len(validation_set_dict[lang]['question_text'])}")
        print(f"Test set size for {lang}: {len(test_set_dict[lang]['question_text'])}")

    return validation_set_dict, test_set_dict

def main():
    if len(sys.argv) == 1:
        print("Missing language(s) as input")
        return
    for lang in languages:
        print("Adding ", lang, "to dict")
        train_set_ = train_set.filter(lambda example: example["language"] == lang)
        val_set_ = validation_set.filter(lambda example: example["language"] == lang)
        train_set_dict[lang] = process_dataset(train_set_, lang)
        val_set_dict[lang] = process_dataset(val_set_, lang)

if __name__ == "__main__":
    main()