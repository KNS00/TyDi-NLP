# This script takes quite a long time to run. It creates the dictionary but also translates it to English.

from datasets import load_dataset
import string
import nltk
import json
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
print("Import Success")

from googletrans import Translator
translator = Translator()

# Loading the dataset
dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"]
validation_set = dataset["validation"]

languages = ['indonesian', 'bengali', 'arabic']
# Creating a dictionary for the training and validation set, that holds all three languages.
train_set_dict = {}
val_set_dict = {}

for language in languages:
    print("Adding ", language, "to dict")
    train_set_dict[language] = train_set.filter(lambda example: example["language"] == language)
    val_set_dict[language] = validation_set.filter(lambda example: example["language"] == language)

#for key in train_set_dict:
#    train_set_dict[key] = train_set_dict[key].select(range(100))

#for key in val_set_dict:
#    val_set_dict[key] = val_set_dict[key].select(range(100))

#for key in train_set_dict:
#    train_set_dict[key] = train_set_dict[key].select(range(5655, 5756))

#for key in val_set_dict:
#    val_set_dict[key] = val_set_dict[key].select(range(5655, 5756))



# Tokenizing question words, text and answers for training set
for language in languages:
    print("\nTokenizing question, answer and document text for", language)
    train_set_dict[language] = train_set_dict[language].map(lambda example: {'question_words': nltk.word_tokenize(example['question_text']), 'answer_words': nltk.word_tokenize(example['annotations']['answer_text'][0]), 'doc_words': nltk.word_tokenize(example['question_text']) + nltk.word_tokenize(example['document_plaintext']), 'doc_text_words': nltk.word_tokenize(example['document_plaintext'])})
    val_set_dict[language] = val_set_dict[language].map(lambda example: {'question_words': nltk.word_tokenize(example['question_text']), 'answer_words': nltk.word_tokenize(example['annotations']['answer_text'][0]), 'doc_words': nltk.word_tokenize(example['question_text']) + nltk.word_tokenize(example['document_plaintext']), 'doc_text_words': nltk.word_tokenize(example['document_plaintext'])})
    print("Done\n")



# Removing stopwords
language_mapping = {
    'indonesian': 'indonesian',
    'bengali': 'english',  # Placeholder
    'arabic': 'arabic'
}

def remove_stopwords_from_list(word_list, language):
    nltk_language = language_mapping.get(language, 'english')
    stop_words = set(stopwords.words(nltk_language))

    # Adding punctuation to removed chars
    punctuations = string.punctuation + "،''``"

    return [word for word in word_list if word.lower() not in stop_words and word not in punctuations]


def translator_(example, source_language):
    translated_question = translator.translate(example, src=source_language, dest='en').text
    #translated_doc_text = translator.translate(example['document_plaintext'], src=source_language, dest='en').text
    #example['question_text_eng'] = translated_question
    #example['document_plaintext_eng'] = translated_doc_text
    #example['question_words_eng'] = nltk.word_tokenize(translated_question)
    #example['doc_text_words_eng'] = nltk.word_tokenize(translated_doc_text)
    return translated_question


print("Removing stopwords")
for lang in languages:
    train_set_dict[lang] = train_set_dict[lang].map(
        lambda example: {
            'question_words': remove_stopwords_from_list(nltk.word_tokenize(example['question_text']), lang),
            'answer_words': remove_stopwords_from_list(nltk.word_tokenize(example['annotations']['answer_text'][0]), lang),
            'doc_words': remove_stopwords_from_list(nltk.word_tokenize(example['question_text']) + nltk.word_tokenize(example['document_plaintext']), lang),
            'doc_text_words': remove_stopwords_from_list(nltk.word_tokenize(example['document_plaintext']), lang),
            'question_text_eng' : translator_(example['question_text'], lang),
            'doc_plaintext_eng' : translator_(example['document_plaintext'], lang)
        }
    )
for lang in languages:
    train_set_dict[lang] = train_set_dict[lang].map(
        lambda example: {
            'question_words_eng' : remove_stopwords_from_list(nltk.word_tokenize(example['question_text_eng']), 'english'),
            'doc_text_words_eng' : remove_stopwords_from_list(nltk.word_tokenize(example['doc_plaintext_eng']), 'english')
        }
    )


for lang in languages:
    val_set_dict[lang] = val_set_dict[lang].map(
        lambda example: {
            'question_words': remove_stopwords_from_list(nltk.word_tokenize(example['question_text']), lang),
            'answer_words': remove_stopwords_from_list(nltk.word_tokenize(example['annotations']['answer_text'][0]), lang),
            'doc_words': remove_stopwords_from_list(nltk.word_tokenize(example['question_text']) + nltk.word_tokenize(example['document_plaintext']), lang),
            'doc_text_words': remove_stopwords_from_list(nltk.word_tokenize(example['document_plaintext']), lang),
            'question_text_eng' : translator_(example['question_text'], lang),
            'doc_plaintext_eng' : translator_(example['document_plaintext'], lang)
        }
    )
for lang in languages:
    val_set_dict[lang] = val_set_dict[lang].map(
        lambda example: {
            'question_words_eng' : remove_stopwords_from_list(nltk.word_tokenize(example['question_text_eng']), 'english'),
            'doc_text_words_eng' : remove_stopwords_from_list(nltk.word_tokenize(example['doc_plaintext_eng']), 'english')
        }
    )