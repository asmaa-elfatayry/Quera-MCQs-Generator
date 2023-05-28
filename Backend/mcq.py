
# import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
# from sense2vec import Sense2Vec
# import requests
from collections import OrderedDict
import string
import pke
import nltk
# from nltk import FreqDist
nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import stopwords
# from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor


def generate_one_edit_away_words(word):
    """
    Generates all possible edits that are one edit away from the input word.
    Edits can include deletions, transpositions, replacements, or insertions
    of a single character in the input word.
    """
    # Define a string of lowercase letters, space, and punctuation
    letters = 'abcdefghijklmnopqrstuvwxyz ' + string.punctuation
    
    # Generate a list of tuples representing all possible splits of the input word
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    # Generate a list of words by deleting one character from the input word
    deletions = [L + R[1:] for L, R in splits if R]
    
    # Generate a list of words by transposing two adjacent characters in the input word
    transpositions = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    
    # Generate a list of words by replacing one character in the input word with a character from 'letters'
    replacements = [L + c + R[1:] for L, R in splits if R for c in letters]
    
    # Generate a list of words by inserting one character from 'letters' into the input word
    insertions = [L + c + R for L, R in splits for c in letters]
    
    # Return a set containing all the generated edits
    return set(deletions + transpositions + replacements + insertions)




def get_similar_words_with_sense2vec(input_word, sense2vec_model):
    """
    Retrieves similar words using sense2vec model based on an input word.
    Performs preprocessing on the input word, generates word edits, and
    compares them with the most similar words obtained from the sense2vec model.
    """
    output = []  # Initialize an empty list to store the output words
    
    # Preprocess the input word by removing punctuation and converting to lowercase
    input_word_preprocessed = input_word.translate(input_word.maketrans("", "", string.punctuation))
    input_word_preprocessed = input_word_preprocessed.lower()
    
    # Generate word edits (words that are one edit away from the input word)
    word_edits = generate_one_edit_away_words(input_word_preprocessed)
    
    # Replace spaces in the input word with underscores
    input_word = input_word.replace(" ", "_")
    
    # Get the best sense of the input word using the sense2vec model
    sense = sense2vec_model.get_best_sense(input_word)
    
    # Get the most similar words to the input word from the sense2vec model
    most_similar = sense2vec_model.most_similar(sense, n=15)
    
    compare_list = [input_word_preprocessed]  # Initialize a list to store processed words for comparison
    
    # Loop through each word in the most similar words obtained from the sense2vec model
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ")  # Extract the word and replace underscores with spaces
        append_word = append_word.strip()  # Remove leading/trailing whitespaces
        append_word_processed = append_word.lower()  # Convert the word to lowercase
        append_word_processed = append_word_processed.translate(append_word_processed.maketrans("", "", string.punctuation))  # Remove punctuation from the word
        
        # Check if the processed word is not already in compare_list, not equal to the input word,
        # and not in the generated word edits
        if append_word_processed not in compare_list and input_word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
            output.append(append_word.title())  # Append the processed word to the output list with title case
            compare_list.append(append_word_processed)  # Add the processed word to the compare_list for future comparison
    
    out = list(OrderedDict.fromkeys(output))  # Remove duplicates from the output list
    
    return out  # Return the list of similar words


def get_distractor_options(answer, sense2vec_model):
    """
    Retrieves distractor options for a given answer word using sense2vec model.
    """
    distractors = []  # Initialize an empty list to store distractors
    
    try:
        distractors = get_similar_words_with_sense2vec(answer, sense2vec_model)  # Get similar words using sense2vec model
        if len(distractors) > 0:
            print("Sense2vec distractors retrieved successfully for word:", answer)
            return distractors, "sense2vec"  # Return the distractors and method used as "sense2vec"
    except:
        print("Failed to retrieve Sense2vec distractors for word:", answer)

    return distractors, "None"  # Return empty distractors and method used as "None"


def tokenize_sentences(text):
    """
    Tokenizes input text into sentences.
    """
    sentences = [sent_tokenize(text)]  # Tokenize text into sentences using sent_tokenize()
    sentences = [sentence for sublist in sentences for sentence in sublist]  # Flatten the list of sentences
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]  # Remove short sentences with less than 20 characters
    return sentences  # Return the list of tokenized sentences


def get_sentences_for_keywords(keywords, sentences):
    """
    Retrieves sentences containing keywords from a list of sentences.
    """
    keyword_processor = KeywordProcessor()  # Initialize a keyword processor
    keyword_sentences = {}  # Initialize an empty dictionary to store sentences containing keywords
    
    for keyword in keywords:  # Iterate through the list of keywords
        keyword = keyword.strip()  # Remove leading/trailing whitespaces
        keyword_sentences[keyword] = []  # Create a key in the dictionary for each keyword and initialize an empty list
        keyword_processor.add_keyword(keyword)  # Add each keyword to the keyword processor

    for sentence in sentences:  # Iterate through the list of sentences
        keywords_found = keyword_processor.extract_keywords(sentence)  # Extract keywords from the sentence using the keyword processor
        for keyword in keywords_found:  # Iterate through the extracted keywords
            keyword_sentences[keyword].append(sentence)  # Append the sentence to the list of sentences for the corresponding keyword in the dictionary

    for keyword in keyword_sentences.keys():  # Iterate through the dictionary keys (keywords)
        sentences = keyword_sentences[keyword]  # Retrieve the list of sentences for each keyword
        sentences = sorted(sentences, key=len, reverse=True)  # Sort the sentences by length in descending order
        keyword_sentences[keyword] = sentences  # Update the dictionary with the sorted list of sentences for each keyword

    delete_keys = []
    for keyword in keyword_sentences.keys():  # Iterate through the dictionary keys (keywords)
        if len(keyword_sentences[keyword]) == 0:  # Check if the list of sentences for a keyword is empty
            delete_keys.append(keyword)  # If so, add the keyword to the list of keys to be deleted
    for keyword in delete_keys:  # Iterate through the list of keys to be deleted
        del keyword_sentences[keyword]  # Delete the corresponding key-value pair from the dictionary

    return keyword_sentences  # Return the dictionary containing sentences for each keyword


def is_far(words_list, current_word, threshold, normalized_levenshtein):
    """
    Checks if a current word is far from a list of words based on normalized Levenshtein distance.
    """
    score_list = []
    for word in words_list:
        # Compute the normalized Levenshtein distance between the current word and each word in the list
        score_list.append(normalized_levenshtein.distance(word.lower(), current_word.lower()))

    if min(score_list) >= threshold:
        # If the minimum distance is greater than or equal to the threshold, return True
        return True
    else:
        # Otherwise, return False
        return False


def filter_phrases(phrase_keys, max_phrases, normalized_levenshtein):
    """
    Filters a list of phrase keys based on normalized Levenshtein distance and a maximum number of phrases.
    """
    filtered_phrases = []

    if len(phrase_keys) > 0:
        filtered_phrases.append(phrase_keys[0])  # Add the first phrase key to the filtered list

        for phrase_key in phrase_keys[1:]:
            if is_far(filtered_phrases, phrase_key, 0.7, normalized_levenshtein):
                # Check if the current phrase key is far from the filtered list based on normalized Levenshtein distance
                filtered_phrases.append(phrase_key)  # If so, add it to the filtered list

            if len(filtered_phrases) >= max_phrases:
                # Check if the filtered list has reached the maximum number of phrases
                break  # If so, exit the loop

    return filtered_phrases


def get_nouns_multipartite(text):
    """
    Extracts the top nouns and proper nouns (named entities) from a given text using MultipartiteRank algorithm.
    """
    out = []
    # Create an instance of MultipartiteRank extractor
    extractor = pke.unsupervised.MultipartiteRank()
    # Load the input text and specify the language as English
    extractor.load_document(input=text, language='en')
    # Specify the parts-of-speech (POS) to be considered as candidates (nouns and proper nouns)
    pos = {'PROPN', 'NOUN'}
    
    # Define a list of stop words to be excluded from candidates
    stoplist = list(string.punctuation)
    stoplist += stopwords.words('english')
    
    # Select candidates based on POS
    extractor.candidate_selection(pos=pos)
    
    # Build the Multipartite graph and rank candidates using random walk,
    # with alpha controlling the weight adjustment mechanism and threshold/method parameters
    try:
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
    except:
        return out

    # Get the n best keyphrases from the extractor
    keyphrases = extractor.get_n_best(n=10)

    # Extract the candidate phrases from the keyphrases and append them to the output list
    for key in keyphrases:
        out.append(key[0])

    return out




def get_phrases(doc):
    """
    Extracts noun phrases from a given document using spaCy's noun chunking functionality,
    and returns a list of unique phrases.
    """
    phrases = {}
    for np in doc.noun_chunks:
        phrase = np.text
        len_phrase = len(phrase.split())
        if len_phrase > 1:
            if phrase not in phrases:
                phrases[phrase] = 1
            else:
                phrases[phrase] += 1

    phrase_keys = list(phrases.keys())
    # Sort the phrase keys by length in descending order
    phrase_keys = sorted(phrase_keys, key=lambda x: len(x), reverse=True)
    # Limit the number of phrase keys to 50
    phrase_keys = phrase_keys[:50]
    return phrase_keys



def MCQs_available(word, s2v):
    # Replace spaces with underscores to convert the word into a format
    # that can be used with the word sense embedding model
    word = word.replace(" ", "_")
    
    # Call the get_best_sense() method on the s2v object to retrieve the best sense
    # of the word based on the embeddings
    sense = s2v.get_best_sense(word)
    
    # If a sense is retrieved (i.e., sense is not None), return True indicating that
    # MCQs are available for the input word, otherwise return False
    if sense is not None:
        return True
    else:
        return False


def get_keywords(nlp, text, max_keywords, s2v, fdist, normalized_levenshtein, no_of_sentences):
    """
    Extracts keywords from a given text using multiple techniques, filters and ranks them, and returns
    a list of top keywords.
    """
    doc = nlp(text)
    max_keywords = int(max_keywords)

    # Get keywords using the MultipartiteRank algorithm
    keywords = get_nouns_multipartite(text)
    # Sort keywords by frequency using the provided frequency distribution
    keywords = sorted(keywords, key=lambda x: fdist[x])
    # Filter keywords by relevance using the filter_phrases function
    keywords = filter_phrases(keywords, max_keywords, normalized_levenshtein)

    # Get noun phrases from the document
    phrase_keys = get_phrases(doc)
    # Filter noun phrases by relevance using the filter_phrases function
    filtered_phrases = filter_phrases(phrase_keys, max_keywords, normalized_levenshtein)

    # Combine keywords and filtered noun phrases to get a total list of relevant phrases
    total_phrases = keywords + filtered_phrases

    # Further filter the total phrases based on the minimum of max_keywords and 2 times the number of sentences
    total_phrases_filtered = filter_phrases(total_phrases, min(max_keywords, 2 * no_of_sentences), normalized_levenshtein)

    # Filter the relevant phrases that are available as multiple-choice questions (MCQs)
    answers = []
    for answer in total_phrases_filtered:
        if answer not in answers and MCQs_available(answer, s2v):
            answers.append(answer)

    # Limit the final list of answers to max_keywords
    answers = answers[:max_keywords]
    return answers


def generate_questions_mcq(keyword_sent_mapping, device, tokenizer, model, sense2vec, normalized_levenshtein):
    batch_text = []
    answers = keyword_sent_mapping.keys()

    # Loop through each answer (keyword) in the keyword-sentence mapping dictionary
    for answer in answers:
        txt = keyword_sent_mapping[answer]
        context = "context: " + txt
        text = context + " " + "answer: " + answer + " </s>"
        batch_text.append(text)

    # Encode the batch_text using the tokenizer, pad the sequences to the maximum length, and return as PyTorch tensors
    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")

    # Print a message to indicate that the model is being run for generation
    print("Running model for generation")

    # Extract the input_ids and attention_masks from the encoding
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # Generate text using the pre-trained language model with the given input_ids and attention_masks
    with torch.no_grad():
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=150
        )

    # Create an empty dictionary to store the output MCQs
    output_array = {}
    output_array["questions"] = []

    # Loop through the generated outputs and process each MCQ
    for index, val in enumerate(answers):
        # Create a dictionary to store the individual MCQ
        individual_question = {}

        # Get the generated text for the current MCQ
        out = outs[index, :]

        # Decode the generated text to remove special tokens and extract the question statement
        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        Question = dec.replace("question:", "")

        # Remove leading/trailing whitespaces from the question statement
        Question = Question.strip()

        # Populate the individual_question dictionary with the question statement, question type, answer, ID, and options
        individual_question["question_statement"] = Question
        individual_question["question_type"] = "MCQ"
        individual_question["answer"] = val
        individual_question["id"] = index + 1
        individual_question["options"], individual_question["options_algorithm"] = get_distractor_options(val, sense2vec)

        # Filter the options using a measure of string similarity (normalized_levenshtein)
        individual_question["options"] = filter_phrases(individual_question["options"], 10, normalized_levenshtein)

        # Select a subset of options as extra options (beyond the first three) for the MCQ
        index = 3
        individual_question["extra_options"] = individual_question["options"][index:]
        individual_question["options"] = individual_question["options"][:index]

        # Set the context for the MCQ as the corresponding sentence from the keyword-sentence mapping
        individual_question["context"] = keyword_sent_mapping[val]

        # If the MCQ has options, append it to the output_array dictionary
        if len(individual_question["options"]) > 0:
            output_array["questions"].append(individual_question)

    # Return the generated MCQs as the output_array
    return output_array


