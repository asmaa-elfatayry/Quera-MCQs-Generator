# Importing necessary libraries

import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
from sense2vec import Sense2Vec
import nltk
import numpy
import random
from nltk import FreqDist

nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
import mcq
from mcq import get_keywords
from mcq import get_sentences_for_keywords
from mcq import generate_questions_mcq
import time


# Defining a class named MCQGen
class MCQGen:

    def __init__(self):
        # Initialize the MCQGen class

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')  # Initialize T5 tokenizer
        model = T5ForConditionalGeneration.from_pretrained('Parth/result')  # Load pre-trained T5 model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available, else use CPU
        model.to(device)  # Move the model to the selected device
        self.device = device  # Store the selected device in class attribute
        self.model = model  # Store the loaded T5 model in class attribute
        self.nlp = spacy.load('en_core_web_sm')  # Load spaCy English model

        self.s2v = Sense2Vec().from_disk('../s2v_old')  # Load Sense2Vec model
        self.fdist = FreqDist(brown.words())  # Create frequency distribution of words from Brown corpus
        self.normalized_levenshtein = NormalizedLevenshtein()  # Initialize NormalizedLevenshtein for calculating string similarity
        self.set_seed(42)  # Set random seed for reproducibility

    def set_seed(self, seed):
        # Set random seed for reproducibility
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def predict_mcq(self, context):
        # Generate MCQs for given context

        start = time.time()  # Start timer

        inp = {
            "input_text": context.get("input_text"),  # Get input text from context
            "max_questions": context.get("max_questions", 4)
            # Get maximum number of questions to generate from context, default is 4
        }

        text = inp['input_text']  # Extract input text
        sentences = mcq.tokenize_sentences(text)  # Tokenize input text into sentences
        joiner = " "
        modified_text = joiner.join(sentences)  # Join tokenized sentences into modified text

        keywords = get_keywords(self.nlp, modified_text, inp['max_questions'], self.s2v, self.fdist,
                                self.normalized_levenshtein, len(sentences))
        # Extract keywords from modified text using various techniques

        keyword_sentence_mapping = get_sentences_for_keywords(keywords, sentences)
        # Get sentences containing the keywords

        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(
                keyword_sentence_mapping[k][:3])  # Extract first 3 words from each sentence containing the keyword
            keyword_sentence_mapping[k] = text_snippet

        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            return final_output  # If no keywords found, return empty final output
        else:
            try:
                generated_questions = generate_questions_mcq(keyword_sentence_mapping, self.device, self.tokenizer,
                                                             self.model, self.s2v, self.normalized_levenshtein)
                # Generate MCQs for the sentences containing the keywords

            except:
                return final_output  # If error occurs during MCQ generation, return empty final output
            end = time.time()  # End timer

            final_output["statement"] = modified_text  # Store the modified text in final output
            final_output["questions"] = generated_questions["questions"]  # Store the generated MCQs in final output
            final_output["time_taken"] = end - start  # Store the time taken for MCQ generation

            if torch.device == 'cuda':
                # If using CUDA, empty the GPU cache to free up memory
                torch.cuda.empty_cache()

            return final_output


from pprint import pprint
import nltk

# nltk.download('stopwords')


context = {
    "input_text": '''Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.IBM has a rich history with machine learning. One of its own, Arthur Samuel, is credited for coining the term, “machine learning” with his research (PDF, 481 KB) (link resides outside IBM) around the game of checkers. Robert Nealey, the self-proclaimed checkers master, played the game on an IBM 7094 computer in 1962, and he lost to the computer. Compared to what can be done today, this feat seems trivial, but it’s considered a major milestone in the field of artificial intelligence. '''

}


# pprint (output)


def generate_mcq_questions(text):
    print("Generating MCQs...")
    print(text)
    qg = MCQGen()
    output = qg.predict_mcq(text)
    # these are the questions
    print(output)
    questions = output['questions']
    result = []

    for question in questions:
        correct_answer = question['answer']
        wrong_choices = random.sample(question['options'] + question['extra_options'], 3)
        choices = [correct_answer] + wrong_choices
        random.shuffle(choices)

        if len(choices) == 4:
            q = {}
            q['question_statement'] = question['question_statement']
            q['choices'] = choices
            q['correct_answer'] = chr(ord('a') + choices.index(correct_answer))
            result.append(q)

    return result
# This function now creates a list of dictionaries, where each dictionary represents a single multiple-choice question. The dictionary has three keys: question_statement, choices, and correct_answer. The choices key is a list of the four answer choices for the question, and the correct_answer key is the letter corresponding to the correct answer choice. The list of dictionaries is returned as the output of the function.


# generate_mcq_questions(output)
