# -*- coding: utf-8 -*-
"""Naive Chatbot"""
import logging
import pickle
import string
import re

import numpy as np
import tensorflow as tf
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from typing import Optional

"""A simple chatbot that utilizes an intent classifier then matching with predefined text mappings.

Typical usage example:

    my_bot = NaiveChatbot(pretrained=True,
                          query_tokenizer_path="/../query_tokenizer.pickle", 
                          intent_tokenizer_path="/../intent_tokenizer.pickle", 
                          model_weights_path="/../checkpoint.ckpt",
                          db_responses2text_path="/../db_responses2text.pickle",
                          db_intent2response_path="/../db_intent2response.pickle",
                          db_stopwords_path="/../db_stopwords.pickle")
        user_input = input("user  > ")
        print("bot  > ", my_bot.get_reply(user_input))
"""

vocab_size = 500
embedding_dim = 128
max_length = 32
oov_tok = '<OOV>'  # Out of Vocabulary
training_portion = 1
previous_reply = 'احنا لسه في بداية الكلام'
arabic_punctuations = '''«»`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

def load_pickle_data(filepath):
    with open(filepath, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


class NaiveChatbot:

    def __get_model(self):
        # TODO(mshetairy): Create a .gin for model hyperparameters
        number_of_intents = len(self.intent_tokenizer.index_word.keys())
        number_of_classes = number_of_intents + 1
        model = Sequential(name="naive_chatbot")
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(embedding_dim)))
        model.add(Dense(number_of_classes, activation='softmax'))
        logging.info(model.summary())

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-6)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def __init__(self,
                 pretrained: bool = False,
                 query_tokenizer_path: Optional[str] = None,
                 intent_tokenizer_path: Optional[str] = None,
                 model_weights_path: Optional[str] = None,
                 db_responses2text_path: Optional[str] = None,
                 db_intent2response_path: Optional[str] = None,
                 db_stopwords_path: Optional[str] = None,
                 db_transliteration_path: Optional[str] = None):
        """Initializing an instance of the chatbot.

        Args:
            pretrained: If True loads required tokenizers and model weights.
            query_tokenizer_path: path to the Arabic query Tokenizer.
            intent_tokenizer_path: path to the Label Tokenizer of the user query's
                intent.
            model_weights_path: path to the pretrained intent classifier model
                weights.
            db_responses2text_path: path to the mapping of bot response type to
                possible text outcomes.
            db_intent2response_path: path to the mapping of user intents to
                possible bot response types.

        Raises:
            ValueError: An error occurred in the files paths.
        """
        if pretrained:
            if not all([query_tokenizer_path,
                        intent_tokenizer_path,
                        model_weights_path,
                        db_responses2text_path,
                        db_intent2response_path]):
                raise ValueError("All arguments must be strings when pretrained is True.")
            self.query_tokenizer = load_pickle_data(query_tokenizer_path)
            self.intent_tokenizer = load_pickle_data(intent_tokenizer_path)
            self.model = self.__get_model()
            self.model.load_weights(model_weights_path).expect_partial()
            self.db_responses2text = load_pickle_data(db_responses2text_path)
            self.db_intent2response = load_pickle_data(db_intent2response_path)
            # self.db_stopwords = load_pickle_data(db_stopwords_path)
            self.db_transliteration = load_pickle_data(db_transliteration_path)
            logging.info("Successfully loaded tokenizers, database and pretrained weights.")
        else:
            # Handle non-pretrained case if needed
            # ...
            pass

        # Additional class attributes or methods
        # ...
        pass

    def preprocess_query(self, query):
        text = query.translate(str.maketrans('', '', punctuations_list))
        # remove diacritics
        text = re.sub(arabic_diacritics, '', str(text))
        # remoce emoji
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
        query = regrex_pattern.sub(r'',text)
        norm = normalize_unicode(query)
        # Normalize alef variants to 'ا'
        norm = normalize_alef_ar(norm)
        # Normalize alef maksura 'ى' to yeh 'ي'
        norm = normalize_alef_maksura_ar(norm)
        # Normalize teh marbuta 'ة' to heh 'ه'
        norm = normalize_teh_marbuta_ar(norm)

        sent_safebw = self.db_transliteration(norm)
        return sent_safebw

    def __get_predictions(self, data):
        """Gets numerical model predictions."""
        model = self.model
        predictions = []
        for i in range(0, len(data)):
            prediction = model.predict(data[i, :].reshape(1, -1), verbose=0)
            predictions.append(np.argmax(prediction))
        return np.array(predictions)

    def get_intent(self, text, threshold=0.4):
        """Classifies the intent behind the input text."""
        intent_tokenizer = self.intent_tokenizer
        model = self.model
        query_tokenizer = self.query_tokenizer
        # db_stopwords = self.db_stopwords

        # for word in db_stopwords:
        #     token = ' ' + word + ' '
        #     text = text.replace(token, ' ')
        #     text = text.replace(' ', ' ')
        norm = self.preprocess_query(text)
        seq = query_tokenizer.texts_to_sequences([norm])
        padded = pad_sequences(seq, maxlen=max_length)
        pred = model.predict(padded, verbose=0)

        try:
            if np.max(pred) < threshold:
                label = ['']
            else:
                label = intent_tokenizer.sequences_to_texts(np.array([[np.argmax(pred)]]))
            label = ['other'] if label == [''] else label
            answer = label
        except:
            answer = ['other']
        return answer

    def get_reply(self, text, threshold=0.4):
        global previous_reply
        intent = self.get_intent(text, threshold)[0]
        if intent == "request_repeat":
            return previous_reply
        response_type = np.random.choice(self.db_intent2response[intent])
        reply = np.random.choice(self.db_responses2text[response_type])
        previous_reply = reply
        return reply
