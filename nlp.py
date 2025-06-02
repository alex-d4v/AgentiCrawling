import spacy
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_text(text, model_name="en_core_web_sm"):
    """
        Preprocess a text by removing punctuations and verbs.
    """
    # Load spaCy model
    nlp = spacy.load(model_name)
    # Process the text with spaCy
    doc = nlp(text)
    # Filter out punctuation and verbs
    filtered_tokens = []
    for token in doc:
        if token.is_punct:
            continue
        if token.pos_.startswith('VERB'):
            continue
        if token.is_stop:
            continue
        # Keep the token
        filtered_tokens.append(token.text)
    # Join the remaining tokens back into text
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def extract_topics(df , model):
    # Create custom vectorizer to focus on meaningful terms
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words="english",
        min_df=2,  # Term must appear in at least 2 documents
        max_df=0.5,  # Term must appear in less than 50% of documents
        lowercase=True
    )
    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=model,
        vectorizer_model=vectorizer_model,
        min_topic_size=int(np.log(len(df))-1),# Minimum size of a topic , assuming a logarithmic scale
        nr_topics=len(df),
        calculate_probabilities=True,
        verbose=True
    )
    print("Fitting BERTopic model...")
    # Fit the model and predict topics
    topics, probabilities = topic_model.fit_transform(df['preprocessed_text'].tolist())
    print(f"Discovered {len(set(topics))} topics")
    return topics
    