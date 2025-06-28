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
    nlp.max_length = 50_000_000  # 50 million characters
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

def extract_topics(df, model , nr_topics=None):
    # Ensure min_topic_size is at least 2
    print(f"Examining {len(df)} texts for topic extraction...")
    topic_model = BERTopic(
        embedding_model=model,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True
    )
    print("Fitting BERTopic model...")
    topics, probabilities = topic_model.fit_transform(df['preprocessed_text'].tolist())
    print(f"Discovered {len(set(topics))} topics")
    # Print topic words
    for topic_id in set(topics):
        if topic_id != -1:
            words = [word for word, _ in topic_model.get_topic(topic_id)[:3]]
            print(f"Topic {topic_id}: {', '.join(words)}")
    return topics, topic_model
    