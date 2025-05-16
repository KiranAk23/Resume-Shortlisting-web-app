import spacy
import logging

# Set up logging
logging.basicConfig(
    filename='test_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    logging.info("Loading spaCy model without excluding transformer...")
    model_path = r"D:/Phase1/Final_project/Resume_ranking/MODELS/spacy_model/model2/model-last"
    nlp = spacy.load(model_path)  # Do NOT exclude transformer
    logging.info("spaCy model loaded with components: %s", nlp.pipe_names)

    # Test the model on sample text
    text = "John Doe is a Python Backend Developer with 3 years of experience."
    doc = nlp(text)
    logging.info("Sample text processed: %s", text)
    for ent in doc.ents:
        logging.info("Entity: %s, Label: %s", ent.text, ent.label_)

    # Test tokenization
    tokens = [token.text for token in doc]
    logging.info("Tokens: %s", tokens)

    # Test if vocab is valid
    if not hasattr(nlp, 'vocab') or nlp.vocab is None:
        logging.error("spaCy model has no valid vocab attribute")
        raise ValueError("spaCy model has no valid vocab attribute")
    logging.info("Vocab is valid")

except Exception as e:
    logging.error(f"Error loading or using spaCy model: {str(e)}")
    raise