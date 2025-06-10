import spacy

nlp = spacy.load("en_core_web_sm")

def extract_claim(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]  

input_text = "The Indian government has announced free electricity to all farmers starting July 2025."
claims = extract_claim(input_text)
print(claims)
