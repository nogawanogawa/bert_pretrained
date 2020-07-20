import spacy
from spacy import displacy
nlp = spacy.load('ja_ginza')

def mask_quantity(s):
    doc = nlp(s) 
    
    result = []

    for sent in doc.sents:
        for token in sent:
            if token._.ne.endswith('QUANTITY'):
                result.append("[MASK]") 
            else:
                result.append(token.orth_) 

    text = ""
    for token in result:
        text = text + token

    return text

if __name__ == "__main__":
    text = "最大全長は約13メートル、最大体重は約9トンと、現在まで報告されている獣脚類の中で史上最大級の体格を誇る種の一つに数えられており、古今東西を通じて最大最強と名高い肉食動物でもある。"
    masked_text = mask_quantity(text)
    print(masked_text)