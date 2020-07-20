import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

def word_predict(text):
    
    pretrained_model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"

    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name)
    tokenized_text = tokenizer.tokenize(text)

    if "[MASK]" in tokenized_text:
        masked_indices = [i for i, x in enumerate(tokenized_text) if x == '[MASK]']

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])

    model = BertForMaskedLM.from_pretrained(pretrained_model_name)
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(tokens_tensor)
        for masked_index in masked_indices:
            predictions = outputs[0][0, masked_index].topk(1) # 予測結果を1件取得

            for i, index_t in enumerate(predictions.indices):
                index = index_t.item()
                token = tokenizer.convert_ids_to_tokens([index])[0]
                tokenized_text[masked_index] = token
    
    text = ""
    for token in tokenized_text:
        if token.startswith("##"):
            text = text + token.strip("#")
        else :
            text = text + token

    return(text)


if __name__ == "__main__":
    text =  "最大全長は[MASK][MASK][MASK]、最大体重は[MASK][MASK][MASK]と、現在まで報告されている獣脚類の中で史上最大級の体格を誇る種の[MASK][MASK]に数えられており、古今東西を通じて最大最強と名高い肉食動物でもある。"
    tokenized_text = word_predict(text=text)
    print(tokenized_text)
