from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res1 = classifier("I've been waiting for a HuggingFace course my whole life.")
res2 = classifier("I hate this so much!")
res3 = classifier("I dont know what I think about this")

print(res1)
print(res2)
print(res3)

sum_res = classifier(["I've been waiting for a HuggingFace course my whole life.",
                    "I hate this so much!", "I dont know what I think about this"])
print(sum_res)



## Test danish models
tokenizer_id = AutoTokenizer.from_pretrained("Maltehb/-l-ctra-danish-electra-small-cased-ner-dane")
model_id = AutoModelForTokenClassification.from_pretrained("Maltehb/-l-ctra-danish-electra-small-cased-ner-dane")

classifier = pipeline("sentiment-analysis", model = model_id, tokenizer = tokenizer_id)

dk1 = classifier("Jeg har ventet på en HuggingFace kursus hele mit liv.")
dk2 = classifier("jeg kan rigtigt godt lide makrelmadder")
dk3 = classifier("jeg kan ikke lide makrelmadder")
dk4 = classifier("jeg synes jeg lærer meget hver dag")

for each in [dk1, dk2, dk3, dk4]:
    print(each)



## Test danish models

tokenizer_id = AutoTokenizer.from_pretrained("danish-sentiment-analysis-model")
model_id = AutoModelForSequenceClassification.from_pretrained("danish-sentiment-analysis-model")

classifier = pipeline("sentiment-analysis", model = model_id, tokenizer = tokenizer_id)

dk1 = classifier("Jeg har ventet på en HuggingFace kursus hele mit liv.")
dk2 = classifier("jeg kan rigtigt godt lide makrelmadder")
dk3 = classifier("jeg kan ikke lide makrelmadder")
dk4 = classifier("jeg synes jeg lærer meget hver dag")

for each in [dk1, dk2, dk3, dk4]:
    print(each)
