from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res1 = classifier("I've been waiting for a HuggingFace course my whole life.")
res2 = classifier("I hate this so much!")
res3 = classifier("I dont know what I think about this")

print(res1)
print(res2)
print(res3)

sum_res = classifier(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!", "I dont know what I think about this"])
print(sum_res)