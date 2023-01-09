import tensorflow_hub as hub
import tensorflow_text as text

preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"

bert_preprocess_model = hub.KerasLayer(preprocess_url)

text_test = ["nice movie indeed","I love python programming"]
text_preprocessed = bert_preprocess_model(text_test)

bert_model = hub.KerasLayer(encoder_url)
bert_results = bert_model(text_preprocessed)

print(bert_results)

