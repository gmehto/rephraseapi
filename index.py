from flask import Flask, jsonify, request
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import urllib.parse

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>" \
           "<form method='POST'><button>API</button></form>"


@app.route("/rephrase/<string:akey>/<string:rtext>", methods=["GET"])
def get(rtext, akey):
    key = "asdfghjklmomlkjhgfdsamumma"
    if key == akey:
        ftext = rtext
        if not ftext == "":
            model_name = 'tuner007/pegasus_paraphrase'
            torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

            def get_response(input_text, num_return_sequences):
                batch = tokenizer.prepare_seq2seq_batch([input_text], truncation=True, padding='longest', max_length=60,
                                                        return_tensors="pt").to(torch_device)
                translated = model.generate(**batch, max_length=60, num_beams=10,
                                            num_return_sequences=num_return_sequences,
                                            temperature=1.5)
                tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
                return tgt_text

            def reapi(cont):
                context = cont
                splitter = SentenceSplitter(language='en')

                sentence_list = splitter.split(context)
                # print(sentence_list)

                paraphrase = []

                for i in sentence_list:
                    a = get_response(i, 1)
                    paraphrase.append(a)

                paraphrase2 = [' '.join(x) for x in paraphrase]
                # print(paraphrase2)

                paraphrase3 = [' '.join(x for x in paraphrase2)]
                paraphrased_text = str(paraphrase3).strip('[]').strip("'")

                return paraphrased_text

            pptext = reapi(urllib.parse.unquote(ftext))

            pass
        dic = [
            {"Rephrased_text": pptext}
        ]
        return jsonify({'text': dic})
    else:
        return jsonify({"Error": "Wrong api key"})




if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
