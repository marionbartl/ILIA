import time
from datetime import datetime
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import stanza
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm


class Timer:
    def __init__(self):
        self.start_var = None
        self.end_var = None

    def convert_to_preferred_format(self):
        sec = (self.end_var - self.start_var) % (24 * 3600)
        hour = sec // 3600
        sec %= 3600
        min = sec // 60
        sec %= 60
        # return "%02d:%02d:%02d" % (hour, min, sec)
        return hour, min, sec

    def start(self):
        self.start_var = time.time()

    def stop(self, verbosity=True):
        self.end_var = time.time()
        if verbosity:
            h, m, s = self.convert_to_preferred_format()
            # print('processing took {0:0.0f} minutes and {1:0.0f} seconds'.format(m, s))
            return "processing time: {0:02d}:{1:02d}:{2:02d}".format(int(h), int(m), int(s))
        else:
            return self.end_var - self.start_var


def timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H:%M")


import os


def convert_bytes(num):
    """
    from https://stackoverflow.com/questions/2104080/how-do-i-check-file-size-in-python
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["B", "K", "M", "G", "T"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    from https://stackoverflow.com/questions/2104080/how-do-i-check-file-size-in-python
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)
    else:
        return None


def text_processing(raw_text):
    nlp = stanza.Pipeline(
        "en", processors="tokenize", use_gpu=True, pos_batch_size=3000, logging_level="FATAL"
    )

    parsed_doc = nlp(raw_text)

    tokenized_text = [[word.text for word in sent.tokens] for sent in parsed_doc.sentences]
    no_tokens = sum([len(sent) for sent in tokenized_text])

    return tokenized_text, no_tokens


def logging(text, file):
    with open(file, "a+") as f:
        f.write(text + "\n")


def save_w2v(model, model_dir, model_name):
    """function to save a word2vec model in three different modes:
    1. as gensim model
    2. just the vectors as TXT document
    3. the vectors as binary document"""
    # create a model path
    model_path = os.path.join(model_dir, model_name)
    # save the model
    model.save(model_path + ".model")
    # get just the vectors and save them as txt and binary
    word_vectors = model.wv
    word_vectors.save_word2vec_format(model_path + ".txt", binary=False)
    word_vectors.save_word2vec_format(model_path + ".bin", binary=True)


def big_num_to_string(num):
    # get the length of the number
    num = int(num)
    length = len(str(num))
    if length > 9:
        return "{0:.3f}".format(num / 10**9).rstrip("0").rstrip(".") + "B"
    elif length > 6:
        return "{0:.3f}".format(num / 10**6).rstrip("0").rstrip(".") + "M"
    elif length > 3:
        return "{0:.3f}".format(num / 10**3).rstrip("0").rstrip(".") + "K"
    else:
        return str(num)


def corpus_stats(corpus_path):
    line_counter = 0
    line_counts = []
    word_counter = 0
    word_counter_per = dict()

    for sub_corpus in os.listdir(corpus_path):
        print(f"processing {sub_corpus} now")
        # neutral copy-over
        word_counter_per[sub_corpus] = 0
        for fname in os.listdir(os.path.join(corpus_path, sub_corpus)):
            file_line_counter = 0
            with open(os.path.join(corpus_path, sub_corpus, fname), mode="r") as in_file:

                for line in in_file:
                    line_counter += 1
                    file_line_counter += 1
                    num_words = len(line.split())
                    word_counter += num_words
                    word_counter_per[sub_corpus] += num_words

            line_counts.append(file_line_counter)

    print(f"this many lines are in the corpus: {line_counter}")
    print(f"this many lines holds a single file on average: {int(np.mean(line_counts))}")
    print(f"this many words in overall corpus: {word_counter}")
    print(f"distribution per corpus:\n{word_counter_per}")


def visualize_process_time(file="logs/processing_time10K.csv"):
    df = pd.read_csv("logs/processing_time10K.csv")
    # add up seconds by row
    df["seconds"] = df["seconds"].cumsum()

    # create graphic that shows relationship between number of tokens and processing time
    plt.scatter(df["tokens"], df["seconds"])
    plt.xlabel("number of tokens")
    plt.ylabel("processing time in seconds")
    plt.title("Processing time for 10K tokens")
    plt.show()


def calculate_perplexity(model_path, dataset_id):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"device info:\n{torch.cuda.get_device_properties(0)}")

    model_id = model_path
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # # from https://huggingface.co/docs/transformers/en/perplexity
    # model_id = "openai-community/gpt2-large"
    # model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    # # Download from datasets library
    if dataset_id == "wikitext":
        test = load_dataset(dataset_id, "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    else:
        # Work with local files
        data_files = {"test": dataset_id}
        test = load_dataset("data/fine-tuning/", data_files=data_files)
        encodings = tokenizer("\n\n".join(test["test"]["text"]), return_tensors="pt")

    print("data info:")
    print(encodings.keys())
    print(encodings.input_ids.shape)
    print("data loaded")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    t = Timer()

    t.start()

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    print(t.stop())

    print(f"model: {model_id}")
    print(f"dataset: {dataset_id}")
    print(f"perplexity: {ppl.item()}")

    return
