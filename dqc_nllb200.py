import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu as sentence_similarity
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

vectorizer = TfidfVectorizer()
smoothie = SmoothingFunction().method4

TASK = "translation"
MODEL = "facebook/nllb-200-distilled-600M"
LANGS = ["Kinyarwanda","English","French","Swahili","Luganda","Lingala"]
LANGUAGES={
    "english": "eng_Latn",
    "kinyarwanda": "kin_Latn",
    "swahili": "swh_Latn",
    "luganda": "lug_Latn",
    "lingala": "lin_Latn",
    "french": "fra_Latn"
}

df = pd.read_csv("/content/data.csv")

device = 0 if torch.cuda.is_available() else -1
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def choose_language(language): 
    result = LANGUAGES.get(language.lower())
    if result:
        return result        
    return

def translate(text, source_lang, target_lang, max_length=2000):
    """
    Translate text from source language to target language
    """
    src_lang = choose_language(source_lang)
    tgt_lang= choose_language(target_lang)
    if src_lang==None:
        return "Error: the source langage is incorrect"
    elif tgt_lang==None:
        return "Error: the target language is incorrect"

    translation_pipeline = pipeline(TASK,
                                    model=model,
                                    tokenizer=tokenizer,
                                    src_lang=src_lang,
                                    tgt_lang=tgt_lang,
                                    max_length=max_length,
                                    device=device)
    result = translation_pipeline(text)
    return result[0]['translation_text']

def backtranslate(text, source_lang, target_lang):
    translation = translate(text, source_lang, target_lang)
    back_translation = translate(translation, target_lang, source_lang)
    return back_translation

def bleuscore_similarity_score(text1, text2):
  similarity_score = sentence_similarity(text1, text2, smoothing_function=smoothie)
  return similarity_score

def cosine_similarity_score(text1, text2):
  tfidf = vectorizer.fit_transform([text1, text2])
  cosine_sim = cosine_similarity(tfidf[0], tfidf[1])
  return cosine_sim

def single_language(df, cols_name, languages, size=1000):
  df_ = df[:size]
  src_bt = []
  tgt_bt = []
  src_bleu_score = []
  tgt_bleu_score = []
  src_cs_score = []
  tgt_cs_score = []

  for index, row in df_[:].iterrows():
    row = row.astype(str)

    # back translating the first language
    bt_text = backtranslate(row[cols_name[0]], languages[0], languages[1])
    src_bt.append(bt_text)
    bleu = bleuscore_similarity_score(row[cols_name[0]], bt_text) 
    src_bleu_score.append(bleu)
    cs = cosine_similarity_score(row[cols_name[0]], bt_text)
    src_cs_score.append(cs[0][0])
    # print(f"bleu: {bleu} \tcs: {cs} \tbt: {bt_text} \toriginal: {row[cols_name[0]]}\n")

    # back translating the second language
    bt_text = backtranslate(row[cols_name[1]], languages[1], languages[0])
    tgt_bt.append(bt_text)
    bleu = bleuscore_similarity_score(row[cols_name[1]], bt_text) 
    tgt_bleu_score.append(bleu)
    cs = cosine_similarity_score(row[cols_name[1]], bt_text)
    tgt_cs_score.append(cs[0][0])
    # print(f"bleu: {bleu} \tcs: {cs} \tbt: {bt_text} \toriginal: {row[cols_name[1]]}\n\n")

    if float(index/50)==0.0:
      print(f"Index: {index}")

  df_[cols_name[0]+"_nllb_backtranslation"] = src_bt
  df_[cols_name[1]+"_nllb_backtranslation"] = tgt_bt
  df_[cols_name[0]+"_nllb_bleuscore"] = src_bleu_score
  df_[cols_name[1]+"_nllb_bleuscore"] = tgt_bleu_score
  df_[cols_name[0]+"_nllb_cosinesimilarity"] = src_cs_score
  df_[cols_name[1]+"_nllb_cosinesimilarity"] = tgt_cs_score
  return df_

if __name__=="__main__":
  new_df = single_language(df, ["en","rw"], ["English", "Kinyarwanda"], size=100)
  new_df.to_csv("/content/data-test.csv")
