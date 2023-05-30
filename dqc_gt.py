import requests
import json
import string
import nltk
from nltk.translate.bleu_score import sentence_bleu as sentence_similarity
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df = pd.read_csv("/content/data-test.csv")

vectorizer = TfidfVectorizer()
smoothie = SmoothingFunction().method4

class GoogleTranslate:
    api_url = 'https://translate.googleapis.com/translate_a/single'
    client = '?client=gtx'
    dt = '&dt=t'

def translate_row(row, target_lang, source_lang):
    sl = f"&sl={source_lang}"
    tl = f"&tl={target_lang}"
    
    # Remove punctuation marks from the input text
    translator = str.maketrans('', '', string.punctuation)
    text = row['content'].translate(translator)
    
    r = requests.get(f"{GoogleTranslate.api_url}{GoogleTranslate.client}{GoogleTranslate.dt}{sl}{tl}&q={text}")
    if r.status_code == 200:
        response_data = json.loads(r.text)
        translated_text = response_data[0][0][0]
    else:
        translated_text = ""
    row['translated_text'] = translated_text
    return row

def translate(text, source_lang, target_lang):
  sl = f"&sl={source_lang}"
  tl = f"&tl={target_lang}"

  translator = str.maketrans('','', string.punctuation)
  text = text.translate(translator)
  r = requests.get(f"{GoogleTranslate.api_url}{GoogleTranslate.client}{GoogleTranslate.dt}{sl}{tl}&q={text}")
  if r.status_code == 200:
    response_data = json.loads(r.text)
    translated_text = response_data[0][0][0]
  else:
    translated_text = None
  return translated_text

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

    print(f"\t BackTranslation at Index: {index}")

  df_[cols_name[0]+"_gt_backtranslation"] = src_bt
  df_[cols_name[1]+"_gt_backtranslation"] = tgt_bt
  df_[cols_name[0]+"_gt_bleuscore"] = src_bleu_score
  df_[cols_name[1]+"_gt_bleuscore"] = tgt_bleu_score
  df_[cols_name[0]+"_gt_cosinesimilarity"] = src_cs_score
  df_[cols_name[1]+"_gt_cosinesimilarity"] = tgt_cs_score
  return df_

new_df = single_language(df, ["en","rw"], ["English", "Kinyarwanda"], size=100)
new_df.to_csv("/content/data-test.csv")
