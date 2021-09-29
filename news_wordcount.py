import pandas as pd
import numpy as np
import MeCab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint
import os 
import networkx as nx
import itertools
from apyori import apriori

def xlsx_to_df(path_dir):
    
    file_list = os.listdir(path_dir)
    file_paths = []
    total_df = pd.DataFrame()
    
    for file in file_list[1:]:
        if 'xlsx' in file:
            file_path = path_dir + '/' + file

        file_paths.append(file_path)
        df = pd.read_excel(file_path)        

        total_df = total_df.append(df)
        print(total_df.shape)
    
    return total_df

## 기사 전처리(year column 생성, 중복 제거)

def preprocess_df(dataframe):
    dataframe['year'] = dataframe['일자'].apply(str).apply(lambda x:x[0:4])
    
    ## 중복기사 & 광고성 기사 제외
    print(dataframe.shape)
    dataframe.drop_duplicates(subset=['제목', '일자'], inplace=True)
    dataframe.drop_duplicates(subset=['제목','언론사'], inplace=True)
    print(dataframe.shape)
    
    return dataframe

## anlysis용 df 추출
def extract_df(dataframe):
    df_anal = dataframe[['year', '제목', '본문']]
    df_anal.reset_index(drop=True, inplace=True)
    
    return df_anal

## anlysis용 df 주제 기준 추출
def extract_df_byKeywords(df_org, keyword_list):
    
    kwrd = keyword_list[0]
    for kw in keyword_list[1:]:
        kwrd = kwrd + '|' + kw
    
    df_result = df_org.loc[(df_org['제목'].str.contains(kwrd))|
                           (df_org['본문'].str.contains(kwrd)), ['제목', '본문', 'year']]
    
    return df_result

## 형태소 분석기

def getNVM_lemma(text):
    tokenizer = MeCab.Tagger()
    parsed = tokenizer.parse(text)
    word_tag = [w for w in parsed.split('\n')]
    
    pos = []
    tags = ['NNG', 'NNP']
    for word_ in word_tag[:-2]:
        word = word_.split('\t')
        tag = word[1].split(',')
        
        if(len(word[0]) < 2):
            continue
#         if(tag[-1] != '*'):
#             t = tag[-1].split('/')
#             if(len(t[0]) > 1 and ('VV' in t[1] or 'VA' in t[1] or 'VX' in t[1])):
#                 pos.append(t[0])
        else :
            if(tag[0] in tags):
                pos.append(word[0])
    return pos


## thesaurus로 키워드 변경

def str_replace(dataframe, thesaurus_dir):
    dataframe = dataframe.apply(str)
    dataframe = dataframe.apply(lambda x:x.strip().lower())
    
    thesaurus_df = pd.read_excel(thesaurus_dir)
    from_list = thesaurus_df['text_from']
    to_list = thesaurus_df['text_to']
    
    for i, text in enumerate(from_list):
        dataframe=dataframe.str.replace(text, to_list[i])
    
    return dataframe


## 전체 데이터 키워드 빈도 분석 

def wordcount(df_anal, path_dir, stop_words):
    
    vectorizer = CountVectorizer(tokenizer = getNVM_lemma, min_df = 2, max_df=1.0, stop_words = stop_words)
    
    anal_context = str_replace(df_anal['제목'], path_dir)
    dtm = vectorizer.fit_transform(anal_context)

    vocab_ = dict()
    for idx, word in enumerate(vectorizer.get_feature_names()):
        vocab_[word] = dtm.getcol(idx).sum()
    words_ = sorted(vocab_.items(), key = lambda x:x[1], reverse = True)

    result_df = pd.DataFrame(columns=['키워드', '빈도'])

    result_df['키워드'] = [word[0] for word in words_[:]]
    result_df['빈도'] = [word[1] for word in words_[:]]

    return result_df


## 전체 데이터 키워드 빈도 분석 

def wordTfidfcount(df_anal, path_dir, stop_words):
    
    vectorizer = TfidfVectorizer(tokenizer = getNVM_lemma, min_df = 2, max_df=1.0, stop_words = stop_words)
    
    anal_context = str_replace(df_anal['제목'], path_dir)
    dtm = vectorizer.fit_transform(anal_context)

    vocab_ = dict()
    for idx, word in enumerate(vectorizer.get_feature_names()):
        vocab_[word] = dtm.getcol(idx).sum()
    words_ = sorted(vocab_.items(), key = lambda x:x[1], reverse = True)

    result_df = pd.DataFrame(columns=['키워드', '빈도'])

    result_df['키워드'] = [word[0] for word in words_[:]]
    result_df['빈도'] = [word[1] for word in words_[:]]

    return result_df


## 연도별 키워드 빈도 분석

def wordcount_byY(df_, from_, to_, thesaurus_dir, stop_words):
    
    countvectorizer = CountVectorizer(tokenizer = getNVM_lemma, min_df = 2, max_df=1.0, stop_words = stop_words)
    
    total_df = pd.DataFrame()
    
    for year in range(from_,to_+1):
        anal_context = str_replace(df_.loc[df_['year']==f'{year}', '제목'], thesaurus_dir)
        dtm = countvectorizer.fit_transform(anal_context)
        
        vocab_ = dict()
        for idx, word in enumerate(countvectorizer.get_feature_names()):
            vocab_[word] = dtm.getcol(idx).sum()
        words_ = sorted(vocab_.items(), key = lambda x:x[1], reverse = True)

        result_df = pd.DataFrame(columns=[f'키워드_{year}', f'빈도_{year}'])

        result_df[f'키워드_{year}'] = [word[0] for word in words_[:]]
        result_df[f'빈도_{year}'] = [word[1] for word in words_[:]]

        total_df = pd.concat([total_df, result_df], axis=1)
        
    return total_df


def wordTfidfcount_byY(df_, from_, to_, thesaurus_dir):
    
    stop_words = ['지난해', '이후', '이번', '기자', '투데이', '머니', '사진', '무단', '배포', '금지', '연합뉴스', '뉴시스', '뉴스', '제공',
              '지디넷', '이웃추가', '디지털타임즈', '이데일리', '금융정보단말기', '아시아경제', '스포츠조선', '조선비즈', '모바일 주식',
              '모바일 경향', '모바일앱 다운로드', '전자신문', '인터넷전자신문', '앵커', '멘트']
    vectorizer = TfidfVectorizer(tokenizer = getNVM_lemma, min_df = 2, max_df=1.0, stop_words = stop_words)
    
    total_df = pd.DataFrame()
    
    for year in range(from_,to_+1):
        anal_context = str_replace(df_.loc[df_['year']==f'{year}', '제목'], thesaurus_dir)
        dtm = vectorizer.fit_transform(anal_context)
        
        vocab_ = dict()
        for idx, word in enumerate(vectorizer.get_feature_names()):
            vocab_[word] = dtm.getcol(idx).sum()
        words_ = sorted(vocab_.items(), key = lambda x:x[1], reverse = True)

        result_df = pd.DataFrame(columns=[f'키워드_{year}', f'빈도_{year}'])

        result_df[f'키워드_{year}'] = [word[0] for word in words_[:]]
        result_df[f'빈도_{year}'] = [word[1] for word in words_[:]]

        total_df = pd.concat([total_df, result_df], axis=1)
        
    return total_df


## 원하는 키워드 빈도만 추출

def extract_kwrd(wordcount_df, kwrd_list):
    
    df_result = wordcount_df.loc[wordcount_df['키워드'].isin(kwrd_list), ['키워드', '빈도']]
    df_result.reset_index(drop=True, inplace=True)
    df_result.head()
    
    return df_result

## 원하는 키워드 빈도만 추출

def extract_kwrd_byY(wordcount_df, kwrd_list):

    df_result = pd.DataFrame()
    for year in range(2010, 2022):
        df_year = wordcount_df.loc[wordcount_df[f'키워드_{year}'].isin(kwrd_list), [f'키워드_{year}',f'빈도_{year}']]
        df_year.reset_index(drop=True, inplace=True)
        df_result = pd.concat([df_result, df_year], axis=1)

    return df_result


