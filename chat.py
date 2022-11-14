import urllib.parse
import urllib
import json
import numpy as np
import pickle
import chitchat_dataset as ccc
import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder

LR = pickle.load(open('models/LRmodel.pkl','rb')) 
model_embed = pickle.load(open('Pickled_Data/model_4embed.pkl','rb'))
message_dataset = list(ccc.MessageDataset())
pca_model = pickle.load(open('Pickled_Data/pca.pkl','rb'))
embedding_1 = np.load('Pickled_Data/chitchat_embeddings.npy')
comments_data = pickle.load(open('Pickled_Data/embeddings.pkl','rb'))
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def possible_soln(q, topic):
  query = "title:"+q+" comments:"+q
  query_embed = model_embed.encode(q,convert_to_tensor=True) 
  preds = LR.predict(query_embed.reshape(1,384)) 
  print(preds)
  returnres = {}
  compr = pca_model.transform(query_embed.reshape(1,384))
  returnres['question']=q
  returnres['query'] = [compr[0][0],compr[0][1],compr[0][2]]
  returnres['preds'] = preds[0]
  if preds[0]==1:
    df,m = chitchat(query_embed)
    hits = df.sen.tolist()
    cross_inp = [[q, hit] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    df['cross_scores'] = cross_scores
    final_response = df.sort_values(by=['cross_scores'],ascending=False)
    returnres['message_response'] = final_response.iloc[0].sen
    print(30*"-+-")
    returnres['sim_msg'] = final_response.sen.values.tolist()
    embed_res = np.array(final_response.embed.values.tolist())
    embed_cos = pca_model.transform(embed_res)
    returnres['cos_embed'] = embed_cos
    # print(embed_cos.shape )
    return returnres

  else:
    df, m = reddit_index(q,query_embed, topic)
    if type(df) is not list:
      returnres['message_response'] = m
      returnres['sim_msg'] = df.sen.values.tolist()
      embed_res = np.array(df.embed.values.tolist())
      embed_cos = pca_model.transform(embed_res)
      returnres['cos_embed'] = embed_cos
      print(30*"+===+")
      return returnres
    else:
      returnres['message_response'] = m
      return returnres

  
  # return m

def reddit_index(q,query_embed, topic):
  query = "title:"+q+" comments:"+q
  filter1 = '' if topic=="All" else "fq=topic:"+topic
  d1 = {'cos_score':[],'l1':[],'sen':[],'embed':[]}

  url = "http://34.134.85.191:8983/solr/IRF22Proj/query?q="+urllib.parse.quote(query)+"&q.op=OR&"+filter1+"&indent=true&fl=id,score,topic,*&wt=json&rows=10"
  print(url)
  data = urllib.request.urlopen(url)
  m = json.load(data)['response']
  d = []
  if len(m['docs'])!=0:
    for i in range(len(m['docs'])):
      title = m['docs'][i]['title'] 
      print(m['docs'][i]['index'])
      selected = comments_data.loc[comments_data['index']==int(m['docs'][i]['index'])]
      sentences = selected.comments.values[0]
      embedding_2 = selected.embeddings.values[0]
      print(embedding_2.shape)
      embedding_3 = query_embed
      m2 = util.pytorch_cos_sim(embedding_3, embedding_2) #user query, answer
      preds, inds = m2.topk(2, dim=1)
      d1['cos_score'] = d1['cos_score']+[preds.numpy()[0][0],preds.numpy()[0][1]] 
      d1['l1'] = d1['l1']+[inds.numpy()[0][0],inds.numpy()[0][1]]
      # print(preds, inds)
      d1['sen'] =  d1['sen'] + [np.array(sentences)[inds.numpy()[0]][0],np.array(sentences)[inds.numpy()[0]][1]]
      d1['embed'] = d1['embed'] + [embedding_2[inds.numpy()[0]][0],embedding_2[inds.numpy()[0]][1]]
      # d.append(res)
    # print(sorted(d, key=lambda d: d['score'], reverse=True))
    df = pd.DataFrame(d1).sort_values(by=['cos_score'],ascending=False).head(5)
    hits = df.sen.tolist()
    cross_inp = [[q, hit] for hit in hits]
    print(cross_inp)
    cross_scores = cross_encoder.predict(cross_inp)
    df['cross_scores'] = cross_scores
    print(5*"+")
    final_response = df.sort_values(by=['cross_scores'],ascending=False)
    m1 = final_response.iloc[0].sen
    # print(df.sort_values(by=['cos_score'],ascending=False))
    return final_response,m1
  else:
    m1="Sorry, I cannot understand"
    return d,m1


def chitchat(embedding_3):
  m1 = util.pytorch_cos_sim(embedding_3, embedding_1)
  preds, inds = m1.topk(2, dim=1)
  print(preds,inds)
  d1 = {'cos_score':[],'l1':[],'sen':[],'embed':[]}
  d1['cos_score'] = [preds.numpy()[0][0],preds.numpy()[0][1]]
  d1['l1'] = [inds.numpy()[0][0],inds.numpy()[0][1]]
  d1['sen'] = [message_dataset[inds.numpy().tolist()[0][0]+1],message_dataset[inds.numpy().tolist()[0][1]+1] ]
  d1['embed'] =  [embedding_1[inds.numpy().tolist()[0][0]+1],embedding_1[inds.numpy().tolist()[0][1]+1] ]
  df = pd.DataFrame(d1)
  print(df)
  return df,d1['sen']
  ## message, preds, 

  # d1 = {'cos_score':[],'l1':[],'sen':[]}
  # print(urllib.parse.quote(query))
  # filter1 = '' if topic=="All" else "fq=topic:"+topic
  # url = "http://34.134.85.191:8983/solr/IRF22Proj/query?q="+urllib.parse.quote(query)+"&q.op=OR&"+filter1+"&indent=true&fl=id,score,topic,*&wt=json&rows=20"
  # print(url)
  # data = urllib.request.urlopen(url)
  # m = json.load(data)['response']