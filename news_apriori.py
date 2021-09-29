import networkx as nx
import itertools
from apyori import apriori

def make_NounPos_byMecab(df_context, stop_words):
    pos = []
    for line in df_context:
        tokenizer = MeCab.Tagger()
        parsed = tokenizer.parse(line)
        word_tag = [w for w in parsed.split('\n')]

        pos1 = []
        tags = ['NNG', 'NNP']
        stop_words = stop_words
        for word_ in word_tag[:-2]:
            word = word_.split('\t')
            tag = word[1].split(',')

            if(len(word[0]) < 2):
               continue
            else :
               if(tag[0] in tags):
                    if word[0] not in stop_words:
                        pos1.append(word[0])
        pos.append(pos1)
        
    new_pos = []
    for group in pos:
        group_ = list(set(group))
        new_pos.append(group_)
    
    return new_pos

def apriori_bynotY(df_anal, stop_words, min_support):
    
    df_context = df_anal['제목']

    pos = make_NounPos_byMecab(df_context, stop_words)

    result = list(apriori(pos, min_support = min_support, max_length = 2))

    df_apr = pd.DataFrame(result) 
    df_apr['length'] = df_apr['items'].apply(lambda x: len(x))
    df_apr2 = df_apr.loc[(df_apr['length'] == 2)&
                         (df_apr['support'] >= min_support)].sort_values(by = 'support', ascending = False)
    print(df_apr2.shape)

    node_list = df_apr2['items']
    node_list = [list(x) for x in node_list]
    # node_list
    weight_list = list(df_apr2['support'])
    weight_list_r = []
    for w in weight_list:
        w2 = round(w, 5)
        weight_list_r.append(w2)
        # weight_list_r
        
    return node_list, weight_list_r

def apriori_byY(df_anal, year, stop_words=None, min_support=0.005):
    df_context = df_anal.loc[df_anal['year']==f'{year}', '제목']
    
    pos = make_NounPos_byMecab(df_context, stop_words)
    
    result = list(apriori(pos, min_support = min_support, max_length = 2))

    df_apr = pd.DataFrame(result) 
    df_apr['length'] = df_apr['items'].apply(lambda x: len(x))
    df_apr2 = df_apr.loc[(df_apr['length'] == 2)&
                         (df_apr['support'] >= min_support)].sort_values(by = 'support', ascending = False)
    print(df_apr2.shape)
    
    node_list = df_apr2['items']
    node_list = [list(x) for x in node_list]
    # node_list
    weight_list = list(df_apr2['support'])
    weight_list_r = []
    for w in weight_list:
        w2 = round(w, 5)
        weight_list_r.append(w2)
        
    return node_list, weight_list_r

def apriori_to_excel(df_anal, stop_words, min_support, year=None):
    
    node_list, weight_list = apriori_bynotY(df_anal, stop_words, min_support)
    
    df_meltset = pd.DataFrame(columns=[['node0','node1','support']])
    node0_list = []
    node1_list = []
    support_list = []
    for i, node in enumerate(node_list):
        node0_list.append(node[0])
        node1_list.append(node[1])
        support_list.append(weight_list[i])
    df_meltset['node0'] = node0_list
    df_meltset['node1'] = node1_list
    df_meltset['support'] = support_list

    df_meltset.to_excel(f'./apriori_{year}_total.xlsx')
    

def make_networkGraph(df_anal, stop_words, min_support, filename):
    
    node_list, weight_list = apriori_bynotY(df_anal, stop_words, min_support)
    
    G_ec = nx.Graph()
    for i, node in enumerate(node_list):
        G_ec.add_edge(node[0], node[1])
        G_ec.add_edge(node[0], node[1], weight = weight_list[i])
    
    print(G_ec.number_of_nodes())
    nx.write_gexf(G_ec, f"./{filename}_network.gexf")

    
def make_networkGraph_byKeyword(df_anal, stop_words, min_support, filename, keywords_list):
    
    keyword = keywords_list[0]
    node_list, weight_list = apriori_bynotY(df_anal, stop_words, min_support)
    
    G_ec = nx.Graph()
    for i, node in enumerate(node_list):
        if (node[0] in keywords_list) or (node[1] in keywords_list):
            G_ec.add_edge(node[0], node[1])
            G_ec.add_edge(node[0], node[1], weight = weight_list[i])
    
    print(G_ec.number_of_nodes())
    nx.write_gexf(G_ec, f"./{filename}_network_for_{keyword}.gexf")