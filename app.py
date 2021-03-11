import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly as px
import networkx as nx

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from tqdm import tqdm
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve

st.set_option('deprecation.showPyplotGlobalUse', False)

# #Set title
st.title("Community Detection & Link Prediction in Social Networks")

#set subtitle

st.write("""
### Exploratory Data Analysis
""")

# dataset_name=st.sidebar.selectbox('Select dataset',('fb-pages-nodes','fb-pages-edges'))

with open("fb-pages-nodes.csv",encoding="utf8") as f:
    fb_nodes = f.read().splitlines() 

# load edges (or links)
with open("fb-pages-edges.csv",encoding="utf8") as f:
    fb_links = f.read().splitlines()

def get_datalength():
  return len(fb_nodes), len(fb_links)

# def get_dataset(name):
#     data=None
#     if name=='fb-pages-nodes':
#         data=datasets.load_fb-pages-nodes()
#     else:
#         data=datasets.load_fb-pages-edges()

#     x=data.data
#     y=data.target
    
#     return x,y

# x,y=get_dataset(dataset_name)
# st.dataframe(x)
length=get_datalength()
st.write('Datalength:',length)
node_list_1 = []
node_list_2 = []

for i in tqdm(fb_links):
    node_list_1.append(i.split(',')[0])
    node_list_2.append(i.split(',')[1])

fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})

def get_dataframe():
 
    return fb_df.head(),fb_df

dataframedata,fb_df=get_dataframe()
st.write('Datadataframedata:',dataframedata)

#set subtitle

st.write("""
### Create a Graph
""")

G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())

def get_graph():
        # create graph

    # plot graph
    plt.figure(figsize=(50,50))

    pos = nx.random_layout(G, seed=23)
    nx.draw(G, with_labels=True,  pos = pos,label = 'Restaurants', node_size = 5000, alpha = 0.6, width = 0.7)

    plt.legend(loc=0, prop={'size': 50},scatterpoints = 1,numpoints = 2)

    return plt.show()

showgraph=get_graph()
# st.write('ShowGraph:',showgraph)
st.pyplot(showgraph)

#set subtitle

st.write("""
## Community Detection
""")

st.write("""
### Girvan-Newmann Approach
""")

def edge_to_remove(graph):
    
    G_dict = nx.edge_betweenness_centrality(graph)

    edge = ()

    # extract the edge with highest edge betweenness centrality score

    for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse = True):
        
        edge = key
            
        break

    return edge

G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())
def girvan_newman(graph):
    
    # find number of connected components
    
    sg = nx.connected_components(graph)
    
    sg_count = nx.number_connected_components(graph)

    while(sg_count == 1):
        
        graph.remove_edge(edge_to_remove(graph)[0], edge_to_remove(graph)[1])
        
        sg = nx.connected_components(graph)
        
        sg_count = nx.number_connected_components(graph)

    return sg
# find communities in the graph
c = girvan_newman(G.copy())

# find the nodes forming the communities
node_groups = []

for i in c:
  node_groups.append(list(i))
st.write('Allnodes:',node_groups)


#set subtitle

st.write("""
### Plotting the communities
""")

pos = nx.random_layout(G, seed=23)
def get_plot():
    plt.figure(figsize=(50,50))

    nx.draw(G,nodelist=node_groups[0],node_color='blue', label='Dine in Restaurants',   pos = pos, node_size = 5000, alpha = 0.6, width = 0.7 )       
    nx.draw(G,nodelist=node_groups[1],node_color='red', label='Fast Food(Drive thru) Restaurants',  pos = pos, node_size = 5000, alpha = 0.6, width = 0.7)

    plt.legend(loc=1, prop={'size': 50},scatterpoints = 1)

    return plt.show()

graphplot=get_plot()
# st.write('ShowPlot:',graphplot)
st.pyplot(graphplot)

#set subtitle

st.write("""
## Most Inflential nodes for marketing
""")

st.write("""
### Influential nodes based on degree centrality
""")

st.write('Top 100 Degree centrality nodes:')

centrality = nx.degree_centrality(G)
centrality_dict = dict(sorted(centrality.items(), reverse= True, key=lambda x: x[1])[:100])
st.write(centrality_dict)

blue_community = []
red_community = []
x = centrality.keys()
for node in x:
    if node in node_groups[0]:
        blue_community.append(node)
    else: 
        red_community.append(node)

st.write('Most influencing node in Blue Community according to degree centrality:')

node = centrality_dict.keys()
for n in node:
    if n in blue_community:
        st.write(n)
        st.write(centrality_dict[n])
        break;

st.write('Most influencing node in Blue Community according to degree centrality:')

node = centrality_dict.keys()
for n in node:
    if n in red_community:
        st.write(n)
        st.write(centrality_dict[n])
        break;

st.write("""
### Influential nodes based on closeness centrality
""")

st.write('Top 170 CLoseness centrality nodes:')

centrality = nx.closeness_centrality(G)
centrality_dict = dict(sorted(centrality.items(), reverse= True, key=lambda x: x[1])[:170])
st.write(centrality_dict)

blue_community = []
red_community = []
x = centrality.keys()
for node in x:
    if node in node_groups[0]:
        blue_community.append(node)
    else: 
        red_community.append(node)

st.write('Most influencing node in Blue Community according to closeness centrality:')

node = centrality_dict.keys()
for n in node:
    if n in blue_community:
        st.write(n)
        st.write(centrality_dict[n])
        break;

st.write('Most influencing node in Red Community according to closeness centrality:')

node = centrality_dict.keys()
for n in node:
    if n in red_community:
        st.write(n)
        st.write(centrality_dict[n])
        break;

st.write("""
### Influential nodes based on Betweeness Centrality
""")

st.write('Top 100 Betweenness centrality nodes:')

centrality = nx.betweenness_centrality(G)
centrality_dict = dict(sorted(centrality.items(), reverse= True, key=lambda x: x[1])[:100])
st.write(centrality_dict)

blue_community = []
red_community = []
x = centrality.keys()
for node in x:
    if node in node_groups[0]:
        blue_community.append(node)
    else: 
        red_community.append(node)

st.write('Most influencing node in Blue Community according to betweeness centrality:')

node = centrality_dict.keys()
for n in node:
    if n in blue_community:
        st.write(n)
        st.write(centrality_dict[n])
        break;

st.write('Most influencing node in Red Community according to betweeness centrality:')

node = centrality_dict.keys()
for n in node:
    if n in red_community:
        st.write(n)
        st.write(centrality_dict[n])
        break;

st.write("""
### Link Prediction
""")

st.write('Create an adjacency matrix to find unconnected pairs')

# combine all nodes in a list
node_list = node_list_1 + node_list_2

# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))

# build adjacency matrix
adj_G = nx.to_numpy_matrix(G, nodelist = node_list)
def get_adj_mat():
 
    return adj_G.shape

shape=get_adj_mat()
st.write('shape:',shape)

st.write("""
### Get all unconnected pairs - Negative Samples
""")

# get unconnected node-pairs
all_unconnected_pairs = []

# traverse adjacency matrix
offset = 0
for i in tqdm(range(adj_G.shape[0])):
  for j in range(offset,adj_G.shape[1]):
    if i != j:
      if nx.shortest_path_length(G, str(i), str(j)) <=2:
        if adj_G[i,j] == 0:
          all_unconnected_pairs.append([node_list[i],node_list[j]])

  offset = offset + 1

def get_unconn():

    return len(all_unconnected_pairs)

unconndata=get_unconn()
st.write('Unconnected Pairs:',unconndata)

node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]

data = pd.DataFrame({'node_1':node_1_unlinked, 
                     'node_2':node_2_unlinked})

# add target variable 'link'
data['link'] = 0

st.write("""
### Remove Links from Connected Node Pairs – Positive Samples
""")

initial_node_count = len(G.nodes)

fb_df_temp = fb_df.copy()

# empty list to store removable links
omissible_links_index = []

for i in tqdm(fb_df.index.values):
  
  # remove a node pair and build a new graph
  G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index = i), "node_1", "node_2", create_using=nx.Graph())
  
  # check there is no spliting of graph and number of nodes is same
  if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
    omissible_links_index.append(i)
    fb_df_temp = fb_df_temp.drop(index = i)

def get_conn():

        return len(omissible_links_index)

conndata=get_conn()
st.write('Connected Pairs:',conndata)

st.write("""
### Data for Model Training
""")

# create dataframe of removable edges
fb_df_ghost = fb_df.loc[omissible_links_index]

# add the target variable 'link'
fb_df_ghost['link'] = 1

data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

def get_link():
    return data['link'].value_counts()

linkdata=get_link()
st.write('Training Model:',linkdata)

st.write("""
### Feature Extraction
""")

# drop removable edges
fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

# build graph
G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

st.write('Train the node2vec model')

from node2vec import Node2Vec

# Generate walks
node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

# train node2vec model
n2w_model = node2vec.fit(window=7, min_count=1)
x = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(data['node_1'], data['node_2'])]

xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)
lr = LogisticRegression(class_weight="balanced",max_iter=1000)


st.write(lr.fit(xtrain, ytrain))

predictions = lr.predict(xtest)
preds = lr.predict_proba(xtest)

st.write('Accuracy Score:',roc_auc_score(ytest, preds[:,1]))

st.write("""
### Visualisation
""")
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics
cm =  metrics.confusion_matrix(ytest, predictions)
st.write('CM:',cm)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual links')
plt.xlabel('Predicted links')
st.pyplot(fig)

st.write("Accuracy:",metrics.accuracy_score(ytest, predictions))
st.write("Precision:",metrics.precision_score(ytest,predictions))
st.write("Recall:",metrics.recall_score(ytest, predictions))


st.subheader("ROC Curve")
st.image('data.png')
# preds = lr.predict_proba(xtest)[::,1]
# fpr, tpr, _ = metrics.roc_curve(ytest, preds)
# auc = metrics.roc_auc_score(ytest, preds)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()
# st.pyplot()

# plot_roc_curve(xtest,fpr, tpr)
# st.pyplot()


st.write("""
### Fitting it to a LightGBM model to increase efficiency
""")
import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type':'gbdt',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}

# train lightGBM model
model = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=1000
                   )

y_pred = model.predict(xtest)
for i in range(len(y_pred)):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:
       y_pred[i]=0
cm_lgbm =  metrics.confusion_matrix(ytest, y_pred)

st.write(cm_lgbm)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm_lgbm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual links')
plt.xlabel('Predicted links')
st.pyplot(fig)

st.write("Accuracy:",metrics.accuracy_score(ytest, y_pred))
st.write("Precision:",metrics.precision_score(ytest,y_pred))
st.write("Recall:",metrics.recall_score(ytest, y_pred))