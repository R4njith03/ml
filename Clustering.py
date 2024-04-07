#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# show up charts when export notebooks
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset used: [Pokemon stats](https://www.kaggle.com/datasets/abcsds/pokemon)

# In[3]:


data=pd.read_csv('Pokemon.csv')
data.head()


# In[4]:


data.isna().any()


# In[5]:


data.shape


# In[6]:


data.info()


# In[9]:


types = data['Type 1'].isin(['Grass', 'Fire', 'Water']) # True for pokemon with type1 as 'Grass', 'Fire or 'Water
types


# In[11]:


data[types] # Pokemon with type1 as 'Grass', 'Fire or 'Water


# In[14]:


drop_cols=['Type 1','Type 2','Generation','Legendary','#']
pokemon=data.drop(columns=drop_cols)


# In[30]:


pokemon.head()


# # Clustering

# In[58]:


from sklearn.cluster import KMeans
# k means
kmeans = KMeans(n_clusters=3, random_state=0)
pokemon['cluster'] = kmeans.fit_predict(pokemon[['Attack', 'Defense']])
# get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
print(cen_x)
## add to df
pokemon['cen_x'] = pokemon.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
pokemon['cen_y'] = pokemon.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
# define and map colors
colors = ['blue', 'green', 'red']
pokemon['c'] = pokemon.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})


# In[29]:


pokemon.head(20)


# ### From the above we can see that Pokemon in Blue clusters have low Attack and Defense, Pokemon in Green clusters have higher Attack than Defense and Pokemon in Red clusters have higher Defence than Attack

# In[56]:


plt.figure(figsize=(10,6))
plt.scatter(pokemon.Attack, pokemon.Defense, c=pokemon.c, alpha = 0.6, s=5)
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.show()


# ## Adding Speed to the plot

# In[31]:


plt.figure(figsize=(10,6))
plt.scatter(pokemon.Attack, pokemon.Defense, c=pokemon.c, alpha = 0.6, s=pokemon.Speed)
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.show()


# ## Dividing into Training and Testing data to test accuracy

# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)


# In[47]:


kmeans=KMeans(n_clusters=3,random_state=0)
kmeans.fit(X_train[['Attack','Defense']])


# In[48]:


kmeans.predict(X_train[['Attack','Defense']])


# In[49]:


train_labels=kmeans.fit_predict(X_train[['Attack','Defense']])


# ### The silhouette score ranges from -1 to 1, where: <br>A score close to +1 indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.<br>A score close to 0 indicates that the object is on or very close to the decision boundary between two neighboring clusters.<br>A score close to -1 indicates that the object may have been assigned to the wrong cluster.

# In[54]:


from sklearn.metrics import silhouette_score
silhouette_score = silhouette_score(X_train[['Attack','Defense']], train_labels)
print(f"Silhoutte Score: {silhouette_score}")


# In[ ]:




