import pandas as pd
import numpy as np
import pynndescent
import pyarrow as pa
import pyarrow.parquet as pq
import json
import warnings
"""

Example usage:
topic_model : A fitted toponymy
document_df: A pandas dataframe with document metadata

topicdb = TopicDatabase.from_toponymy(topic_model, document_df)

#== Save to and load from a file ==#
topicdb.save("my_topic_model.tdb")
topicdb = TopicDatabase.load("my_topic_model.tdb")

#== Example queries ==#
hs_vector = SentenceTransformer(example_model).encode("lorem ipsum")

# get the embeddings semantically close to a query

topicdb.q.nearby(hs_vector).embeddings()

# get the topics containing the semantic neighbours

topicdb.q.nearby(hs_vector).topics()

# an example of a very convoluted query 
# to get the documents inside the parents of
# the topics that contain the vectors nearby your query

topicdb.q.nearby(hs_vector).topics.parents.inside.documents()

"""

class RootQuery:
    """ A class for topicdb.q to infer which type of query the user is making. """
    def __init__(self, db):
        self.db = db
    
    def __getattr__(self, name):
        if hasattr(TopicQuery, name):
            return getattr(TopicQuery(self.db), name)
        elif hasattr(IndexQuery, name):
            return getattr(IndexQuery(self.db), name)
        else:
            raise AttributeError(name)
        
    def __str__(self):
        return "Empty Query"

class Query:
    def __init__(self, db, value=None):
        self.db = db
        self.value = value

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    def unwrap(self):
        return self.value

class TopicQuery(Query):
    def __init__(self, db, value=None):
        self.db = db
        self.value = value

    def join(self, topics=None):
        if topics is None:
            topics = self.value
        value = self.db.join(topics)
        return TopicQuery(self.db, value)

    def meet(self, topics=None):
        if topics is None:
            topics = self.value
        value = self.db.meet(topics)
        return TopicQuery(self.db, value)
    
    def parents(self, topics=None):
        if topics is None:
            topics = self.value
        value = self.db.parents(topics)
        return TopicQuery(self.db, value)
    
    def children(self, topics=None):
        if topics is None:
            topics = self.value
        value = self.db.children(topics)
        return TopicQuery(self.db, value)

    def inside(self, topics=None):
        if topics is None:
            topics = self.value
        value = self.db.inside(topics)
        return IndexQuery(self.db, value)
    
    def info(self, topics=None):
        if topics is None:
            topics = self.value
        value = self.db.info(topics)
        return value
    
class IndexQuery(Query):
    def __init__(self, db, value=None):
        self.db = db
        self.value = value

    def embeddings(self, indices=None):
        if indices==None:
            indices = self.value
        value = self.db.embeddings(indices)
        return value

    def optimal_join(self, indices=None, inclusion_weight=0.5):
        if indices==None:
            indices = self.value
        value = self.db.optimal_join(np.array(indices), inclusion_weight=inclusion_weight)
        return TopicQuery(self.db, value)

    def nearby(self, hs_vector):
        value = self.db.nearby(hs_vector)
        return IndexQuery(self.db, value)

    def indices(self):
        value = self.db.indices(self.value)
        return IndexQuery(self.db, value)

    def topics(self, indices=None, logic='OR'):
        if indices==None:
            indices = self.value
        value = self.db.topics(indices, logic=logic)
        return TopicQuery(self.db, value)

    def documents(self, indices=None):
        if indices==None:
            indices = self.value
        value = self.db.documents(indices)
        return value
    
    def where(self, selector, indices=None):
        if indices==None:
            indices = self.value
        value = self.db.where(indices, selector)
        return IndexQuery(self.db, value)

    
class TopicDatabase:
    """
    Storage and queryable class for the outputs of a Toponymy topic model.

    Example query: ``topicdb.q.nearby([high_space_vector]).topics().info()``

    Parameters
    ----------
    topicFrame : pd.DataFrame
        The dataframe storing information about topics.
    vectorArray :np.ndarray
        The embedding vectors of the documents.
    vectorFrame : pd.DataFrame, optional
        A dataframe of document metadata.
    reducedArray : np.ndarray, optional
        The reduced (clusterable) vectors of the documents.
    sep : str, optional (default='/')
        The character used to separate objects in paths.

    Attributes
    -------
    q : RootQuery
        The root attribute for querying the database
    """
    def __init__(
        self,
        topicFrame: pd.DataFrame,
        vectorArray: np.ndarray,
        vectorFrame: pd.DataFrame = None,
        reducedArray: np.ndarray = None,
        sep: str = "/",
    ):    
        self.topicFrame = topicFrame
        self.sep = sep
        # it's hard to get paths otherwise
        self.path = {path.split(sep)[-1]:path for path in topicFrame['path'].values}
        # store indices because pandas gives me a headache
        self.vectorFrame = vectorFrame
        self.vectorFrame['idx']=range(len(vectorFrame))
        
        ## vectorDB at home:
        self.vectorArray = vectorArray
        self.nn_index = pynndescent.NNDescent(vectorArray)

        self.reducedArray = reducedArray

    def _get_topic(self, topic):
        return self.topicFrame[self.topicFrame['uid']==topic]

    @property
    def q(self):
        return RootQuery(self)
    @property
    def qt(self):
        return TopicQuery(self)
    def qi(self):
        return IndexQuery(self)

    #= Tree methods =#
    def join_paths(self, paths):
        if len(paths)==1:
            return paths[0]
        splits = [node.split(self.sep) for node in paths]
        done = False
        i = 0
        max_depth = min(len(s) for s in splits)
        while not done:
            items = set(s[i] for s in splits)
            n_in_layer = len(items)
            if n_in_layer > 1:
                done = True
            else:
                i+=1
            if i==max_depth:
                done = True
        return self.sep.join(splits[0][:i])

    def node(self, path):
        return path.split(self.sep)[-1]

    def parent(self, node):
        return self.sep.join(node.split(self.sep)[:-1])

    #= Topic->Topic methods =#
    def join(self, topics):
        if type(topics)==str:
            topics = [topics]
        paths = [self.path[str(node)] for node in topics]
        return self.node(self.join_paths(paths))
    
    def meet(self, topics):
        # this probably requires a BFS that I don't want to figure out yet
        raise NotImplementedError

    def parents(self, topics):
        if type(topics)==str:
            topics = [topics]
        return set([self.node(self.parent(self.path[node])) for node in topics])
    
    def children(self, topics):
        children_sets = [set(x) for x in self.info(topics)['children'].to_list()]
        return set().union(*children_sets)
    
    #= Topic->Other methods =#
    def inside(self, topics):
        """ Return the indices of vectors in a list of topics """
        if type(topics)==str:
            topics = [topics]
        indices = set()
        for l in self.info(topics)['vectors'].to_list():
            for x in l:
                indices.add(x)
        return list(indices)
    
    def info(self, topics):
        if type(topics)==str:
            topics=[topics]
        return pd.concat([self.topicFrame[self.topicFrame['uid']==t] for t in topics],axis=0)
    
    #= Index Methods =#
    def embeddings(self, indices):
        return self.vectorArray[indices]
    
    def reductions(self, indices):
        if self.reducedArray is None:
            return ValueError("This topicDB doesn't have reduction data.")
        else:
            return self.reducedArray[indices]
    
    def topics(self, indices, logic="OR"):
        """ Return the topics containing a set of indices """
        nodes_list = []      
        for i in indices:
            nodes = set()
            row = self.vectorFrame.filter(regex=r"layer\d+").iloc[0]
            layer = 0
            for c in row:
                if int(c) != -1:
                    nodes.add(topic_uid((layer, c)))
                    break
                layer+=1
            nodes_list.append(nodes)
            
        if logic=="AND":
            return set().intersection(*nodes_list) 
        elif logic=="OR":
            return set().union(*nodes_list) 
        elif logic=="XOR":
            return set().union(*nodes_list) - set.intersection(*nodes_list)
        else:
            return ValueError("`logic` must equal `AND`, `OR` or `XOR`")
    
    def documents(self, indices):
        return self.vectorFrame.iloc[indices]
    
    def where(self, indices, selectors):
        if self.vectorFrame is None:
            raise AttributeError("You haven't provided a dataframe of document information to this topic database")
        df = self.vectorFrame.iloc[indices]
        for attr, value in selectors.items():
            df = df[df[attr]==value]
        return df['idx'].to_list()

    def optimal_join(self, indices, inclusion_weight=0.5):
        """
        tree: iterable of vertex paths (strings)
        indices: iterable of query indices
        inclusion_weight: float in [0,1]
        """
    
        Q = set(indices)
        qsize = len(Q)
        tree = self.topicFrame['path'].values
    
        # Precompute max depth
        depths = {v: v.count('/') for v in tree}
        max_depth = max(depths.values())
    
        best_node = None
        best_score = -1
    
        alpha = inclusion_weight
    
        for v in tree:
            Iv = self.inside(self.node(v))
            if Iv.size==0:
                continue
    
            Iv = set(Iv)
    
            # Inclusion score
            overlap = len(Q & Iv)
            if overlap == 0:
                continue  # multiplicative form kills it anyway
    
            coverage = overlap / qsize
    
            # Depth score
            depth_score = depths[v] / max_depth
    
            # Multiplicative tradeoff
            score = (coverage ** alpha) * (depth_score ** (1 - alpha))
    
            if score > best_score: 
                best_score = score
                best_node = v
    
        return self.node(best_node)
        
    #= Vector Methods =#
    def nearby(self, hs_vector):
        nbhrs, dists = self.nn_index.query(hs_vector)
        return nbhrs.flatten()


    #= Saving and loading =#
    def save(self, path:str):
        """
        Save the topic database to a (parquet) file at `path`.
        """
        # Serialize dataframes
        topicFrame = serialize_df(self.topicFrame)
        vectorFrame = serialize_df(self.vectorFrame)
        
        # Convert vectorArrays to DataFrame for Arrow
        if self.reducedArray is None:
            vectors = self.vectorArray
        else:
            vectors = np.concat([self.vectorArray, self.reducedArray], axis=1)      
        vectorArrayDF = pd.DataFrame(vectors)
        # Create Arrow tables
        table_topic = pa.Table.from_pandas(topicFrame, preserve_index=False)
        table_vector = pa.Table.from_pandas(vectorFrame, preserve_index=False)
        table_array = pa.Table.from_pandas(vectorArrayDF, preserve_index=False)

        # Concatenate tables vertically and store a column indicating the type
        table_topic = table_topic.append_column("__table_type__", pa.array(["topicFrame"]*len(table_topic)))
        table_vector = table_vector.append_column("__table_type__", pa.array(["vectorFrame"]*len(table_vector)))
        table_array = table_array.append_column("__table_type__", pa.array(["vectorArray"]*len(table_array)))

        full_table = pa.concat_tables([table_topic, table_vector, table_array], promote=True)

        # Metadata
        metadata = {
            "serial_version": "0.2",
            "n_features":str(self.vectorArray.shape[1]),
            "n_reduced":0 if self.reducedArray is None else str(self.reducedArray.shape[1]),
            "sep": self.sep,
        }

        # Check size of metadata
        for key, value in metadata.items():
            if len(value.encode("utf-8")) > 64000:
                warnings.warn(f"Metadata '{key}' exceeds 64 KB. Parquet may truncate it!")

        # Convert metadata to bytes for Arrow
        pa_metadata = {k: v.encode("utf-8") for k, v in metadata.items()}
        full_table = full_table.replace_schema_metadata(pa_metadata)

        # Write single Parquet file
        pq.write_table(full_table, path)


    @classmethod
    def load(cls, path):
        """
        Load a topic database from `path`.
        """
        # Read table with metadata
        table = pq.read_table(path)
        df = table.to_pandas()
        
        # Extract metadata
        meta = table.schema.metadata or {}
        # why not use bytes? ¯\_(ツ)_/¯
        version = meta[b'serial_version'].decode('utf-8')
        if version != "0.2":
            raise ValueError(f"You need to convert the saved data from v{version} to v0.2")
        sep = meta[b'sep'].decode('utf-8')
        n_features = int(meta[b'n_features'].decode('utf-8'))
        n_reduced = int(meta[b'n_reduced'].decode('utf-8'))
        
        # Split tables by type
        topicFrame = df[df["__table_type__"] == "topicFrame"].drop(columns="__table_type__").dropna(axis=1, how='all')
        vectorFrame = df[df["__table_type__"] == "vectorFrame"].drop(columns="__table_type__").dropna(axis=1, how='all')
        vectorArrayDF = df[df["__table_type__"] == "vectorArray"].drop(columns="__table_type__").dropna(axis=1, how='all')
        vectors = vectorArrayDF.to_numpy()
        reducedArray = vectors[:, -n_reduced:]
        vectorArray = vectors[:,0:n_features]
        
        # Deserialize object columns
        topicFrame = deserialize_df(topicFrame)
        vectorFrame = deserialize_df(vectorFrame)
        
        return cls(topicFrame, vectorArray, vectorFrame, reducedArray, sep=sep)
    
    @classmethod 
    def from_toponymy(cls, topic_model, document_df: pd.DataFrame=None):
        """
        Generate a TopicDatabase from a fitted Toponymy. 

        Parameters
        ----------
        topic_model: Toponymy
            The toponymy to be converted.
        document_df: pd.Dataframe, optional
            Optional data about the documents in the model

        Returns
        -------
        TopicDatabase
        """
        n_layers = len(topic_model.cluster_layers_)
        c_tree = {}
        for k in topic_model.cluster_tree_:
            c_tree[str(k)] = [str(i) for i in topic_model.cluster_tree_[k]]
        vectorTopics = vector_topics(topic_model)
        topics = populate_topics(topic_model)
        topicFrame = pd.DataFrame.from_dict(topics, orient="index")

        vectorLayerFrame = pd.DataFrame([{
            f'layer{layer}': vectorTopics[k][layer] for layer in range(n_layers)
        } for k in vectorTopics.keys()])

        topicdb= cls(
            topicFrame=topicFrame,
            vectorArray=topic_model.embedding_vectors_,
            vectorFrame=pd.concat([document_df.reset_index(), vectorLayerFrame], axis=1),
            reducedArray=topic_model.clusterable_vectors_,
            sep='/',
        )
        return topicdb

"""=================== Utilities ====================="""

def child_tree_to_parent_tree(child_tree):
    parent_tree = {}
    for p, children in child_tree.items():
        for c in children:
            parent_tree[topic_uid(c)] = topic_uid(p)
    return parent_tree
    
def node_path(node, parent_tree, sep='/'):
    if node not in parent_tree:
        raise ValueError(f"Node {node} is not in the tree")
    path = [node]
    while node in parent_tree:
        node = parent_tree[node]
        path.append(node)
    path.reverse()
    return sep.join(map(str, path))

def node_in_path(node, path, sep='/'):
    return node in path.split(sep)

def node(path, sep='/'):
    return path.split(sep)[-1]

def parent(node, sep='/'):
    return sep.join(node.split(sep)[:-1])


import base64
def topic_uid(tup) -> str:
    a,b = tup
    a=int(a)
    b=int(b)+1 # Because unclustered is -1 and we can't convert negative to unsigned. 
    combined = (a << 10) | b   # pack into 20 bits
    return base64.urlsafe_b64encode(combined.to_bytes(3, "big")).rstrip(b'=').decode()

def uid_to_ints(s: str):
    ## returns layer, cluster
    padded = s + '=' * (-len(s) % 4)
    combined = int.from_bytes(base64.urlsafe_b64decode(padded), "big")
    return combined >> 10, (combined & 0x3FF)-1


def populate_topics(topic_model):
    topics = {}
    parent_tree = child_tree_to_parent_tree(topic_model.cluster_tree_)
    for i, layer in enumerate(topic_model.cluster_layers_):
        for c in set(layer.cluster_labels):
            if c == -1:
                continue
            uid = topic_uid((i,c))
            topic_info = {
                'uid':uid,
                'layer':i,
                'cluster_number':c,
                'path':node_path(uid, parent_tree),
                'children':[topic_uid(x) for x in topic_model.cluster_tree_.get((i,c), [])],
                'vectors':(layer.cluster_labels == c).nonzero()[0],
                'name':layer.topic_names[c],
                'exemplars':layer.exemplars[c],
                'keyphrases':layer.keyphrases[c],
            }
            topics[uid] = topic_info

    # add the root node manually
    n_layers = len(topic_model.cluster_layers_)
    root_uid = topic_uid((n_layers, 0))
    topics[root_uid] = {
        'uid':root_uid,
        'layer':n_layers,
        'cluster_number':0,
        'path':root_uid,
        'children':[topic_uid((n_layers-1, c)) for c in np.unique(topic_model.cluster_layers_[-1].cluster_labels)],
        'vectors':np.arange(0,len(topic_model.embedding_vectors_),dtype=int),
        'name':"Root Node",
        'exemplars':[],
        'keyphrases':[],
    }
    
    return topics

import pandas as pd
def vector_topics(topic_model):
    n_vectors = topic_model.clusterable_vectors_.shape[0]
    return {
        i:[int(topic_model.cluster_layers_[l].cluster_labels[i])
           for l in range(len(topic_model.cluster_layers_))]
        for i in range(n_vectors)
    }

def query_all(indices, frame, vec_df):
    nodes = set()
    for i in indices:
        for _, node in vec_df.iloc[i].items():
            if node[1] != -1:
                nodes.add(node)
                break
    paths = [frame.path[str(node)] for node in nodes]
    res = frame.join(paths)
    return res

def serialize_df(df):
    df_copy = df.copy()
    convert=False
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            # Convert lists/dicts to JSON strings
            try:
                df_copy[col] = df_copy[col].apply(json.dumps)
            except TypeError as _:
                convert=True
                df_copy[col] = df_copy[col].apply(lambda x: [int(y) for y in x])
                df_copy[col] = df_copy[col].apply(json.dumps)
    if convert:
        print("Warning: JSON cannot serialize int64, converting to int32.")
    return df_copy

def deserialize_df(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            try:
                df_copy[col] = df_copy[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            except Exception:
                pass  # leave as string if not JSON
    return df_copy