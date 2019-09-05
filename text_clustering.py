import numpy
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

#'alt.atheism',
# 'comp.graphics',
# 'comp.os.ms-windows.misc',
# 'comp.sys.ibm.pc.hardware',
# 'comp.sys.mac.hardware',
# 'comp.windows.x',
# 'misc.forsale',
# 'rec.autos',
# 'rec.motorcycles',
# 'rec.sport.baseball',
# 'rec.sport.hockey',
# 'sci.crypt',
# 'sci.electronics',
# 'sci.med',
# 'sci.space',
# 'soc.religion.christian',
# 'talk.politics.guns',
# 'talk.politics.mideast',
# 'talk.politics.misc',
# 'talk.religion.misc'


def get_20newsgroups_data(
    train_test,
    cats=["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"],
):
    # cats = ['alt.atheism',
    #         'sci.space',
    #         'rec.autos',
    #         # 'rec.motorcycles',
    #         # 'rec.sport.baseball',
    #         # 'rec.sport.hockey',
    #         # 'sci.crypt',
    #         # 'sci.electronics',
    #         # 'sci.med',
    #         ]
    data = fetch_20newsgroups(
        subset=train_test,
        shuffle=True,
        remove=("headers", "footers", "quotes"),
        categories=cats,
    )
    target_names = data.target_names

    def truncate(text):
        return text[0 : min(len(text), 1000)]

    return [
        (truncate(d), [target_names[target]])
        for d, target in zip(data.data, data.target)
        if len(d.split(" ")) > 5
    ]


def load_texts():
    gold_categories = ["comp.windows.x", "rec.sport.baseball", "rec.motorcycles"]
    text_category = get_20newsgroups_data("train", cats=gold_categories)
    categories = [t[0] for _, t in text_category]
    tb = LabelEncoder()
    tb.fit(categories)
    texts = [text for text, _ in text_category]
    print("got %d texts" % len(texts))
    categories = tb.transform(categories)
    return texts, categories


def cluster_text(texts, n_clusters):
    vectorizer = TfidfVectorizer(
        min_df=5, sublinear_tf=True, max_features=10000, max_df=0.75, ngram_range=(1, 1)
    )
    X = vectorizer.fit_transform(texts)
    print("num features: %d" % len(vectorizer.get_feature_names()))
    svd = TruncatedSVD(n_components=500, n_iter=57, random_state=42)
    X = svd.fit_transform(X)
    print("ratio of explained: %f" % sum(svd.explained_variance_ratio_))
    # TODO: why is this ratio so bad??  seems like there is not that much correlation in tf-idfed data!??!
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_cluster = kmeans.predict(X)

    def calc_dist_std(x):
        m = numpy.mean(x, axis=0)
        dists = [numpy.linalg.norm(xr - m, 2) for xr in x]
        return numpy.std(dists)

    mean_std_in = numpy.mean(
        [calc_dist_std(X[y_cluster == i, :]) for i in range(n_clusters)]
    )
    std_between = calc_dist_std(kmeans.cluster_centers_)
    score = mean_std_in / std_between
    print("in-inbetween-deviation-ratio: %0.2f" % float(score))
    return X, y_cluster


texts, categories = load_texts()
X, y_cluster = cluster_text(texts, len(set(categories)))  # X is dimension reduced data


# In[37]:


# TSNE is just for fun! in order to visualize the clusters
from MulticoreTSNE import MulticoreTSNE as TSNE

# from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, n_jobs=4)
X_embedded = tsne.fit_transform(X)


# In[70]:


from matplotlib import pyplot as plt


def plot_tsned(X_embedded, y_cluster, encoded_categories):

    norm = plt.Normalize(1, 4)
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    # maxx = numpy.max(X_embedded, axis=0)
    # minn = numpy.min(X_embedded, axis=0)
    sc = ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=encoded_categories,
        s=40,
        cmap=cmap,
        marker="o",
        # norm=norm,
        linewidths=0.0,
    )
    m = {1: 1, 2: 0, 0: 2}
    sc = ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=[m[k] for k in y_cluster],
        s=10,
        cmap=cmap,
        # norm=norm,
        linewidths=0.0,
    )
    plt.show()


plot_tsned(X_embedded, y_cluster, categories)
# the plot show 3 not that distinctive clusters; true-categories are depicted with larger dots;
# a green dot inside of a bigger one that is colored yellow means, that kmeans assigned this document to the
# green cluster; whereas it should
