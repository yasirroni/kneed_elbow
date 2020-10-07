from kneed import KneeLocator

def kneed_elbow(x, model=None, fit_num=1, cluster_nums=range(2,10)):
    '''
    model           : KMeans() object from sklearn.cluster, KMeans() is used if None
    x               : {array-like, sparse matrix} of shape (n_samples, n_features)
    fit_num         : number of model.fit(x), default = 1
    cluster_nums    : iterable containing list of cluster numbers, default = range(2,10)
    '''
    
    if model == None:
        from sklearn.cluster import KMeans
        model = KMeans()
        
    distortions = []
    for cluster_num in cluster_nums:
        model.n_clusters=cluster_num
        inertia = 0
        for _ in range(fit_num):
            model.fit(x)
            inertia += model.inertia_
        distortions.append(inertia/fit_num)
    return KneeLocator(cluster_nums, distortions, S=1.0, curve="convex", direction="decreasing")
