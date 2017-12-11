import codecs
import jieba


class ReadFile(object):
    def __init__(self, directory):
        self.directory = directory

    def FileListInDirectory(self):
        import os
        filesInDirectory = os.listdir(self.directory)
        fileList = []
        for eachFile in filesInDirectory:
            child = os.path.join('%s%s' % (self.directory+"\\", eachFile))
            fileList.append(child)
        return fileList

class GetSimilarWords(object):
    def __init__(self, sourceFilename=None, resultFilename=None):
        self.sourceFilename = sourceFilename
        self.resultFilename = resultFilename

    def splitWord(self):

        fread = codecs.open(self.sourceFilename, encoding="GBK", errors="ignore")
        fwrite = codecs.open(self.resultFilename, 'a', encoding="utf-8")

        lines = fread.readlines()
        for line in lines:
            line.replace('\t', '').replace('\n', '').replace(' ', '')
            seg_list = jieba.cut(line, cut_all=False)
            fwrite.write(" ".join(seg_list))

        fread.close()
        fwrite.close()

    def train(self):
        from gensim.models import word2vec

        sentences = word2vec.Text8Corpus(u"result.txt")
        model = word2vec.Word2Vec(sentences, size=200)
        print(model)
        return model

    def SimilarBetweenWords(self, model, vocab1, vocab2):
        try:
            y1 = model.similarity(vocab1, vocab2)
        except KeyError:
            y1 = 0
        print("the similarity between vocab1 and vocab2 is:", y1)

    def SimilarWordList(self, model, vocab):
        list = model.most_similar(vocab, topn=5)
        print("Similar Word List to Vocab is:")
        for item in list:
            print(item[0], item[1])

    def ModelSave(self, model):
        model.save("wordVectorModel.model")

    def ModelLoad(self, modelname):
        from gensim.models import word2vec
        model = word2vec.Word2Vec.load(modelname)
        return model

    def clusterUsingWord2Vec(self, model):
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.cluster import AgglomerativeClustering
        word2vec_dict = {}
        for i in model.wv.vocab.keys():
            try:
                word2vec_dict[i] = model[i]
            except:
                pass

        clusters = MiniBatchKMeans(n_clusters=1000, max_iter=100, batch_size=100, n_init=3, init_size=2000)
        X = np.array([v.T for _, v in word2vec_dict.items()])
        y = [k for k, _ in word2vec_dict.items()]
        clusters.fit(X)
        from collections import defaultdict
        cluster_dict = defaultdict(list)
        for word, label in zip(y, clusters.labels_):
            cluster_dict[label].append(word)

        for i in range(len(cluster_dict)):
            print(cluster_dict[i])

        return cluster_dict

    def cos(self, vector1, vector2):
        dot_produce = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_produce += a*b
            normA += a**2
            normB += b**2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_produce/((normB*normA)**0.5)

    def AccurateSimilarWords(self, cluster_dict, model):
        similarity = [[0 for i in range(200)] for i in range(200)]
        for i in range(len(cluster_dict)):
            ClusterWords = cluster_dict[i]
            for j in range(len(ClusterWords)):
                for k in range(j+1, len(ClusterWords)):
                    similarity[j][k] = self.cos(model[ClusterWords[j]], model[ClusterWords[k]])
                    print(ClusterWords[j], ClusterWords[k], similarity[j][k])


if __name__ == "__main__":
    list1 = ReadFile("E:\\NLP Project\\untitled1\C000008").FileListInDirectory()
    getSimilarWords = GetSimilarWords()
    for _ in list1:
        GetSimilarWords(_, "result.txt").splitWord()
    model = getSimilarWords.train()
    getSimilarWords.SimilarBetweenWords(model, "公司", "企业")
    getSimilarWords.SimilarWordList(model, "公司")
    getSimilarWords.ModelSave(model)
    print(model[u"公司"])
    cluster_result = getSimilarWords.clusterUsingWord2Vec(model)
    getSimilarWords.AccurateSimilarWords(cluster_result, model)