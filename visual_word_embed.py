import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def main():
    embeddings_file = 'data/word_embeddings.p'
    embed_dict = load_embeddings(embeddings_file)

    vocabulary = embed_dict.keys()
    word_vec = np.array(embed_dict.values())

    tsne = TSNE(n_components=2, random_state=0, n_iter=5000)
    manifold_learn_all = tsne.fit_transform(word_vec)

    nwords = 15
    nplots = 30

    magnitude = np.sqrt(np.sum(word_vec **2, axis=1, keepdims=True))
    cosine_similarity_mat = np.dot(word_vec, word_vec.T) / np.dot(magnitude, magnitude.T)

    np.set_printoptions(suppress=True)

    # get random words and cosine similarities to other words
    for i in np.random.permutation(word_vec.shape[0])[:nplots]:
        fig = plt.figure(figsize=(10, 10))
        # plot the 30 most similar words
        sorted_cs = np.argsort(np.abs(cosine_similarity_mat[i]))
        most_similar = sorted_cs[-nwords:]
        least_similar = sorted_cs[:nwords]

        # plot the 30 most similar words
        plt.plot(np.ones(nwords), (np.arange(nwords) + 1), 'go')
        # plot the 30 least similar words
        plt.plot(np.ones(nwords), -np.arange(nwords), 'ro')

        for m, j in zip(np.arange(nwords) + 1, most_similar):
            plt.annotate(vocabulary[j] + ' (%.5f)' % cosine_similarity_mat[i,j], xy=(1.2, m), xytext=(0, 0), textcoords='offset points')

        for m, k in zip(-np.arange(nwords), least_similar):
            plt.annotate(vocabulary[k] + ' (%.5f)' % cosine_similarity_mat[i,k], xy=(1.2, m), xytext=(0, 0), textcoords='offset points')

        plt.axis([0, 3, -nwords, nwords+1])
        plt.title('Cosine Similarity to "' + vocabulary[i] + '"')
        plt.savefig('CS_%s_%d.png' % (vocabulary[i], nwords), bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        # show the mapping in terms of TSNE distribution

        for m, x, y in zip(np.arange(word_vec.shape[0]), manifold_learn_all[:, 0], manifold_learn_all[:, 1]):
            if (m in most_similar):
                is_special = 2
            elif (m in least_similar):
                is_special = 1
            else:
                is_special = 0

            if (is_special):
                plt.annotate(vocabulary[m], xy=(x + 0.1, y + 0.1), xytext=(0,0), textcoords='offset points')

            opacity = .1 if is_special == 0 else 1
            color = 'g' if is_special == 2 else ('r' if is_special == 1 else 'b')
            plt.scatter(manifold_learn_all[m, 0], manifold_learn_all[m, 1], c=color, alpha=opacity)


        plt.savefig('tSNE_%s_%d.png' % (vocabulary[i], nwords), bbox_inches='tight')
        plt.close()


def load_embeddings(file_name):
    """ Load in the embeddings """
    return pickle.load(open(file_name, 'rb'))


if __name__ == '__main__':
    main()
