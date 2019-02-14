import word2vec


def main():

    print('== PREDICTIONS ==')
    
    model = word2vec.load('./text8.bin')
    print("model.vocab")
    print(model.vocab)
    print("model.vectors.shape")
    print(model.vectors.shape)
    print("model.vectors")
    print(model.vectors)
    print("model['dog'].shape")
    print(model['dog'].shape)
    print("model['dog'][:10]")
    print(model['dog'][:10])
    print("model.distance('dog', 'cat', 'fish')")
    print(model.distance('dog', 'cat', 'fish'))

    print('== SIMILARITY ==')

    indexes, metrics = model.similar('dog')
    print('indexes')
    print(indexes)
    print('metrics')
    print(metrics)
    print("model.vocab[indexes]")
    print(model.vocab[indexes])
    print("model.generate_response(indexes, metrics)")
    print(model.generate_response(indexes, metrics))
    print("model.generate_response(indexes, metrics).tolist()")
    print(model.generate_response(indexes, metrics).tolist())

    print('== PHRASES ==')
    indexes, metrics = model.similar('los_angeles')
    print('model.generate_response(indexes, metrics).tolist()')
    print(model.generate_response(indexes, metrics).tolist())

    print('== ANALOGIES ==')

    indexes, metrics = model.analogy(pos=['king', 'woman'], neg=['man'])
    print('indexes')
    print(indexes)
    print('metrics')
    print(metrics)
    print('model.generate_response(indexes, metrics).tolist()')
    print(model.generate_response(indexes, metrics).tolist())

    print('== CLUSTERS ==')
    clusters = word2vec.load_clusters('./text8-clusters.txt')
    print('clusters.vocab')
    print(clusters.vocab)
    print('clusters.get_words_on_cluster(90).shape')
    print(clusters.get_words_on_cluster(90).shape)
    print('clusters.get_words_on_cluster(90).shape[:10]')
    print(clusters.get_words_on_cluster(90)[:10])
    model.clusters = clusters
    indexes, metrics = model.analogy(pos=['paris', 'germany'], neg=['france'])
    print('model.generate_response(indexes, metrics).tolist()')
    print(model.generate_response(indexes, metrics).tolist())
    

if __name__ == '__main__':
    main()    
