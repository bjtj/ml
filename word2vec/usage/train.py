import word2vec

def main():
    word2vec.word2phrase('./text8', './text8-phrases', verbose=True)
    word2vec.word2vec('./text8-phrases', './text8.bin', size=100, verbose=True)
    word2vec.word2clusters('./text8', './text8-clusters.txt', 100, verbose=True)

if __name__ == '__main__':
    main()




