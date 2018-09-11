# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html


from sklearn.preprocessing import LabelEncoder

def main():
    le = LabelEncoder()
    le.fit([1,2,2,6])
    print(le.classes_)

    print(le.transform([1,1,2,6]))

    print(le.inverse_transform([0, 0, 1, 2]))

if __name__ == '__main__':
    main()
