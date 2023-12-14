from src.MnistClassifier import MnistClassifier


def main():
    classifier = MnistClassifier(primal=False)
    classifier.train()
    print(f"Accuracy: {round(classifier.score(), 2)}")
    classifier.display_confusion_matrix()


if __name__ == '__main__':
    main()
