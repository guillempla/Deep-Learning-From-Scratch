import argparse


def main():
    parser = argparse.ArgumentParser(description='Evaluate accuracy')
    parser.add_argument('predictions', type=str,
                        help='file path to predictions of your model')
    parser.add_argument('truth', type=str,
                        help='file path to ground truth labels')

    args = parser.parse_args()

    with open(args.predictions, 'r') as rf:
        pred = rf.read().split()
    with open(args.truth, 'r') as rf:
        truth = rf.read().split()

    assert len(pred) == len(truth)

    hits = sum([p == t for p, t in zip(pred, truth)])

    print(f'Accuracy {hits/len(truth):2.4f}')


if __name__ == '__main__':
    main()
