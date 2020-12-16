import utils


def main():
    checkpoint, device, category_names, filepath, top_k = parse_predict_args()

    model, criterion, optimizer, epochs = load(save_dir, device)
    top_p, top_classes = predict(image_path, model, topk)

    print(top_p)
    print(top_classes)


if __name__== "__main__":
    main()