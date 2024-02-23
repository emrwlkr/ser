import torch

def run_infer(model, image, params, label):
    print(f"Experiment name: {params.name}")
    print(f" Hyperparameters to create {params.name}\n Epochs: {params.epochs}\n Batch Size: {params.batch_size}\n Learning Rate: {params.learning_rate}")
    #infer label
    model.eval()
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)[0].item()

    # calculate confidence of label
    confidence = max(list(torch.exp(output)[0]))

    # generate ascii art
    pixels = image[0][0]
    print(generate_ascii_art(pixels))
    print(f"Model label prediction: {pred}\n with {confidence:.4f} confidence.")


def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "
