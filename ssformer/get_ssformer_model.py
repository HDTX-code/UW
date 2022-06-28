from ssformer.build_ssformer_model import build


def get_ssformer_model(model_name, num_classes):
    model = build(model_name=model_name, class_num=num_classes)
    return model