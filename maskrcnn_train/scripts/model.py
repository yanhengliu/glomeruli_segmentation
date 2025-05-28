import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes=2, pretrained=True):
    """
    Create a Mask R-CNN instance segmentation model.

    If pretrained=True, loads COCO-pretrained weights and adapts the heads
    to predict the specified number of classes.

    Args:
        num_classes (int): Total number of classes (including background).
                           For one object class + background, set to 2.
        pretrained (bool):   Whether to load pretrained COCO weights.
                             If True, loads default weights; otherwise,
                             initializes the model randomly.

    Returns:
        torch.nn.Module: The configured Mask R-CNN model.
    """
    # Load a Mask R-CNN ResNet-50 FPN model
    if pretrained:
        # Use COCO pretrained weights
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None)

    # Replace the box predictor to match our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor to match our number of classes
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # default hidden layer size in Mask R-CNN
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)

    return model
