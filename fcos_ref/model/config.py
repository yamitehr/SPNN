class DefaultConfig():
    #backbone
    pretrained=True
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True

    #head
    class_num=80
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000


class SPNNConfig():
    """Config for FCOS with SPNN backbone on VOC (256x256 input)."""
    # backbone
    backbone_type = 'spnn'
    pretrained = False
    freeze_stage_1 = False
    freeze_bn = False

    # SPNN-specific
    spnn_hidden = 256
    spnn_mix_type = 'householder'
    spnn_pretrained = None  # path to ImageNet classification checkpoint

    # fpn
    fpn_out_channels = 256
    use_p5 = True

    # head
    class_num = 20  # VOC has 20 classes
    use_GN_head = True
    prior = 0.01
    add_centerness = True
    cnt_on_reg = True

    # training
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

    # inference
    score_threshold = 0.05
    nms_iou_threshold = 0.6
    max_detection_boxes_num = 1000


class SPNNE2EConfig():
    """Config for end-to-end invertible SPNN detector on VOC.
    Single scale at stride 16, all boxes assigned to one level."""
    # backbone
    backbone_type = 'spnn_e2e'
    pretrained = False
    freeze_stage_1 = False
    freeze_bn = False

    # SPNN-specific
    spnn_hidden = 256
    spnn_mix_type = 'householder'
    spnn_pretrained = None

    # head (not used — SPNN outputs directly)
    class_num = 20
    use_GN_head = True
    prior = 0.01
    add_centerness = True
    cnt_on_reg = True

    # training — single scale
    strides = [16]
    limit_range = [[-1, 999999]]

    # inference
    score_threshold = 0.05
    nms_iou_threshold = 0.6
    max_detection_boxes_num = 1000