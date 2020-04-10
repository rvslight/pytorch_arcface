class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 10#20 #500 #13938
    metric = 'arc_margin' #'add_margin' #''arc_margin' #None #'arc_margin' None #
    easy_margin = True
    use_se = True
    loss = None #'smooth_l1_loss' #'focal_loss'

    display = True
    finetune = False

    train_root = './data/Datasets/webface/CASIA-WebFace/'
    train_list = './data/Datasets/webface/train_data_13938.txt'
    val_list = './data/Datasets/webface/val_data_1000.txt'

    test_root = './data/Datasets/webface/CASIA-WebFace/'
    test_list = './data/Datasets/webface/test_data_1000.txt'

    lfw_root = './data/Datasets/lfw/lfw-align-128'
    lfw_test_list = './data/Datasets/lfw/lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    train_load_model_path = 'checkpoints/soft_07_resnet18_140.pth'
    # load_model_path = 'checkpoints/resnet18_90.pth'
    test_model_path = 'checkpoints/resnet18_140.pth'
    save_interval = 10

    s = 30
    m = 0.5  # 0.35
    train_batch_size = 128 #256 #64 #16  # batch size
    test_batch_size = 60

    input_shape = (3, 128, 128)
    optimizer = 'radam' #'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 10  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 150
    lr = 1e-1  # initial learning rate
    milestones = [15,30,45,60,70,90]
    # lr_step = 35
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
