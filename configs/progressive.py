class Config(object): 
    # random seed
    seed = 0

    # optimize
    init_lr_E = 1e-4
    step_size = 500
    gamma = 0.9
    max_epochs = 30
    loss = 'Softmax'
    temp = 0.1

    # dataset
    input_data = 'dct'    
    batch_size = 8
    num_workers = 8
    resize_size = (128, 128)
    class_num = 15

    # train augnet
    inc = 32
    kernel_size = 3
    augnet_lr = 1e-2
    w_dist_pre = 1e-4
    w_close_known= 1e-2
    known_sim_limit = 0.95
    mse_lowbound = 0
    
    # train classifier
    cls_pre = True
    start_cls_pre_epoch = 1

    # evaluation
    metric = 'acc'
    cluster_method='kmeans'

    # save and vis
    vis_interval = 500
    save_interval = 5
    test_interval = 1

