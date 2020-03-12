def add_eql_config(cfg):
    """
    Add config for EQL.
    """
    cfg.MODEL.ROI_HEADS.LAMBDA = 0.00177
    cfg.MODEL.ROI_HEADS.PRIOR_PROB = 0.001

    # legacy cfg key (make model compatible with previous ckpt)
    cfg.MODEL.ROI_HEADS.FREQ_INFO = ""
