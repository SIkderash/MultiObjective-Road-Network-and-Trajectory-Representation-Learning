import os
import logging

def setup_logger(log_dir, config):
    os.makedirs(log_dir, exist_ok=True)
    log_name = generate_ablation_name(config) + ".log"
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("AblationLogger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def generate_ablation_name(config):
    flags = []
    flags.append("MTM")
    if config.get('use_temporal_encoding'): flags.append("temporal")
    if config.get('use_time_embeddings'): flags.append("timeemb")
    if config.get('use_spatial_fusion'): flags.append("spatialfusion")
    if config.get('use_traj_traj_cl'): flags.append("trajtraj")
    if config.get('use_traj_node_cl'): flags.append("trajnode")
    if config.get('use_node_node_cl'): flags.append("nodenode")
    flags.append(config.get("contrastive_type", "none"))
    return "_".join(flags)