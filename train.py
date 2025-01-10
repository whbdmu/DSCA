import argparse
import datetime
import os.path as osp
import time
import torch
import torch.utils.data
import torch.nn.functional as F
from datasets import PersonSearchUDADataMoudle
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch_da
from models.seqnet_da import SeqNetDa
from models.cpm import ClusterProxyMemory
from utils import (
    mkdir,
    resume_from_ckpt,
    save_on_master,
    set_random_seed,
    generate_pseudo_labels,
    generate_cluster_features,
    generate_class_features,
)
from apex import amp
from spcl.models.dsbn import convert_dsbn
from spcl.utils.faiss_rerank import compute_jaccard_distance
from spcl.evaluators import extract_dy_features
from sklearn.cluster import DBSCAN


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    device = torch.device(cfg.DEVICE)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model and convert dsbn")
    model = SeqNetDa(cfg)
    """用于在模型中切换批归一化和域自适应归一化层之间"""
    convert_dsbn(model.roi_heads.reid_head)
    model.to(device)
    print(model)
    # build dataset module
    DataMoudle = PersonSearchUDADataMoudle(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        DataMoudle.setup(stage="test")
        gallery_loader, query_loader = DataMoudle.test_dataloader()
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    # build source predict dataloader
    DataMoudle.setup(stage="predict")
    predict_loader_source = DataMoudle.predict_dataloader(is_source=True)
    # init source domian identity level centroid
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_fea_dict = extract_dy_features(cfg, model, predict_loader_source, device, is_source=True)
    source_centers = generate_class_features(sour_fea_dict)
    print(f"source_centers length: {len(source_centers)}")
    print("the last one is the feature of 5555, remember don't use it")

    # build cluster memory
    source_classes = DataMoudle.dataset_source_train.num_train_pids
    memory = ClusterProxyMemory(
        256, source_classes, source_classes, temp=0.05, momentum=cfg.MODEL.UPDATE_FACTOR.ONLINE
    ).to(device)
    memory.features = source_centers.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    model.roi_heads.memory = memory
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1)

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    target_start_epoch = cfg.TARGET_REID_START
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        if epoch == target_start_epoch:
            # DBSCAN cluster
            eps = 0.5
            print(f"Clustering criterion: eps: {eps:.3f}")
            cluster = DBSCAN(eps=eps, min_samples=4, metric="precomputed", n_jobs=-1)

        if epoch >= target_start_epoch:
            # init target domain instance level features
            # we can't use target domain GT detection box feature to init, this is only for measuring the upper bound of cluster performance
            # for dynamic clustering method, we use the proposal after several epoches for first init, moreover, we'll update the memory with proposal before each epoch
            print("==> Initialize target-domain instance features in the hybrid memory")
            DataMoudle.setup(stage="predict")
            tgt_cluster_loader = DataMoudle.predict_dataloader(is_source=False)
            if epoch == target_start_epoch:
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(
                    cfg, model, tgt_cluster_loader, device, is_source=False
                )
            else:
                if args.resume and epoch == start_epoch:
                    target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(
                        cfg, model, tgt_cluster_loader, device, is_source=False
                    )
                else:
                    target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(
                        cfg,
                        model,
                        tgt_cluster_loader,
                        device,
                        is_source=False,
                        memory_proposal_boxes=img_proposal_boxes,
                        memory_target_features=target_features,
                        momentum=cfg.MODEL.UPDATE_FACTOR.OFFLINE,
                    )
            sorted_keys = sorted(target_features.keys())
            print("target_features instances :" + str(len(sorted_keys)))
            target_features = torch.cat([target_features[name] for name in sorted_keys], 0)
            target_features = F.normalize(target_features, dim=1).to(device)

            negative_fea = torch.cat([negative_fea[name] for name in sorted(negative_fea.keys())], 0)
            negative_fea = F.normalize(negative_fea, dim=1).to(device)
            print("hard negative instances :" + str(len(negative_fea)))

            # Calculate distance
            print("==> Create pseudo labels for unlabeled target domain with self-paced policy")
            rerank_dist = compute_jaccard_distance(target_features, k1=30, k2=6, search_option=3, use_float16=True)
            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            print(f"pseudo_labels length :{len(pseudo_labels)}")
            # merge source dataset and target dataset, set pseudo_labels after source domain
            pseudo_labels = generate_pseudo_labels(pseudo_labels, source_classes, num_ids)
            # use sorted_keys for searching pseudo_labels
            print("==> Modifying labels in target domain to build new training set")
            DataMoudle.reset_dataset_with_pseudo_labels(img_proposal_boxes, sorted_keys, pseudo_labels)
            # re-intalizating features memory
            cluster_features = generate_cluster_features(pseudo_labels, target_features, source_classes)
            print(f"cluster_features length: {len(cluster_features)}")
            source_centers = memory.features[0:source_classes].clone()
            memory.features = torch.cat((source_centers, cluster_features), dim=0).to(device)
            memory.features = torch.cat((memory.features, negative_fea), dim=0).to(device)
            memory.num_samples = memory.features.shape[0]
            print(f"total features length: {len(memory.features)}")
            target_features = target_features.cpu().clone()
        else:
            memory.num_samples = source_classes
        DataMoudle.setup(stage="train")
        train_loader_s, train_loader_t = DataMoudle.train_dataloader()

        train_one_epoch_da(cfg, model, optimizer, train_loader_s, train_loader_t, device, epoch, tfboard)
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            DataMoudle.setup(stage="test")
            gallery_loader, query_loader = DataMoudle.test_dataloader()
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "amp": amp.state_dict(),
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to configuration file.",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate the performance of a given checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume from the specified checkpoint.")
    parser.add_argument(
        "--ckpt",
        help="Path to checkpoint to resume or evaluate.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    main(args)
