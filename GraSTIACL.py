import argparse
import logging
import random
import os

import matplotlib.pyplot
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC, SVC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from datasets import ADNIDataset
from datasets import TUEvaluator
from unsupervised.embedding_evaluation import EmbeddingEvaluation, get_emb_y
from unsupervised.encoder import TA_encoder
from unsupervised.learning import GInfoMinMax
from unsupervised.view_learner import ViewLearner
from unsupervised.learning import GraSTI
from unsupervised.utils import initialize_node_features, set_tu_dataset_y_shape
from scipy import interp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd



def calc_regloss(z, aug, memory, temperature: float = 0.1, pos_only: bool = False):
    device = z.device
    b = z.size(0)
    z = F.normalize(z, dim=-1)
    aug = F.normalize(aug, dim=-1)
    memory = F.normalize(memory, dim=-1)

    logits = torch.einsum("if, jf -> ij", z, aug) / temperature
    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((b, b), dtype=torch.bool, device=device)
    pos_mask.fill_diagonal_(True)

    m_logits = torch.einsum("if, jf -> ij", z, memory) / temperature
    exp_logits = torch.exp(m_logits)
    log_prob = logits if pos_only else logits - torch.log(exp_logits.sum(1, keepdim=True))
    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)

    loss = -mean_log_prob_pos.mean()

    return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    my_transforms = Compose([set_tu_dataset_y_shape])
    dataset = ADNIDataset(args.path, args.name, transform=my_transforms)

    dataset.data.y = dataset.data.y.squeeze()

    evaluator = TUEvaluator()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    beta = 0.5
    encoder = TA_encoder.TAEncoder(num_dataset_features=3, beta=beta, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
                             drop_ratio=args.drop_ratio, pooling_type=args.pooling_type)
    grasti = GraSTI.ToyNet(input_dim=90, hidden_dim=args.vib_hidden_dim)
    print("Encoder parameter count:")
    encoder_total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total: {encoder_total_params}")
    print("GraSTI parameter count:")
    GraSTI_total_params = sum(p.numel() for p in grasti.parameters() if p.requires_grad)
    print(f"Total: {GraSTI_total_params}")
    model = GInfoMinMax(
        TA_encoder.TAEncoder(num_dataset_features=3, beta=beta, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
                             drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        GraSTI.ToyNet(input_dim=90, hidden_dim=args.vib_hidden_dim),
        args.emb_dim).to(device)
    print("Model parameter count:")
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {model_total_params}")
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
    view_learner = ViewLearner(
        TA_encoder.TAEncoder(num_dataset_features=3, beta=beta, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers,
                             drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        GraSTI.ToyNet(input_dim=90, hidden_dim=args.vib_hidden_dim)).to(device)
    print("View parameter count:")
    view_total_params = sum(p.numel() for p in view_learner.parameters() if p.requires_grad)
    print(f"Total: {view_total_params}")


    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True, max_iter=10000), evaluator,
                                 dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True)

    model.eval()
    train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, beta, dataset)
    logging.info(
        "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score, test_score))

    model_losses = []
    view_losses = []
    view_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []
    valid_std_curve = []
    test_std_curve = []
    train_std_curve = []
    valid_f1_curve = []
    test_f1_curve = []
    train_f1_curve = []
    valid_f1_std_curve = []
    test_f1_std_curve = []
    train_f1_std_curve = []
    valid_sen_curve = []
    test_sen_curve = []
    train_sen_curve = []
    valid_sen_std_curve = []
    test_sen_std_curve = []
    train_sen_std_curve = []
    valid_spe_curve = []
    test_spe_curve = []
    train_spe_curve = []
    valid_spe_std_curve = []
    test_spe_std_curve = []
    train_spe_std_curve = []
    train_pre_curve = []
    valid_pre_curve = []
    test_pre_curve = []
    train_pre_std_curve = []
    valid_pre_std_curve = []
    test_pre_std_curve = []

    test_auc_curve = []
    test_auc_std_curve = []
    aug_edge_weights = []
    test_emb_pic = []
    for epoch in range(1, args.epochs + 1):
        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0
        aug_edge_weight_all = torch.zeros(args.template, args.template)
        for batch in dataloader:
            # set up
            batch = batch.to(device)
            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            model.eval()

            x, _, edge_logits = model(batch.batch, batch.x, batch.edge_index, beta, None, batch.edge_weight,
                                      batch.dyn_weight)
            _, mu, std, edge_prod = view_learner(batch.batch, batch.x, batch.edge_index, beta, None,
                                                 batch.edge_weight, batch.dyn_weight)
            temperature = 1
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs_ = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs_.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = batch.edge_weight * torch.sigmoid(gate_inputs).squeeze()
            x_aug, _, _ = model(batch.batch, batch.x, batch.edge_index, beta, None, batch_aug_edge_weight,
                                batch.dyn_weight)
            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - torch.sigmoid(gate_inputs).squeeze()

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")
            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()

            mu = torch.reshape(mu, [args.batch_size, -1])
            std = torch.reshape(std, [args.batch_size, -1])
            kld_loss = - 0.5 * torch.mean((1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1))
            view_loss = model.calc_loss(x, x_aug) - args.kld_lambda * kld_loss
            view_loss_all += view_loss.item() * batch.num_graphs
            reg_all += reg.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()

            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _, edge_logits = model(batch.batch, batch.x, batch.edge_index, beta, None, batch.edge_weight,
                                      batch.dyn_weight)
            _, mu, std, edge_prod = view_learner(batch.batch, batch.x, batch.edge_index, beta, None,
                                                 batch.edge_weight, batch.dyn_weight)
            temperature = 1
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs_ = torch.log(eps) - torch.log(1 - eps)
            gate_inputs_ = gate_inputs_.to(device)
            gate_inputs = (gate_inputs_ + edge_logits) / temperature
            batch_aug_edge_weight = batch.edge_weight * torch.sigmoid(gate_inputs).squeeze()
            x_aug, _, _ = model(batch.batch, batch.x, batch.edge_index, beta, None, batch_aug_edge_weight,
                                batch.dyn_weight)

            edge_prod_sig = torch.sigmoid(edge_prod.squeeze()).detach()
            edge_logits_sig = torch.sigmoid((edge_logits + gate_inputs_) / temperature)
            ce_loss = F.binary_cross_entropy(edge_logits_sig, edge_prod_sig)
            model_loss = model.calc_loss(x, x_aug) + args.ce_lambda * ce_loss
            model_loss_all += model_loss.item() * batch.num_graphs

            model_loss.backward()
            model_optimizer.step()
        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)
        fin_aug_edge_weight = aug_edge_weight_all / len(dataloader)
        beta = np.random.beta(fin_reg, 1 - fin_reg)
        logging.info(
            'Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, fin_model_loss,
                                                                   fin_view_loss,
                                                                   fin_reg))
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)
        aug_edge_weights.append(fin_aug_edge_weight)
        if epoch % args.eval_interval == 0:
            model.eval()
            train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, beta, dataset, flag=True)

            logging.info(
                "Metric: {} Train_mean: {} Val_mean: {} Test_mean: {}".format(evaluator.eval_metric, train_score[0],
                                                                              val_score[0],
                                                                              test_score[0]))
            logging.info(
                "Metric: {} Train_std: {} Val_std: {} Test_std: {}".format(evaluator.eval_metric, train_score[1],
                                                                           val_score[1],
                                                                           test_score[1]))
            logging.info(
                "Metric: f1 Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[2], val_score[2],
                                                                              test_score[2]))

            logging.info(
                "Metric: f1 Train_std: {} Val_std: {} Test_std: {}".format(train_score[3], val_score[3], test_score[3]))

            logging.info(
                "Metric: sen Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[4], val_score[4],
                                                                               test_score[4]))

            logging.info(
                "Metric: sen Train_std: {} Val_std: {} Test_std: {}".format(train_score[5], val_score[5],
                                                                            test_score[5]))

            logging.info(
                "Metric: spe Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[6], val_score[6],
                                                                               test_score[6]))

            logging.info(
                "Metric: spe Train_std: {} Val_std: {} Test_std: {}".format(train_score[7], val_score[7],
                                                                            test_score[7]))
            logging.info(
                "Metric: precision Train_mean: {} Val_mean: {} Test_mean: {}".format(train_score[8], val_score[8],
                                                                                     test_score[8]))
            logging.info(
                "Metric: precision Train_std: {} Val_std: {} Test_std: {}".format(train_score[9], val_score[9],
                                                                                  test_score[9]))

        train_f1_curve.append(train_score[2])
        valid_f1_curve.append(val_score[2])
        test_f1_curve.append(test_score[2])
        train_f1_std_curve.append(train_score[3])
        valid_f1_std_curve.append(val_score[3])
        test_f1_std_curve.append(test_score[3])

        train_sen_curve.append(train_score[4])
        valid_sen_curve.append(val_score[4])
        test_sen_curve.append(test_score[4])
        train_sen_std_curve.append(train_score[5])
        valid_sen_std_curve.append(val_score[5])
        test_sen_std_curve.append(test_score[5])

        train_spe_curve.append(train_score[6])
        valid_spe_curve.append(val_score[6])
        test_spe_curve.append(test_score[6])
        train_spe_std_curve.append(train_score[7])
        valid_spe_std_curve.append(val_score[7])
        test_spe_std_curve.append(test_score[7])

        train_curve.append(train_score[0])
        valid_curve.append(val_score[0])
        test_curve.append(test_score[0])
        train_std_curve.append(train_score[1])
        valid_std_curve.append(val_score[1])
        test_std_curve.append(test_score[1])

        train_pre_curve.append(train_score[8])
        valid_pre_curve.append(val_score[8])
        test_pre_curve.append(test_score[8])
        train_pre_std_curve.append(train_score[9])
        valid_pre_std_curve.append(val_score[9])
        test_pre_std_curve.append(test_score[9])

        test_auc_curve.append(test_score[10])
        test_auc_std_curve.append(test_score[11])

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    best_train_epoch = np.argmax(np.array(train_curve))
    best_test_epoch = np.argmax(np.array(test_curve))

    best_f1_train_epoch = np.argmax(np.array(train_f1_curve))
    best_f1_valid_epoch = np.argmax(np.array(valid_f1_curve))
    best_f1_test_epoch = np.argmax(np.array(test_f1_curve))

    best_sen_train_epoch = np.argmax(np.array(train_sen_curve))
    best_sen_valid_epoch = np.argmax(np.array(valid_sen_curve))
    best_sen_test_epoch = np.argmax(np.array(test_sen_curve))

    best_spe_train_epoch = np.argmax(np.array(train_spe_curve))
    best_spe_valid_epoch = np.argmax(np.array(valid_spe_curve))
    best_spe_test_epoch = np.argmax(np.array(test_spe_curve))

    best_pre_train_epoch = np.argmax(np.array(train_pre_curve))
    best_pre_valid_epoch = np.argmax(np.array(valid_pre_curve))
    best_pre_test_epoch = np.argmax(np.array(test_pre_curve))

    best_auc_test_epoch = np.argmax(np.array(test_auc_curve))

    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info(
        'BestTrainScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {}'.format(
            best_train, train_std_curve[best_train_epoch],
            train_f1_curve[best_f1_train_epoch], train_f1_std_curve[best_f1_train_epoch],
            train_sen_curve[best_sen_train_epoch], train_sen_std_curve[best_sen_train_epoch],
            train_spe_curve[best_spe_train_epoch], train_spe_std_curve[best_spe_train_epoch],
            train_pre_curve[best_pre_train_epoch], train_pre_std_curve[best_pre_train_epoch]))
    logging.info(
        'BestValidationScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {}'.format(
            valid_curve[best_val_epoch], valid_std_curve[best_val_epoch],
            valid_f1_curve[best_f1_valid_epoch], valid_f1_std_curve[best_f1_valid_epoch],
            valid_sen_curve[best_sen_valid_epoch], valid_sen_std_curve[best_sen_valid_epoch],
            valid_spe_curve[best_spe_valid_epoch], valid_spe_std_curve[best_spe_valid_epoch],
            valid_pre_curve[best_pre_valid_epoch], valid_pre_std_curve[best_pre_valid_epoch]))
    logging.info(
        'BestTestScore: acc_mean: {} acc_std: {} f1_mean: {} f1_std: {} sen_mean: {} sen_std: {} spe_mean: {} spe_std: {} pre_mean: {} pre_std: {} auc_mean:{} auc_std:{}'.format(
            test_curve[best_test_epoch], test_std_curve[best_test_epoch],
            test_f1_curve[best_f1_test_epoch], test_f1_std_curve[best_f1_test_epoch],
            test_sen_curve[best_sen_test_epoch], test_sen_std_curve[best_sen_test_epoch],
            test_spe_curve[best_spe_test_epoch], test_spe_std_curve[best_spe_test_epoch],
            test_pre_curve[best_pre_test_epoch], test_pre_std_curve[best_pre_test_epoch],
            test_auc_curve[best_f1_test_epoch], test_auc_std_curve[best_auc_test_epoch]))

    return valid_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='GraSTI-ACL ADNI')

    parser.add_argument('--name', type=str, default='GraSTI-ACL',
                        help='dataset.')
    parser.add_argument('--path', type=str, default='',
                        help='path of dataset.')
    parser.add_argument('--template', type=int, default=90,
                        help='dataset template.')
    parser.add_argument('--model_lr', type=float, default=0.0005,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.0005,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=2,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--vib_hidden_dim', type=int, default=400,
                        help='max length of memory bank')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.3,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Train Epochs')
    parser.add_argument('--kld_lambda', default=0.003, type=float,
                        help='Regularization coefficients for loss of KL diverse')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear",
                        help="Downstream classifier is linear or non-linear")
    parser.add_argument('--ce_lambda', type=float, default=2.0,
                        help='Regularization coefficients for loss of cross entrpy')
    parser.add_argument('--seed', type=int, default=123)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)
