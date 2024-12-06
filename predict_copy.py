import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import sklearn.metrics as skmetrics
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_data, get_subject_files
from model import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   save_seq_ids,
                   load_seq_ids)
from logger import get_logger

def compute_performance(cm):
    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float)
    tpfn = np.sum(cm, axis=1).astype(np.float)
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1

def plot_accuracy_loss(accuracies, losses, title):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(title)
    plt.show()

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_f1_score(f1_scores, title):
    plt.figure()
    plt.plot(f1_scores, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.show()

def predict(
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
):
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = get_logger(log_file, level="info")

    subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))

    fname = "{}.txt".format(config["dataset"])
    seq_sids = load_seq_ids(fname)
    logger.info("Load generated SIDs from {}".format(fname))
    logger.info("SIDs ({}): {}".format(len(seq_sids), seq_sids))

    fold_pids = np.array_split(seq_sids, config["n_folds"])

    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    trues = []
    preds = []
    accuracies = []
    f1_scores = []
    for fold_idx in range(config["n_folds"]):
        logger.info("------ Fold {}/{} ------".format(fold_idx+1, config["n_folds"]))
        test_sids = fold_pids[fold_idx]

        logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

        model = TinySleepNet(
            config=config,
            output_dir=os.path.join(model_dir, str(fold_idx)),
            use_rnn=True,
            testing=True,
            use_best=use_best,
        )

        s_trues = []
        s_preds = []
        for sid in test_sids:
            logger.info("Subject ID: {}".format(sid))

            test_files = get_subject_files(
                dataset=config["dataset"],
                files=subject_files,
                sid=sid,
            )

            for vf in test_files: logger.info("Load files {} ...".format(vf))

            test_x, test_y, _ = load_data(test_files)

            logger.info("Test set (n_night_sleeps={})".format(len(test_y)))
            for _x in test_x: logger.info(_x.shape)
            print_n_samples_each_class(np.hstack(test_y))

            if config["model"] == "model-origin":
                for night_idx, night_data in enumerate(zip(test_x, test_y)):
                    night_x, night_y = night_data
                    test_minibatch_fn = iterate_batch_seq_minibatches(
                        night_x,
                        night_y,
                        batch_size=config["batch_size"],
                        seq_length=config["seq_length"],
                    )
                    test_outs = model.evaluate(test_minibatch_fn)
                    s_trues.extend(test_outs["test/trues"])
                    s_preds.extend(test_outs["test/preds"])
                    trues.extend(test_outs["test/trues"])
                    preds.extend(test_outs["test/preds"])

                    save_dict = {
                        "y_true": test_outs["test/trues"],
                        "y_pred": test_outs["test/preds"],
                    }
                    fname = os.path.basename(test_files[night_idx]).split(".")[0]
                    save_path = os.path.join(
                        output_dir,
                        "pred_{}.npz".format(fname)
                    )
                    np.savez(save_path, **save_dict)
                    logger.info("Saved outputs to {}".format(save_path))
            else:
                for night_idx, night_data in enumerate(zip(test_x, test_y)):
                    night_x, night_y = night_data
                    test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                        [night_x],
                        [night_y],
                        batch_size=config["batch_size"],
                        seq_length=config["seq_length"],
                        shuffle_idx=None,
                        augment_seq=False,
                    )
                    if (config.get('augment_signal') is not None) and config['augment_signal']:
                        test_outs = model.evaluate_aug(test_minibatch_fn)
                    else:
                        test_outs = model.evaluate(test_minibatch_fn)
                    s_trues.extend(test_outs["test/trues"])
                    s_preds.extend(test_outs["test/preds"])
                    trues.extend(test_outs["test/trues"])
                    preds.extend(test_outs["test/preds"])

                    save_dict = {
                        "y_true": test_outs["test/trues"],
                        "y_pred": test_outs["test/preds"],
                    }
                    fname = os.path.basename(test_files[night_idx]).split(".")[0]
                    save_path = os.path.join(
                        output_dir,
                        "pred_{}.npz".format(fname)
                    )
                    np.savez(save_path, **save_dict)
                    logger.info("Saved outputs to {}".format(save_path))

        s_acc = skmetrics.accuracy_score(y_true=s_trues, y_pred=s_preds)
        s_f1_score = skmetrics.f1_score(y_true=s_trues, y_pred=s_preds, average="macro")
        s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels=[0,1,2,3,4])

        logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
            len(s_preds),
            s_acc*100.0,
            s_f1_score*100.0,
        ))

        logger.info(">> Confusion Matrix")
        logger.info(s_cm)

        accuracies.append(s_acc)
        f1_scores.append(s_f1_score)

        tf.reset_default_graph()

        logger.info("------------------------")
        logger.info("")

    acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
    cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])

    logger.info("")
    logger.info("=== Overall ===")
    print_n_samples_each_class(trues)
    logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
        len(preds),
        acc*100.0,
        f1_score*100.0,
    ))

    logger.info(">> Confusion Matrix")
    logger.info(cm)

    metrics = compute_performance(cm=cm)
    logger.info("Total: {}".format(metrics[0]))
    logger.info("Number of samples from each class: {}".format(metrics[1]))
    logger.info("Accuracy: {:.1f}".format(metrics[2]*100.0))
    logger.info("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
    logger.info("Per-class Precision: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[4]]))
    logger.info("Per-class Recall: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[5]]))
    logger.info("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]]))

    save_dict = {
        "y_true": trues,
        "y_pred": preds,
        "seq_sids": seq_sids,
        "config": config,
    }
    save_path = os.path.join(
        output_dir,
        "{}.npz".format(config["dataset"])
    )
    np.savez(save_path, **save_dict)
    logger.info("Saved summary to {}".format(save_path))

    # Plotting
    plot_accuracy_loss(accuracies, [1-acc for acc in accuracies], "Accuracy and Loss per Fold")
    plot_confusion_matrix(cm, "Overall Confusion Matrix")
    plot_f1_score(f1_scores, "F1 Score per Fold")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./out_sleepedf/finetune")
    parser.add_argument("--output_dir", type=str, default="./output/predict")
    parser.add_argument("--log_file", type=str, default="./output/output.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.set_defaults(use_best=False)
    args = parser.parse_args()

    predict(
        config_file=args.config_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        use_best=args.use_best,
    )