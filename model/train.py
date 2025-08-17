
import os
import json
import argparse

from utils.process_data import *
from model.regressors import *


def main(args):
    
    alphas = [0.2, 0.5, 0.8]
    margins = [0.001, 0.01, 0.05, 0.1]

    cv_splits = ['fold_random_5']
    model_types = ['mlp', 'cnn', 'light_attention']

    cv_folder = './DMS_enzyme/'
    test_indices = [0, 1, 2, 3, 4]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    epochs = args.epochs
    patience = args.patience
    
    embeddings_base_path = args.embeddings_base_path
    logs_folder = args.logs_folder
    checkpoint_base_path = args.checkpoint_base_path
    save_path = args.save_path    
    
    if args.file_index:
        cv_files = [os.listdir(cv_folder)[args.file_index]]
    else:
        cv_files = os.listdir(cv_folder)
    
    for cv_file in cv_files:
        print("processing:", cv_file)
        name = cv_file.split('_')[0]
        log_file = os.path.join(logs_folder, f'training_log_{name}.json')

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        
        for cv_split in cv_splits:
            for model_type in model_types:
                for alpha in alphas: 
                    for margin in margins:
                        
                        results_dict = {"spearman": [], "mse": []}

                        for test_index in test_indices:
                            embeddings_folder = os.path.join(embeddings_base_path, name)
                            X_train, y_train, X_test, y_test = process_datasets(
                                embeddings_folder, os.path.join(cv_folder, cv_file), cv_split, test_index)
                            print(f"Processing: {cv_split}, test_index: {test_index}")

                            if checkpoint_base_path and save_path:
                                checkpoint_dir = os.path.join(checkpoint_base_path, name)
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                save_path = os.path.join(checkpoint_dir, f'{name}_grid_{cv_split}_{test_index}_{model_type}_{alpha:.1f}_{margin}_seed{seed}.pth')

                            best_spearman, best_mse, _ = train_model_rankloss(X_train, y_train, X_test, y_test, epochs, seed, save_path,
                                                        model_type, alpha, margin, patience)
                            # best_spearman, best_mse, _ = train_evolvepro(X_train, y_train, X_test, y_test)

                            results_dict["spearman"].append(float(best_spearman))
                            results_dict["mse"].append(float(best_mse))

                        log_entry = {
                            "cv_split": cv_split,
                            "model_type": model_type,
                            "alpha": alpha,
                            "margin": margin,
                            "avg_spearman": np.mean(results_dict["spearman"]),
                            "avg_mse": np.mean(results_dict["mse"]),
                            "std_spearman": np.std(results_dict["spearman"]),
                            "std_mse": np.std(results_dict["mse"]),
                            "seed": seed
                        }
                        logs.append(log_entry)

                        with open(log_file, "w") as f:
                            json.dump(logs, f, indent=4)

                        print(f"Finished: {log_entry}")
                            
                        
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embeddings_base_path', type=str, default='./embeddings/ESM2_650M/', 
                        help="Path to the embeddings folder root.")
    parser.add_argument('--logs_folder', type=str, default='./logs/650M_grid/', 
                        help="Path to the logs folder.")
    parser.add_argument('--checkpoint_base_path', type=str, default='./checkpoints/', 
                        help="Path to save model checkpoints.")
    parser.add_argument('--save_path', type=str, default='', 
                        help="Path to save the final model or outputs.")
    parser.add_argument('--seed', type=int, default=3407, 
                        help="Random seed for reproducibility.")
    parser.add_argument('--gpu', type=str, default='4', 
                        help="GPU device to use.")
    parser.add_argument('--epochs', type=int, default=100, 
                        help="Number of epochs to train the model.")
    parser.add_argument('--patience', type=int, default=20, 
                        help="Patience for early stopping.")
    
    parser.add_argument('--file_index', type=int, default=0, 
                        help="Test file index")

                        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)