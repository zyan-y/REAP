import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from utils.process_data import *
from model.header import PLM_RankReg
from model.loss import get_loss_function_baseline, rank_reg_loss

from sklearn.ensemble import RandomForestRegressor


def evaluate_mse_spearman(true_labels, predictions):
    test_mse = mean_squared_error(true_labels, predictions)
    test_spearmanr, _ = spearmanr(true_labels, predictions)
    return test_mse, test_spearmanr


def _set_device_and_seed(seed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)
    return device

def _prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size, device):
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_loader = torch.utils.data.DataLoader(
                    list(zip(X_train_t, y_train_t)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    list(zip(X_test_t, y_test_t)), batch_size=batch_size, shuffle=False)

    input_dim = X_train_t.shape[1]
    return train_loader, test_loader, input_dim

    
def train_plm_rankreg(X_train, y_train, X_test, y_test, epochs, seed, save_path, model_type, alpha=0.5, margin=0.1, patience=10, batch_size=1024):
    device = _set_device_and_seed(seed)
    train_loader, test_loader, input_dim = _prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size, device)
    
    model = PLM_RankReg(input_dim, model_type=model_type, dropout=0)  
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=3e-4, cycle_momentum=False)

    best_model = None  
    best_metrics_in_epochs = [float('-inf'), float('inf')]
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.view(-1, 1)
            loss = rank_loss(outputs, labels, alpha=alpha,margin=margin, num_samples=None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        test_mse, test_spearmanr = evaluate_mse_spearman(true_labels, predictions)
        # print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}, Spearman: {test_spearmanr:.5f}, MSE: {test_mse:.5f}")
        
        if test_spearmanr > best_metrics_in_epochs[0]:  
            best_metrics_in_epochs = [test_spearmanr, test_mse]  
            best_model = model.state_dict()
            patience_counter = 0
            if save_path:
                torch.save(best_model, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break
    print(f"Best Spearman: {best_metrics_in_epochs[0]:.5f}, Best MSE: {best_metrics_in_epochs[1]:.5f}")
    torch.cuda.empty_cache()
    
    return best_metrics_in_epochs[0], best_metrics_in_epochs[1], best_model



def train_model_baseline(X_train, y_train, X_test, y_test, epochs, seed, save_path, model_type, loss_func='MSE', alpha=1e-4, patience=10, batch_size=1024):
    device = _set_device_and_seed(seed)
    train_loader, test_loader, input_dim = _prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size, device)

    model = PLM_RankReg(input_dim, model_type=model_type)  
    model.to(device)
    
    criterion = get_loss_function(loss_func)
    optimizer = optim.Adam(model.parameters(), weight_decay=alpha)

    best_model = None  
    best_metrics_in_epochs = [float('-inf'), float('inf')]
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        test_mse, test_spearmanr = evaluate_mse_spearman(true_labels, predictions)
        # print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}, Spearman: {test_spearmanr:.5f}, MSE: {test_mse:.5f}")
        
        if test_spearmanr > best_metrics_in_epochs[0]:  
            best_metrics_in_epochs = [test_spearmanr, test_mse]  
            best_model = model.state_dict()
            patience_counter = 0
            if save_path:
                torch.save(best_model, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break
    print(f"Best Spearman: {best_metrics_in_epochs[0]:.5f}, Best MSE: {best_metrics_in_epochs[1]:.5f}")
    torch.cuda.empty_cache()
    
    return best_metrics_in_epochs[0], best_metrics_in_epochs[1], best_model


# Kaiyi Jiang et al. ,Rapid in silico directed evolution by a protein language model with EVOLVEpro.
# Science387,eadr6006(2025).DOI:10.1126/science.adr6006
# https://github.com/mat10d/EvolvePro
def train_evolvepro(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                    n_jobs=None, random_state=1, verbose=0, warm_start=False, ccp_alpha=0.0,
                                    max_samples=None)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_mse, test_spearmanr = evaluate_mse_spearman(y_test, y_pred_test)
    return test_spearmanr, test_mse, model