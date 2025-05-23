import os
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

def create_histogram(pes_data, bins=100, range_min=-5000, range_max=5000):
    """将PES数据(N*N*N array)转换为直方图"""
    hist, _ = np.histogram(pes_data.flatten(), bins=bins, range=(range_min, range_max))
    return hist.astype(np.float32)  # 转换为浮点数

def load_data_and_create_features(csv_path, data_root, task_name, normalize=None):
    """加载数据并创建直方图特征"""
    df = pd.read_csv(csv_path, dtype={
        'matid': str
    })
    df = df.dropna(subset=[task_name])
    
    hist_features = []
    labels = []
    matids = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        matid = row['matid']
        label = row[task_name]
        
        # 加载PES数据并创建直方图特征
        pes_data = np.load(os.path.join(data_root, 'pes', f'{matid}.npy'))
        hist = create_histogram(pes_data)
        
        hist_features.append(hist)
        labels.append(label)
        matids.append(matid)
    
    # 转换为numpy数组
    X = np.stack(hist_features)
    y = np.array(labels)
    
    # 归一化标签
    if normalize:
        y = (y - normalize['mean']) / normalize['std']
    
    return X, y, matids

def train_and_evaluate(config):
    """训练并评估线性回归模型"""
    # 准备数据
    train_csv = os.path.join(config['data']['root_dir'], 'benchmark.train.csv')
    val_csv = os.path.join(config['data']['root_dir'], 'benchmark.val.csv')
    test_csv = os.path.join(config['data']['root_dir'], 'benchmark.test.csv')
    
    # 计算训练集的均值和标准差用于归一化
    train_df = pd.read_csv(train_csv)
    train_mean = train_df[config['train']['task_name']].mean()
    train_std = train_df[config['train']['task_name']].std()
    normalize_stats = {'mean': train_mean, 'std': train_std}
    
    print("Loading training data...")
    X_train, y_train, _ = load_data_and_create_features(
        train_csv, config['data']['root_dir'], config['train']['task_name'], normalize_stats
    )
    
    print("Loading validation data...")
    X_val, y_val, _ = load_data_and_create_features(
        val_csv, config['data']['root_dir'], config['train']['task_name'], normalize_stats
    )
    
    print("Loading test data...")
    X_test, y_test, _ = load_data_and_create_features(
        test_csv, config['data']['root_dir'], config['train']['task_name'], normalize_stats
    )
    
    # 训练线性回归模型
    print("Training linear regression model...")
    # model = LinearRegression()
    # model = Lasso()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 在验证集上评估
    y_val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    # 在测试集上评估
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\nEvaluation Results:")
    print(f"Validation R2: {val_r2:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    return {
        'val_r2': val_r2,
        'val_mae': val_mae,
        'test_r2': test_r2,
        'test_mae': test_mae
    }

def main(config_path: str) -> None:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results = train_and_evaluate(config)
    
    # # 保存结果
    # output_dir = config['train']['log_dir']
    # os.makedirs(output_dir, exist_ok=True)
    
    # with open(os.path.join(output_dir, 'linear_regression_results.txt'), 'w') as f:
    #     f.write(f"Validation R2: {results['val_r2']:.4f}\n")
    #     f.write(f"Validation MAE: {results['val_mae']:.4f}\n")
    #     f.write(f"Test R2: {results['test_r2']:.4f}\n")
    #     f.write(f"Test MAE: {results['test_mae']:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    
    main(args.config)
