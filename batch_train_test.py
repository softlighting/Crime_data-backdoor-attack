import torch
import numpy as np
import time
import os
import json
from engine import trainer, test
from Params import args
from utils import seed_torch, makePrint
from model import STHSL
from DataHandler import DataHandler

def train_model(dataset_name, device):
    """训练模型并返回最佳结果"""
    print(f"\n{'='*80}")
    print(f"Training on dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    # 设置数据集
    args.data = dataset_name
    args.device = device
    
    # 确保保存目录存在
    save_dir = os.path.join(args.save, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    seed_torch()
    engine = trainer(device)
    print("start training...", flush=True)
    
    train_time = []
    bestRes = None
    eval_bestRes = dict()
    eval_bestRes['RMSE'], eval_bestRes['MAE'], eval_bestRes['MAPE'] = 1e6, 1e6, 1e6
    update = False
    best_epoch = 0

    for i in range(1, args.epoch+1):
        t1 = time.time()
        metrics, metrics1 = engine.train()
        print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')
        ret = 'Epoch %d/%d, %s %.4f,  %s %.4f' % (i, args.epoch, 'Train Loss = ', metrics, 'preLoss = ', metrics1)
        print(ret)

        test = (i % args.tstEpoch == 0)
        if test:
            res_eval = engine.eval(True, True)
            val_metrics = res_eval['RMSE'] + res_eval['MAE']
            val_best_metrics = eval_bestRes['RMSE'] + eval_bestRes['MAE']
            if (val_metrics) < (val_best_metrics):
                print('%s %.4f, %s %.4f' % ('Val metrics decrease from', val_best_metrics, 'to', val_metrics))
                eval_bestRes['RMSE'] = res_eval['RMSE']
                eval_bestRes['MAE'] = res_eval['MAE']
                update = True
            reses = engine.eval(False, True)
            
            # 保存模型
            model_path = os.path.join(save_dir, f"_epoch_{i}_MAE_{round(reses['MAE'], 2)}_MAPE_{round(reses['MAPE'], 2)}.pth")
            torch.save(engine.model.state_dict(), model_path)
            
            if update:
                print(makePrint('Test', i, reses))
                bestRes = reses.copy()
                best_epoch = i
                best_model_path = model_path
                update = False
        print()
        t2 = time.time()
        train_time.append(t2-t1)
    
    print(makePrint('Best', args.epoch, bestRes))
    return bestRes, best_model_path, best_epoch


def test_model(dataset_name, checkpoint_path, device):
    """测试模型并返回结果"""
    print(f"\n{'='*80}")
    print(f"Testing on dataset: {dataset_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # 设置数据集
    args.data = dataset_name
    args.device = device
    args.checkpoint = checkpoint_path
    
    handler = DataHandler()
    model = STHSL()
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print('model load successfully')

    with torch.no_grad():
        reses = test(model, handler)

    print(makePrint('Test', args.epoch, reses))
    return reses


def main():
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # 定义所有要训练和测试的数据集
    datasets = [
        'NYC',  # 原始数据集
        'NYC_spatial_hyperedge',  # Attack 1
        'NYC_temporal_pattern',   # Attack 2
        'NYC_cross_category',     # Attack 3
    ]
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n\n{'#'*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'#'*80}\n")
        
        try:
            # 训练模型
            train_start = time.time()
            best_res, best_model_path, best_epoch = train_model(dataset_name, device)
            train_time = time.time() - train_start
            
            # 测试模型
            test_start = time.time()
            test_res = test_model(dataset_name, best_model_path, device)
            test_time = time.time() - test_start
            
            # 保存结果
            results[dataset_name] = {
                'training': {
                    'best_epoch': best_epoch,
                    'best_model_path': best_model_path,
                    'metrics': best_res,
                    'training_time': train_time
                },
                'testing': {
                    'metrics': test_res,
                    'testing_time': test_time
                }
            }
            
            print(f"\n✓ Completed {dataset_name}")
            print(f"  Training time: {train_time:.2f}s")
            print(f"  Testing time: {test_time:.2f}s")
            
        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'error': str(e)}
    
    # 保存所有结果到JSON文件
    results_file = 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nAll results saved to {results_file}")
    
    # 打印汇总结果
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"{dataset_name}: ERROR - {result['error']}")
            continue
            
        print(f"\n{dataset_name}:")
        print(f"  Best Epoch: {result['training']['best_epoch']}")
        print(f"  Training Metrics:")
        train_metrics = result['training']['metrics']
        print(f"    RMSE: {train_metrics.get('RMSE', 'N/A'):.4f}")
        print(f"    MAE: {train_metrics.get('MAE', 'N/A'):.4f}")
        print(f"    MAPE: {train_metrics.get('MAPE', 'N/A'):.4f}")
        print(f"  Testing Metrics:")
        test_metrics = result['testing']['metrics']
        print(f"    RMSE: {test_metrics.get('RMSE', 'N/A'):.4f}")
        print(f"    MAE: {test_metrics.get('MAE', 'N/A'):.4f}")
        print(f"    MAPE: {test_metrics.get('MAPE', 'N/A'):.4f}")
        print(f"  Training Time: {result['training']['training_time']:.2f}s")
        print(f"  Testing Time: {result['testing']['testing_time']:.2f}s")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"\nTotal time spent: {t2 - t1:.2f}s ({((t2 - t1)/60):.2f} minutes)")

