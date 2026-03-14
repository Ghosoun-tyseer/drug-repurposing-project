#هذا الكود لاخفاء علاقات قبل تدريب النموذج ثم اختباره بها بعد تدريبه لمعرفة اذا كان سينجح في التنبؤ بعلاقات لم يتدرب عليها من قبل 

import os
import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_auc_score
from model_gat import GAT_DDA
from load_data import load_dataset, generate_feat, remove_graph
from utils import set_seed, get_metrics
from args import args

warnings.filterwarnings('ignore')

def run_blind_test(holdout_ratio=0.1): # جعلناها 0.1 بناءً على تجربتك الناجحة
    set_seed(args.seed)
    device = th.device(f'cuda:{args.device_id}' if th.cuda.is_available() else 'cpu')
    args.device = device 
    
    # 1. تحميل البيانات الأصلية
    df_path = f'../data/{args.dataset}/drug_dis.csv'
    df = pd.read_csv(df_path, header=None).values
    pos_indices = np.argwhere(df == 1)
    all_neg_indices = np.argwhere(df == 0)
    
    # 2. إخفاء العلاقات (Blind Set)
    num_holdout = int(len(pos_indices) * holdout_ratio)
    np.random.shuffle(pos_indices)
    hidden_relations = pos_indices[:num_holdout]  
    train_pos_idx = pos_indices[num_holdout:] 
    
    # 3. بناء الجراف المنقوص (إخفاء هيكلي تام)
    g, g_llm = load_dataset(args)
    g = remove_graph(g, hidden_relations).to(device)
    g_llm = remove_graph(g_llm, hidden_relations).to(device)
    
    features = generate_feat(args, [g, g_llm])
    model = GAT_DDA(in_feats_drug=features['drug'].shape[1], 
                    in_feats_dis=features['disease'].shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = th.nn.BCEWithLogitsLoss()

    # 4. التدريب المتوازن مع زيادة الـ Epochs يدوياً
    epochs_to_run = 1000 # التعديل المطلوب لرفع الـ Recovery Rate
    print(f"Training started for {epochs_to_run} epochs with Balanced Sampling...")
    
    model.train()
    for epoch in range(1, epochs_to_run + 1):
        # موازنة البيانات في كل Epoch
        neg_sample_idx = all_neg_indices[np.random.choice(len(all_neg_indices), len(train_pos_idx))]
        train_samples = np.concatenate([train_pos_idx, neg_sample_idx], axis=0)
        train_labels = th.cat([th.ones(len(train_pos_idx)), th.zeros(len(neg_sample_idx))]).to(device)

        current_g = g_llm if (args.BERT_emb or args.LLM_emb) else g
        score_matrix = model(current_g, features)
        
        pred_scores = score_matrix[train_samples[:, 0], train_samples[:, 1]]
        loss = criterion(pred_scores, train_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            with th.no_grad():
                auc_iter = roc_auc_score(train_labels.cpu(), th.sigmoid(pred_scores).cpu())
                print(f"Epoch {epoch:4d}/{epochs_to_run} | Loss: {loss.item():.4f} | Balanced Train AUC: {auc_iter:.4f}")

    # 5. التقييم النهائي المستقل (Blind Test)
    model.eval()
    with th.no_grad():
        full_score = th.sigmoid(model(current_g, features)).cpu().numpy()
        
        # سكور العلاقات التي لم يرها الموديل أبداً
        hidden_scores = np.array([full_score[r, c] for r, c in hidden_relations])
        
        # سكور عينة من الأصفار للمقارنة العادلة
        test_neg_idx = all_neg_indices[np.random.choice(len(all_neg_indices), len(hidden_scores))]
        negative_scores = np.array([full_score[r, c] for r, c in test_neg_idx])
        
        y_true = [1] * len(hidden_scores) + [0] * len(negative_scores)
        y_pred = np.concatenate([hidden_scores, negative_scores])
        blind_auc = roc_auc_score(y_true, y_pred)

        print("\n" + "="*60)
        print(f"FINAL BLIND TEST RESULTS (Ratio: {holdout_ratio})")
        print(f"STRICT BLIND TEST AUC: {blind_auc:.4f}")
        print("-" * 30)
        # حساب Recovery Rate عند مستويات مختلفة للثقة
        for threshold in [0.3, 0.4, 0.5]:
            recovery = (sum(1 for s in hidden_scores if s > threshold) / num_holdout) * 100
            print(f"Recovery Rate (Score > {threshold}): {recovery:.2f}%")
        print("="*60)

        # حفظ النتائج التفصيلية
        pd.DataFrame({
            'Drug_ID': hidden_relations[:, 0],
            'Disease_ID': hidden_relations[:, 1],
            'Confidence_Score': hidden_scores
        }).to_csv("blind_test_results.csv", index=False)

        # رسم التوزيع البياني
        plt.figure(figsize=(10, 5))
        plt.hist(hidden_scores, bins=30, alpha=0.7, color='blue', label='Hidden Relations (Positives)')
        plt.hist(negative_scores, bins=30, alpha=0.5, color='red', label='True Negatives')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold 0.5')
        plt.title(f'Blind Test Distribution (AUC: {blind_auc:.4f})')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('blind_test_final_plot.png')
        print(f"\n[SAVED] CSV: blind_test_results.csv | Plot: blind_test_final_plot.png")

if __name__ == '__main__':
    run_blind_test(holdout_ratio=0.1)