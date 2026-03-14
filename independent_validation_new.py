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

def run_hybrid_blind_test(holdout_ratio=0.1):
    set_seed(args.seed)
    device = th.device(f'cuda:{args.device_id}' if th.cuda.is_available() else 'cpu')
    args.device = device 

    # 1. إعداد المجلد الجديد للنتائج
    output_dir = "../result/GAT_BlindTest_Experiment"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. تحميل البيانات وتجهيز "العمياء" (Blind Set)
    df_path = f'../data/{args.dataset}/drug_dis.csv'
    df = pd.read_csv(df_path, header=None).values
    pos_indices = np.argwhere(df == 1)
    all_neg_indices = np.argwhere(df == 0)
    
    num_holdout = int(len(pos_indices) * holdout_ratio)
    np.random.shuffle(pos_indices)
    
    # هذه العلاقات لن تدخل في التدريب إطلاقاً
    hidden_relations = pos_indices[:num_holdout]  
    # هذه العلاقات هي التي سنتدرب عليها
    train_available_pos = pos_indices[num_holdout:] 

    # 3. بناء الجراف المنقوص (إخفاء هيكلي)
    g, g_llm = load_dataset(args)
    g = remove_graph(g, hidden_relations).to(device)
    g_llm = remove_graph(g_llm, hidden_relations).to(device)
    
    features = generate_feat(args, [g, g_llm])
    
    # 4. حلقة التدريب (Training Phase)
    model = GAT_DDA(in_feats_drug=features['drug'].shape[1], 
                    in_feats_dis=features['disease'].shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = th.nn.BCEWithLogitsLoss()

    epochs = 1000 # يمكنك زيادتها لـ 2000 حسب حاجتك
    print(f"--- Starting Training on Available Relations ({len(train_available_pos)} samples) ---")

    for epoch in range(1, epochs + 1):
        model.train()
        # موازنة العينة داخل التدريب
        neg_sample_idx = all_neg_indices[np.random.choice(len(all_neg_indices), len(train_available_pos))]
        train_samples = np.concatenate([train_available_pos, neg_sample_idx], axis=0)
        train_labels = th.cat([th.ones(len(train_available_pos)), th.zeros(len(neg_sample_idx))]).to(device)

        current_g = g_llm if (args.BERT_emb or args.LLM_emb) else g
        logits = model(current_g, features)
        pred_scores = logits[train_samples[:, 0], train_samples[:, 1]]
        
        loss = criterion(pred_scores, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with th.no_grad():
                train_auc = roc_auc_score(train_labels.cpu(), th.sigmoid(pred_scores).cpu())
                print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.4f} | Train AUC: {train_auc:.4f}")

    # حفظ أوزان النموذج النهائي
    th.save(model.state_dict(), os.path.join(output_dir, "final_blind_model.pth"))

    # 5. التقييم المزدوج (Double Evaluation)
    model.eval()
    with th.no_grad():
        full_score_matrix = th.sigmoid(model(current_g, features)).cpu().numpy()

        # أ- تقييم العلاقات التي تدرب عليها (ماذا تعلم؟)
        train_pos_scores = np.array([full_score_matrix[r, c] for r, c in train_available_pos])
        # ب- تقييم العلاقات المخفية (ماذا اكتشف؟)
        hidden_pos_scores = np.array([full_score_matrix[r, c] for r, c in hidden_relations])
        # ج- تقييم عينة سلبية
        test_neg_idx = all_neg_indices[np.random.choice(len(all_neg_indices), len(hidden_relations))]
        negative_scores = np.array([full_score_matrix[r, c] for r, c in test_neg_idx])

        # حساب المقاييس للـ Blind Test فقط
        y_true_blind = np.array([1] * len(hidden_pos_scores) + [0] * len(negative_scores))
        y_pred_blind = np.concatenate([hidden_pos_scores, negative_scores])
        auc_b, aupr_b, acc_b, f1_b, pre_b, rec_b, spec_b = get_metrics(y_true_blind, y_pred_blind)

        # 6. حفظ التقارير والملفات
        # ملف السكورز للعلاقات المخفية
        pd.DataFrame({
            'Drug_ID': hidden_relations[:, 0],
            'Disease_ID': hidden_relations[:, 1],
            'Score': hidden_pos_scores,
            'Label': 1
        }).to_csv(os.path.join(output_dir, "hidden_discovery_scores.csv"), index=False)

        # ملف الخلاصة النصية
        with open(os.path.join(output_dir, "experiment_summary.txt"), "w") as f:
            f.write("HYBRID EXPERIMENT REPORT\n" + "="*30 + "\n")
            f.write(f"1. TRAINING STATS (Internal):\n")
            f.write(f"   - Average Score on Train Positives: {np.mean(train_pos_scores):.4f}\n\n")
            f.write(f"2. BLIND TEST STATS (External Discovery):\n")
            f.write(f"   - AUC: {auc_b:.4f} | AUPR: {aupr_b:.4f}\n")
            f.write(f"   - Accuracy: {acc_b:.4f} | F1: {f1_b:.4f}\n")
            f.write(f"   - Recall (Recovery): {rec_b:.4f}\n")
            f.write("-" * 30 + "\n")
            for t in [0.3, 0.5, 0.7]:
                rec_rate = (sum(1 for s in hidden_pos_scores if s > t) / len(hidden_pos_scores)) * 100
                f.write(f"Recovery at threshold {t}: {rec_rate:.2f}%\n")

        # رسم التوزيع للمقارنة
        plt.figure(figsize=(10, 6))
        plt.hist(hidden_pos_scores, bins=30, alpha=0.6, label='Hidden Positives (Discovery)', color='green')
        plt.hist(negative_scores, bins=30, alpha=0.4, label='True Negatives', color='red')
        plt.title(f'Blind Test Performance (AUC: {auc_b:.4f})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "discovery_plot.png"))

        print(f"\n[SUCCESS] All results saved in: {output_dir}")

if __name__ == '__main__':
    run_hybrid_blind_test(holdout_ratio=0.1)