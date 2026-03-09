import os
import sys
import logging
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import KFold
from warnings import simplefilter

# ==========================================================
# 🔹 المرحلة 1: إعداد المسارات والوحدات (Modules Configuration)
# ==========================================================

# الحصول على المسار المطلق لمجلد codes الحالي
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)

# إضافة مجلد codes إلى sys.path لضمان استدعاء الملفات المحلية
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# استدعاء الملفات البرمجية الخاصة بالمشروع
from model import Model
from load_data import load_dataset, remove_graph, generate_feat
from utils import define_logging, get_metrics_auc, set_seed, get_metrics
from args import args

# ==========================================================
# 🔹 المرحلة 2: تحديد مجلد المشروع الرئيسي بدقة (Root Detection)
# ==========================================================

# سنقوم بالبحث عن مجلد المشروع بالاسم لضمان عدم القفز لسطح المكتب
def get_project_root(current_path):
    # نتحقق إذا كان المجلد الحالي هو Drug Repurposing project
    if "Drug Repurposing project" in current_path:
        # نقوم بتقسيم المسار ونأخذ الجزء الذي ينتهي باسم المشروع
        parts = current_path.split("Drug Repurposing project")
        return parts[0] + "Drug Repurposing project"
    return os.path.dirname(current_path) # حل احتياطي

PROJECT_ROOT = get_project_root(CURRENT_DIR)

# إعداد مسار الحفظ في مجلد results
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", args.dataset + "_hidden_eval")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

# ==========================================================
# 🔹 المرحلة 3: تحميل البيانات (المسار المصحح)
# ==========================================================

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', args.dataset, 'drug_dis.csv')

print(f"🔍 Searching for data in: {DATA_FILE}")

if not os.path.exists(DATA_FILE):
    # محاولة أخيرة باستخدام المسار المباشر من المجلد الحالي
    DATA_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", args.dataset, "drug_dis.csv"))
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ لم يتم العثور على الملف. تأكدي من وجود مجلد data داخل Drug Repurposing project.\nالمسار المفحوص: {DATA_FILE}")

# تحميل المصفوفة
try:
    df = pd.read_csv(DATA_FILE, header=None).values
    print("✅ File loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit()

data_pos = np.array(np.where(df == 1)).T 

# 1. إخفاء روابط للتقييم
np.random.seed(args.seed)
num_hide = int(len(data_pos) * 0.1) # HIDE_RATIO = 0.1
hide_idx = np.random.choice(len(data_pos), size=num_hide, replace=False)
hidden_links = data_pos[hide_idx]

# 2. نسخة التدريب بدون المخفي
df_train = df.copy() 
for (i, j) in hidden_links:
    df_train[i, j] = 0  

# 3. تجهيز البيانات للـ K-Fold
data_train_list = []
for i in range(df_train.shape[0]):
    for j in range(df_train.shape[1]):
        data_train_list.append([i, j, df_train[i, j]])
data_train_arr = np.array(data_train_list)

data_pos_train = data_train_arr[data_train_arr[:, -1] == 1]
data_neg_train = data_train_arr[data_train_arr[:, -1] == 0]

# ==========================================================
# 🔹 المرحلة 4: التدريب (K-Fold Training)
# ==========================================================

set_seed(args.seed)
simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger("hidden_eval")
logger.setLevel(logging.INFO)
define_logging(args, logger)

kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
pred_result = np.zeros(df.shape)
fold = 1

for (t_pos_idx, v_pos_idx), (t_neg_idx, v_neg_idx) in zip(kf.split(data_pos_train), kf.split(data_neg_train)):
    logger.info(f"--- Fold {fold}/{args.nfold} ---")

    test_pos_id = data_pos_train[v_pos_idx]
    test_pos_coords = (tuple(test_pos_id[:, 0].astype(int)), tuple(test_pos_id[:, 1].astype(int)))
    test_neg_id = data_neg_train[v_neg_idx]
    test_neg_coords = (tuple(test_neg_id[:, 0].astype(int)), tuple(test_neg_id[:, 1].astype(int)))

    # تحميل الرسم البياني
    g, g_llm = load_dataset(args)
    g = remove_graph(g, test_pos_id).to(args.device)
    g_llm = remove_graph(g_llm, test_pos_id).to(args.device)

    feature = generate_feat(args, [g, g_llm])
    
    mask_test_matrix = np.zeros(df.shape)
    mask_test_matrix[test_pos_coords[0], test_pos_coords[1]] = 1
    mask_test_matrix[test_neg_coords[0], test_neg_coords[1]] = 1
    
    mask_test_idx = np.where(mask_test_matrix == 1)
    mask_train_idx = np.where(mask_test_matrix == 0)

    label_tensor = th.tensor(df_train).float().to(args.device)

    # بناء الموديل
    in_feats_list = [feature['drug'].shape[1], feature['disease'].shape[1]]
    if args.concatenate_type not in ['none', 'as_node']:
        in_feats_list += [feature['drug_LLM'].shape[1], feature['disease_LLM'].shape[1]]

    model = Model(args=args, etypes=g.etypes, ntypes=g.ntypes, in_feats=in_feats_list)
    model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(len(data_neg_train[t_neg_idx]) / len(data_pos_train[t_pos_idx])))

    for epoch in range(1, args.epoch + 1):
        model.train()
        score = model([g, g_llm], feature)
        loss = criterion(score[mask_train_idx].flatten(), label_tensor[mask_train_idx].flatten())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            model.eval()
            with th.no_grad():
                pred = th.sigmoid(score)
                auc_val, _ = get_metrics_auc(label_tensor[mask_test_idx].cpu().numpy(), pred[mask_test_idx].cpu().numpy())
                logger.info(f"Epoch {epoch}: Loss {loss.item():.4f} | Val AUC {auc_val:.4f}")

    model.eval()
    with th.no_grad():
        final_score = model([g, g_llm], feature)
        pred_result[mask_test_idx] = th.sigmoid(final_score).cpu().detach().numpy()[mask_test_idx]
        
    th.save(model.state_dict(), os.path.join(SAVE_PATH, f"model_fold_{fold}.pth"))
    fold += 1

# ==========================================================
# 🔹 المرحلة 5: التقييم النهائي وحفظ النتائج
# ==========================================================

hidden_preds = pred_result[tuple(hidden_links.T.astype(int))]
hidden_labels = np.ones(len(hidden_links))

AUC_hidden, AUPR_hidden = get_metrics_auc(hidden_labels, hidden_preds)
logger.info(f"✅ Final Hidden Links AUC: {AUC_hidden:.4f}")

output_csv = os.path.join(SAVE_PATH, 'final_prediction.csv')
pd.DataFrame(pred_result).to_csv(output_csv, index=False, header=False)
logger.info(f"💾 Results saved in: {SAVE_PATH}")