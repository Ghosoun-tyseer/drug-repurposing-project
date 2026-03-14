import pandas as pd

# تحميل التنبؤات
pred = pd.read_csv("result.csv", header=None)

# تحميل البيانات الأصلية
truth = pd.read_csv("../../../data/Cdataset/drug_dis.csv", header=None)

pairs = []

# جمع جميع الأزواج الممكنة (غير المرتبطة في truth)
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if truth.iloc[i, j] == 0:  # فقط الأزواج التي لم ترتبط مسبقًا
            pairs.append([i, j, pred.iloc[i, j]])

# إنشاء DataFrame للنتائج
df = pd.DataFrame(pairs, columns=["drug", "disease", "score"])

# ترتيب النتائج حسب التنبؤ من الأعلى للأسفل
df = df.sort_values("score", ascending=False)

# حفظ جميع النتائج في ملف CSV
df.to_csv("all_candidates_sorted.csv", index=False)

print("تم حفظ جميع النتائج مرتبة في all_candidates_sorted.csv")