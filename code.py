#!/usr/bin/env python
# coding: utf-8

# In[74]:


import math
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# In[75]:


plt.rcParams['font.sans-serif'] = ['Times New Roman'] 
plt.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.size'] = 14
plt.rcParams['font.serif'] = 'Times New Roman'
warnings.filterwarnings("ignore")


# In[76]:


import math
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import shap

# 打印版本信息
print(f"math: Built-in (No Version)")
print(f"warnings: Built-in (No Version)")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"seaborn: {sns.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"sklearn: {sklearn.__version__}")
print(f"shap: {shap.__version__}")


# In[77]:


data = pd.read_csv('C:/Users/lenovo/Desktop/data.csv')


# In[78]:


data


# In[79]:


data['AHD'].value_counts()


# In[80]:


# 划分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1], test_size=0.3, random_state=0)


# In[81]:


data_trian = pd.concat([X_train,y_train],axis=1)


# In[82]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
def load_and_preprocess_data(data):
    """Load and preprocess data"""
    try:
        X = data.drop(columns=['AHD'])
        y = data['AHD']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return data, X, X_scaled, y
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None, None, None, None


# In[83]:


def perform_lasso_cv(X_scaled, y):
    """Perform LassoCV cross-validation"""
    alphas = np.logspace(-3, 0, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=0)
    lasso_cv.fit(X_scaled, y)
    return lasso_cv


# In[84]:


def save_selected_features(data, selected_features, file_path):
    """Save selected features"""
    final_data = data[selected_features.to_list() + ['AHD']]
    final_data.to_csv(file_path, index=False)
    print(f"Selected features have been saved to {file_path}")


# In[85]:


data_trian, X, X_scaled, y = load_and_preprocess_data(data_trian)
lasso_cv = perform_lasso_cv(X_scaled, y)
optimal_alpha = lasso_cv.alpha_


# In[86]:


lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
lasso.fit(X_scaled, y)
selected_features = X.columns[lasso.coef_ != 0]
print("Features selected by Lasso:", selected_features)


# In[87]:


alphas = np.logspace(-3, 0, 100)
num_features = []
for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, max_iter=10000)
    lasso_temp.fit(X_scaled, y)
    num_nonzero_coeff = np.sum(lasso_temp.coef_ != 0)
    num_features.append(num_nonzero_coeff)


# In[88]:


# 设置字体类型为TrueType (42)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# In[89]:


# 提取每个 alpha 值对应的均方误差 (MSE)
mse_path = np.mean(lasso_cv.mse_path_, axis=1)  # 取交叉验证的平均误差
std_error = np.std(lasso_cv.mse_path_, axis=1)  # 计算标准差
# 获取最佳的 alpha 值（即均方误差最小的alpha）
best_alpha = lasso_cv.alpha_

# 计算1-SE（1标准误差）规则下的 alpha 值
# 1-SE规则的alpha值是指在误差最小值的基础上，增加一个标准误差以内的最大alpha值
min_mse_idx = np.argmin(mse_path)
one_se_threshold = mse_path[min_mse_idx] + std_error[min_mse_idx]
# 找到第一个大于 1-SE 阈值的 alpha 值
alpha_1se =0.0309233002565047
# 绘制 LassoCV 的 MSE 图
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(np.log10(lasso_cv.alphas_), mse_path, 'r.-')  # 横轴是 alpha 的 log 值
ax1.errorbar(np.log10(lasso_cv.alphas_), mse_path, yerr=lasso_cv.mse_path_.std(axis=1), fmt='o', color='b', alpha=0.7)
ax1.axvline(x=np.log10(lasso_cv.alpha_), linestyle='--', color='k')
ax1.axvline(x=np.log10(alpha_1se), linestyle='--', color='k')
ax1.set_xlabel("log(λ)")
ax1.set_ylabel("Binomial Deviance")

ax2 = ax1.secondary_xaxis('top')

# 选择性地减少顶部显示的特征数量刻度
selected_indices = np.linspace(0, len(lasso_cv.alphas_) - 1, num=9, dtype=int) 
selected_alphas = np.log10(lasso_cv.alphas_[selected_indices])
selected_features = np.array(num_features)[selected_indices][::-1]

# 设置次坐标轴的刻度为log(alpha)并显示特征数量
ax2.set_xticks(selected_alphas)
ax2.set_xticklabels(selected_features)
plt.grid(True)

plt.show()


# In[90]:


# 获取路径上的系数和alpha值
alphas_lasso = lasso_cv.alphas_
coefficients_lasso = []
for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    coefficients_lasso.append(lasso.coef_)

coefficients_lasso = np.array(coefficients_lasso)

# 绘制Lasso回归系数与正则化参数的关系图
plt.figure(figsize=(8, 5))
for i in range(coefficients_lasso.shape[1]):
    plt.plot(np.log10(alphas_lasso), coefficients_lasso[:, i], label=f'Feature {i+1}')

plt.axhline(0, color='green', linestyle='--')
plt.xlabel('log(λ)')
plt.ylabel('Coefficients')
plt.grid(True)

plt.show()


# In[91]:


lasso = Lasso(alpha=alpha_1se, max_iter=10000)
lasso.fit(X_scaled, y)
selected_features = X.columns[lasso.coef_ != 0]
print("Features selected by Lasso:", selected_features)


# In[92]:


X_selected = data[[ 'Height', 'Waist', 'LEG', 'ALP', 'Gender']]


# In[93]:


X_selected['AHD'] = data['AHD']


# In[94]:


#划分数据集
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X_selected.iloc[:,:-1],X_selected.iloc[:,-1],train_size=0.7,random_state=2)


# In[95]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)  # 初始化 SMOTE
train_X_resampled, train_Y_resampled = smote.fit_resample(train_X, train_Y)


# In[96]:


train_X,train_Y = train_X_resampled, train_Y_resampled


# In[97]:


#混淆矩阵图
from sklearn import metrics
def plot_confusion_matrix(train_Y,log_pre1,model_name,dataset_name):
    confusion_matrix = metrics.confusion_matrix(train_Y,log_pre1)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
    cm_display.plot(cmap='Blues')
    plt.tight_layout()


# In[98]:


# 求模型的评价指标
from sklearn.metrics import recall_score,f1_score,accuracy_score,auc,precision_score,roc_curve
from sklearn import metrics
def metrict_score(train_Y,log_pre1,train_scaler,model_name,dataset_name,pre_pro):
    
    print('{}模型{}——召回率：{}'.format(model_name,dataset_name,recall_score(train_Y,log_pre1)))
    print('{}模型{}——精确率：{}'.format(model_name,dataset_name,precision_score(train_Y,log_pre1)))
    print('{}模型{}——准确率：{}'.format(model_name,dataset_name,accuracy_score(train_Y,log_pre1, )))
    print('{}模型{}——F1值：{}'.format(model_name,dataset_name,f1_score(train_Y,log_pre1, )))
    confusion_matrix = metrics.confusion_matrix(train_Y,log_pre1)
    specificity = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
    print('{}模型{}——特异度：{}'.format(model_name,dataset_name,specificity))
    TN, FP, FN, TP = confusion_matrix.ravel()
    print('{}模型{}——阳性预测值：{}'.format(model_name,dataset_name,TP / (TP + FP)))
    print('{}模型{}——阴性预测值：{}'.format(model_name,dataset_name,TN / (TN + FN)))
    fpr,tpr,thresholds = roc_curve(train_Y,pre_pro)
    print('{}模型{}——AUC面积：{}'.format(model_name,dataset_name,auc(fpr, tpr)))


# In[99]:


def plot_roc(model,y_scores,y_true,model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_point = (fpr[optimal_idx], tpr[optimal_idx])
    
    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line color changed to red
    plt.scatter(optimal_point[0], optimal_point[1], marker='o', color='red')
    plt.text(optimal_point[0] + 0.03, optimal_point[1] - 0.05, f'{optimal_threshold:.3f} ({optimal_point[0]:.3f},{optimal_point[1]:.3f})', fontsize=12, verticalalignment='center', color='blue')
    plt.plot([optimal_point[0], optimal_point[0]], [optimal_point[1], optimal_point[1]], 'k--', color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC CURVE')
    plt.text(0.7, 0.4, f'AUC = {roc_auc:.3f}', ha='right', color='blue')


# In[100]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[101]:


param_grid = {'max_depth': np.arange(3,10,1),
        'eta':np.arange(0.01,0.5,0.3),
              'n_estimators':np.arange(10,200,10),
        }


# In[102]:


xgb = XGBClassifier(objective='binary:logistic',eval_metric='logloss', nthread=10,random_state=1)
reg = GridSearchCV(xgb,
                   param_grid,
                   scoring='roc_auc',
                   cv=5,verbose = 1).fit(train_X,train_Y)
best_alpha = reg.best_params_
best_score = reg.best_score_


# In[103]:


best_alpha


# In[104]:


best_score


# In[105]:


xgb = XGBClassifier(objective='binary:logistic', nthread=10,random_state=1,eta=0.31,max_depth=9,n_estimators=100)
xgb.fit(train_X, train_Y)
predict1 = xgb.predict(test_X)
predict2 = xgb.predict(train_X)


# In[106]:


plot_confusion_matrix(train_Y,predict2,'xgboost','训练集')
# 保存为 PDF 格式
plt.savefig('d:/xg训练1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/xg训练1.tif', bbox_inches='tight', dpi=600)


# In[107]:


metrict_score(train_Y,predict2,train_X,'xgboost','训练集',xgb.predict_proba(train_X)[:, 1])


# In[108]:


plot_roc(xgb,xgb.predict_proba(train_X)[:, 1],train_Y,'xgboost训练集')
# 保存为 PDF 格式
plt.savefig('d:/xg训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/xg训练2.tif', bbox_inches='tight', dpi=600)


# In[109]:


plot_confusion_matrix(test_Y,predict1,'xgboost','测试集')
# 保存为 PDF 格式
plt.savefig('d:/0/xg测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/0/xg测试2.tif', bbox_inches='tight', dpi=600)


# In[110]:


metrict_score(test_Y,predict1,test_X,'xgboost','测试集',xgb.predict_proba(test_X)[:, 1])


# In[111]:


plot_roc(xgb,xgb.predict_proba(test_X)[:, 1],test_Y,'xgboost测试集')
# 保存为 PDF 格式
plt.savefig('d:/xg测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/xg测试1.tif', bbox_inches='tight', dpi=600)


# In[112]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = xgb.predict_proba(test_X)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# # 逻辑回归

# In[113]:


#标准化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaler = scaler.fit_transform(train_X)
test_scaler = scaler.transform(test_X)


# In[114]:


C = np.logspace(-4, 4, 50)
parameters = dict(C=C)


# In[115]:


#网格搜索
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
#首先利用网格搜索方法

# 网格搜索方式
logistic_Reg = linear_model.LogisticRegression()
reg = GridSearchCV(logistic_Reg,
                   parameters,
                   scoring='roc_auc',
                   cv=5,verbose = 1).fit(train_scaler,train_Y)

best_alpha = reg.best_params_
best_score = reg.best_score_


# In[116]:


best_alpha


# In[117]:


best_score


# In[118]:


logistic_Reg = linear_model.LogisticRegression(C=best_alpha['C'])
logistic_Reg.fit(train_scaler, train_Y)
log_pre = logistic_Reg.predict(test_scaler)
log_pre1 = logistic_Reg.predict(train_scaler)


# In[119]:


plot_confusion_matrix(train_Y,log_pre1,'逻辑回归','训练集')
# 获取当前图表对象
current_figure = plt.gcf()
plt.draw()


# In[122]:


from sklearn.metrics import auc  # 重新导入，覆盖之前被赋值的auc变量

metrict_score(train_Y,log_pre1,train_scaler,'逻辑回归','训练集',logistic_Reg.predict_proba(train_scaler)[:, 1])


# In[123]:


plot_roc(logistic_Reg,logistic_Reg.predict_proba(train_scaler)[:, 1],train_Y,'逻辑回归训练集')


# In[124]:


plot_confusion_matrix(test_Y,log_pre,'逻辑回归','测试集')


# In[125]:


metrict_score(test_Y,log_pre,test_scaler,'逻辑回归','测试集',logistic_Reg.predict_proba(test_scaler)[:, 1])


# In[126]:


plot_roc(logistic_Reg,logistic_Reg.predict_proba(test_scaler)[:, 1],test_Y,'逻辑回归测试集')


# In[70]:


featurfeature_cols = ['Height', 'Waist', 'LEG', 'ALP', 'Gender']
target_col = 'AHD'
group_col = 'Race'

# 1. 划分训练集和测试集（注意 stratify 保持类别分布一致）
train_data, test_data = train_test_split(
    data,
    test_size=0.3,
    random_state=42,
    stratify=data[target_col]
)

# 2. 提取特征和目标变量
train_X = train_data[feature_cols]
train_y = train_data[target_col]
test_X = test_data[feature_cols]
test_y = test_data[target_col]

# 3. 标准化：用 MinMaxScaler 将特征缩放到 [0, 1]
scaler = MinMaxScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

# 4. 建立逻辑回归模型并训练
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(train_X_scaled, train_y)

# 5. 对测试集整体评估（可选）
overall_pred = logistic_model.predict_proba(test_X_scaled)[:, 1]
overall_auc = roc_auc_score(test_y, overall_pred)
print(f"Overall Test AUC: {overall_auc:.3f}\n")

# 6. 在测试集上按种族分组做亚组分析
print("=== 亚组分析：按 Race 分组 ===")
for race_val in sorted(test_data[group_col].unique()):
    sub_data = test_data[test_data[group_col] == race_val]
    sub_X = sub_data[feature_cols]
    sub_y = sub_data[target_col]

    # 标准化（用训练集的 scaler）
    sub_X_scaled = scaler.transform(sub_X)

    # 预测 & 计算 AUC
    if len(sub_y) > 10 and len(sub_y.unique()) > 1:
        sub_pred = logistic_model.predict_proba(sub_X_scaled)[:, 1]
        auc_sub = roc_auc_score(sub_y, sub_pred)
        print(f"Race {race_val}: n={len(sub_y)}, positives={sum(sub_y)}, AUC={auc_sub:.3f}")
    else:
        print(f"Race {race_val}: 样本不足或标签单一，无法计算 AUC")



# In[71]:


# 6. Bootstrap 置信区间计算函数
def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.95):
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    y_true = y_true.reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true.iloc[indices])) < 2:
            # 跳过标签单一的bootstrap样本
            continue
        score = roc_auc_score(y_true.iloc[indices], y_score.iloc[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    mean_score = np.mean(sorted_scores)

    return mean_score, lower, upper

# 7. 按种族分组计算AUC及置信区间，保存结果
results = []
for race_val in sorted(test_data[group_col].unique()):
    sub_data = test_data[test_data[group_col] == race_val]
    sub_X = sub_data[feature_cols]
    sub_y = sub_data[target_col]

    # 标准化
    sub_X_scaled = scaler.transform(sub_X)

    # 计算AUC及置信区间（样本数量足够且标签不单一时）
    if len(sub_y) > 10 and len(sub_y.unique()) > 1:
        sub_pred = logistic_model.predict_proba(sub_X_scaled)[:, 1]
        mean_auc, lower_ci, upper_ci = bootstrap_auc_ci(sub_y, sub_pred, n_bootstraps=1000, alpha=0.95)
        results.append({
            'Race': race_val,
            'Samples': len(sub_y),
            'Positives': sum(sub_y),
            'AUC': mean_auc,
            'Lower_CI_95%': lower_ci,
            'Upper_CI_95%': upper_ci
        })
        print(f"Race {race_val}: n={len(sub_y)}, positives={sum(sub_y)}, AUC={mean_auc:.3f}, 95% CI=({lower_ci:.3f}, {upper_ci:.3f})")
    else:
        print(f"Race {race_val}: 样本不足或标签单一，无法计算 AUC")

# 8. 结果整理为DataFrame并导出
df_results = pd.DataFrame(results)
df_results.to_csv('race_subgroup_auc_with_ci.csv', index=False)
print("\n已保存种族亚组AUC及置信区间结果到 'race_subgroup_auc_with_ci.csv'")


# In[72]:


import matplotlib.pyplot as plt

# 收集 AUC 结果用于画图
race_auc_list = []
race_name_list = []

for race_val in sorted(test_data[group_col].unique()):
    sub_data = test_data[test_data[group_col] == race_val]
    sub_X = sub_data[feature_cols]
    sub_y = sub_data[target_col]
    sub_X_scaled = scaler.transform(sub_X)

    if len(sub_y) > 10 and len(sub_y.unique()) > 1:
        sub_pred = logistic_model.predict_proba(sub_X_scaled)[:, 1]
        auc_sub = roc_auc_score(sub_y, sub_pred)
        race_auc_list.append(auc_sub)
        race_name_list.append(f"Race {race_val}")
    else:
        race_auc_list.append(None)
        race_name_list.append(f"Race {race_val}")

# 画图
plt.figure(figsize=(8, 5))
plt.bar(race_name_list, race_auc_list, color='skyblue')
plt.ylim(0.0, 1.0)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.title('AUC Comparison by Race')
plt.ylabel('AUC')
plt.xlabel('Race Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, 'auc_by_race1.tiff')

plt.savefig(save_path, dpi=300)

plt.show()


# In[57]:


def categorize_poverty(pir):
    if pir <= 1.30:
        return 'Low'
    elif pir <= 3.50:
        return 'Middle'
    else:
        return 'High'

data['Poverty_Level'] = data['PIR'].apply(categorize_poverty)

# ==== 2. 模型训练设置 ====
feature_cols = ['Height', 'Waist', 'LEG', 'ALP', 'Gender']
target_col = 'AHD'
group_col = 'Poverty_Level'

# ==== 3. 划分训练集/测试集 ====
train_data, test_data = train_test_split(
    data,
    test_size=0.3,
    random_state=42,
    stratify=data[target_col]
)

train_X = train_data[feature_cols]
train_y = train_data[target_col]
test_X = test_data[feature_cols]
test_y = test_data[target_col]

# ==== 4. 标准化 ====
scaler = MinMaxScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

# ==== 5. 模型训练 ====
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(train_X_scaled, train_y)

# ==== 6. 整体 AUC ====
overall_pred = logistic_model.predict_proba(test_X_scaled)[:, 1]
overall_auc = roc_auc_score(test_y, overall_pred)
print(f"Overall Test AUC: {overall_auc:.3f}\n")

# ==== 7. 置信区间函数 ====
def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.95):
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    y_true = y_true.reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue
        score = roc_auc_score(y_true.iloc[indices], y_score.iloc[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    mean_score = np.mean(sorted_scores)

    return mean_score, lower, upper

# ==== 8. 亚组分析按 Poverty_Level ====
print("=== 亚组分析：按 Poverty_Level 分组 ===")
results = []

for group_val in sorted(test_data[group_col].unique()):
    sub_data = test_data[test_data[group_col] == group_val]
    sub_X = sub_data[feature_cols]
    sub_y = sub_data[target_col]
    sub_X_scaled = scaler.transform(sub_X)

    if len(sub_y) > 10 and len(sub_y.unique()) > 1:
        sub_pred = logistic_model.predict_proba(sub_X_scaled)[:, 1]
        mean_auc, lower_ci, upper_ci = bootstrap_auc_ci(sub_y, sub_pred)
        print(f"{group_val}: n={len(sub_y)}, positives={sum(sub_y)}, AUC={mean_auc:.3f}, 95% CI=({lower_ci:.3f}, {upper_ci:.3f})")

        results.append({
            'Poverty_Level': group_val,
            'Samples': len(sub_y),
            'Positives': sum(sub_y),
            'AUC': mean_auc,
            'Lower_95CI': lower_ci,
            'Upper_95CI': upper_ci
        })
    else:
        print(f"{group_val}: 样本不足或标签单一，无法计算 AUC")

# ==== 9. 保存结果为 CSV ====
df_poverty_results = pd.DataFrame(results)
df_poverty_results.to_csv('poverty_subgroup_auc_with_ci.csv', index=False)
print("\n已保存贫困亚组分析结果到 'poverty_subgroup_auc_with_ci.csv'")


# In[67]:


import matplotlib.pyplot as plt
import os

# Poverty level mapping dictionary (PIR)
pir_mapping = {
    1: 'Low',
    2: 'Middle',
    3: 'High'
}

# Example group data (PIR values) and corresponding metric values (e.g., AUC or OR)
# Replace these lists with your actual data
pir_values = [1, 2, 3]
metric_values = [0.72, 0.78, 0.65]  # e.g., AUC or OR values

# Generate labels for x-axis
labels = [pir_mapping.get(pir, f"PIR {pir}") for pir in pir_values]

plt.figure(figsize=(6, 5))
plt.bar(labels, metric_values, color='lightgreen')
plt.ylim(0, 1)  # Adjust depending on metric range, e.g., AUC ranges from 0 to 1
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.title('Subgroup analysis by PIR ')
plt.ylabel('AUC')
plt.xlabel('Poverty Level Group')
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure to Desktop in TIFF format
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, 'metric_by_poverty.tiff')
print(f"Saving figure to: {save_path}")
plt.savefig(save_path, dpi=300)
plt.show()


# In[73]:


import matplotlib.pyplot as plt
import os

# 定义映射字典
race_mapping = {
    1: 'Non-Hispanic White',
    2: 'Non-Hispanic Black',
    3: 'Mexican American',
    4: 'Other Race',
    5: 'Other Hispanic'
}

# 假设这部分代码已生成 race_auc_list 和 race_name_list（原来是 "Race {race_val}"）
# 现在我们重新构造横坐标的标签，直接用 mapping 得到
race_labels = []
for race_val in sorted(test_data[group_col].unique()):
    label = race_mapping.get(race_val, f"Race {race_val}")
    race_labels.append(label)

# 如果你的 race_auc_list 中有None，先过滤（或替换成0）
clean_auc_list = [auc if auc is not None else 0 for auc in race_auc_list]

plt.figure(figsize=(8, 6))
plt.bar(race_labels, clean_auc_list, color='skyblue')
plt.ylim(0.0, 1.0)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
plt.title('Subgroup analysis by race ')
plt.ylabel('AUC')
plt.xlabel('Race Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 将图保存到桌面，并指定TIFF格式
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop_path, 'auc_by_race2.tiff')
print(f"Saving figure to: {save_path}")
plt.savefig(save_path, dpi=300)
plt.show()


# In[42]:


import pickle

# 假设 logistic_Reg 是你的逻辑回归模型
with open('D:/0/logistic_Reg.pkl', 'wb') as f:
    pickle.dump(logistic_Reg, f)


# In[43]:


import pickle

# 假设 scaler 是你的标准化器
with open('D:/0/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# In[44]:


with open('logistic_Reg.pkl', 'wb') as file:
    pickle.dump(logistic_Reg, file)

print("模型已保存为 logistic_Reg.pkl 文件")


# In[45]:


# 训练时保存特征顺序
import pickle

with open('D:/0/features.pkl', 'wb') as f:
    pickle.dump(train_scaler.columns, f)

# 预测时加载特征顺序
with open('D:/0/features.pkl', 'wb') as f:
    features = pickle.load(f)

print("模型训练时的特征顺序:", features)


# In[88]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = logistic_Reg.predict_proba(test_scaler)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[215]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[216]:


#首先利用网格搜索方法
param_grid = {'C':np.arange(1,100,20),
             'gamma':[0.001,0.01,0.1],}
# 网格搜索方式
svc = SVC()
reg = GridSearchCV(svc,
                   param_grid,
                   scoring='roc_auc',
                   cv=5,verbose = 1).fit(train_scaler, train_Y)

best_alpha = reg.best_params_
best_score = reg.best_score_


# In[217]:


best_alpha


# In[218]:


best_score


# In[219]:


from sklearn.svm import SVC
svc = SVC(C=61,gamma=0.1,probability=True)
svc.fit(train_scaler,train_Y)
predict_svm = svc.predict(test_scaler)
predict_svm1 = svc.predict(train_scaler)


# In[220]:


plot_confusion_matrix(train_Y,predict_svm1,'svm','训练集')


# In[221]:


metrict_score(train_Y,predict_svm1,train_scaler,'svm','训练集',svc.predict_proba(train_scaler)[:, 1])


# In[222]:


plot_roc(svc,svc.predict_proba(train_scaler)[:, 1],train_Y,'svm训练集')
# 保存为 PDF 格式
plt.savefig('d:/svm训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/svm训练2.tif', bbox_inches='tight', dpi=600)


# In[223]:


plot_confusion_matrix(test_Y,predict_svm,'svm','测试集')
# 保存为 PDF 格式
plt.savefig('d:/svm测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/svm测试1.tif', bbox_inches='tight', dpi=600)


# In[224]:


metrict_score(test_Y,predict_svm,test_scaler,'svm','测试集',svc.predict_proba(test_scaler)[:, 1])


# In[225]:


plot_roc(svc,svc.predict_proba(test_scaler)[:, 1],test_Y,'svm测试集')
# 保存为 PDF 格式
plt.savefig('d:/svm测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/svm测试2.tif', bbox_inches='tight', dpi=600)


# In[112]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = svc.predict_proba(test_scaler)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[138]:


from sklearn.ensemble import RandomForestClassifier


# In[139]:


from sklearn.model_selection import GridSearchCV
#首先利用网格搜索方法
param_grid = { 'max_features':np.arange(1,10,1),

             }
# 网格搜索方式
rfr = RandomForestClassifier(random_state=1,n_estimators=170,max_depth=17,max_features=1)
reg = GridSearchCV(rfr,
                   param_grid,
                   scoring='roc_auc',
                   cv=5,verbose = 1).fit(train_X, train_Y)
best_alpha = reg.best_params_
best_score = reg.best_score_


# In[140]:


best_alpha


# In[141]:


best_score


# In[142]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=1,n_estimators=170,max_depth=17,max_features=1)
rf_model.fit(train_X, train_Y)
y_pred_tree = rf_model.predict(test_X)
y_pred_tree1 = rf_model.predict(train_X)


# In[143]:


plot_confusion_matrix(train_Y,y_pred_tree1,'随机森林','训练集')
# 保存为 PDF 格式
plt.savefig('d:/森林训练1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/森林训练1.tif', bbox_inches='tight', dpi=600)


# In[144]:


metrict_score(train_Y,y_pred_tree1,train_X,'随机森林','训练集',rf_model.predict_proba(train_X)[:, 1])


# In[145]:


plot_roc(rf_model,rf_model.predict_proba(train_X)[:, 1],train_Y,'随机森林训练集')
# 保存为 PDF 格式
plt.savefig('d:/森林训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/森林训练2.tif', bbox_inches='tight', dpi=600)


# In[146]:


plot_confusion_matrix(test_Y,y_pred_tree,'随机森林','测试集')
# 保存为 PDF 格式
plt.savefig('d:/森林测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/森林测试1.tif', bbox_inches='tight', dpi=600)


# In[147]:


metrict_score(test_Y,y_pred_tree,test_X,'随机森林','测试集',rf_model.predict_proba(test_X)[:, 1])


# In[148]:


plot_roc(rf_model,rf_model.predict_proba(test_X)[:, 1],test_Y,'随机森林测试集')
# 保存为 PDF 格式
plt.savefig('d:/森林测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/森林测试2.tif', bbox_inches='tight', dpi=600)


# In[132]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = rf_model.predict_proba(test_X)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[149]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[150]:


# 定义超参数网格
param_grid = {
    'max_depth': np.arange(3, 10, 1),  # 决策树最大深度
    'min_samples_split': np.arange(2, 10, 2),  # 节点划分所需最小样本数
    'min_samples_leaf': np.arange(1, 10, 2)  # 叶子节点最少样本数
}


# In[151]:


# 定义决策树模型
dt = DecisionTreeClassifier(random_state=1)

# 使用 GridSearchCV 进行超参数搜索
reg = GridSearchCV(
    dt, param_grid, scoring='roc_auc', cv=5, verbose=1
).fit(train_X, train_Y)

# 获取最佳参数和最佳分数
best_params = reg.best_params_
best_score = reg.best_score_


# In[152]:


best_params


# In[153]:


best_score


# In[154]:


dt = DecisionTreeClassifier(random_state=1,max_depth=8,min_samples_leaf=9,min_samples_split=2)
dt.fit(train_X, train_Y)
dt_predict = dt.predict(test_X)
dt_predict1 = dt.predict(train_X)


# In[155]:


plot_confusion_matrix(train_Y,dt_predict1,'决策树','训练集')
# 保存为 PDF 格式
plt.savefig('d:/决策训练1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/决策训练1.tif', bbox_inches='tight', dpi=600)


# In[156]:


metrict_score(train_Y,dt_predict1,train_X,'决策树','训练集',dt.predict_proba(train_X)[:, 1])


# In[157]:


plot_roc(dt,dt.predict_proba(train_X)[:, 1],train_Y,'决策树训练集')
# 保存为 PDF 格式
plt.savefig('d:/决策训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/决策训练2.tif', bbox_inches='tight', dpi=600)


# In[158]:


plot_confusion_matrix(test_Y,dt_predict,'决策树','测试集')
# 保存为 PDF 格式
plt.savefig('d:/决策测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/决策测试2.tif', bbox_inches='tight', dpi=600)


# In[159]:


metrict_score(test_Y,dt_predict,test_X,'决策树','测试集',dt.predict_proba(test_X)[:, 1])


# In[160]:


plot_roc(dt,dt.predict_proba(test_X)[:, 1],test_Y,'决策树测试集')
# 保存为 PDF 格式
plt.savefig('d:/决策测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/决策测试1.tif', bbox_inches='tight', dpi=600)


# In[151]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = dt.predict_proba(test_X)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[161]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


# In[162]:


# 定义超参数网格
param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}


# In[163]:


# 定义 MLP 分类器
mlp = MLPClassifier(max_iter=1000, random_state=1)

# 使用 GridSearchCV 进行超参数搜索
reg = GridSearchCV(
    mlp, param_grid, scoring='roc_auc', cv=5, verbose=1
).fit(train_X, train_Y)

# 获取最佳参数和最佳分数
best_params = reg.best_params_
best_score = reg.best_score_


# In[164]:


best_params


# In[165]:


best_score


# In[166]:


mlp = MLPClassifier(max_iter=1000, random_state=1,activation='tanh',hidden_layer_sizes=(100,),solver= 'adam')
mlp.fit(train_X, train_Y)
mlp_predict = mlp.predict(test_X)
mlp_predict1 = mlp.predict(train_X)


# In[167]:


plot_confusion_matrix(train_Y,mlp_predict1,'mlp','训练集')
# 保存为 PDF 格式
plt.savefig('d:/mlp训练1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/mlp训练1.tif', bbox_inches='tight', dpi=600)


# In[168]:


metrict_score(train_Y,mlp_predict1,train_X,'mlp','训练集',mlp.predict_proba(train_X)[:, 1])


# In[169]:


plot_roc(mlp,mlp.predict_proba(train_X)[:, 1],train_Y,'mlp训练集')
# 保存为 PDF 格式
plt.savefig('d:/mlp训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/mlp训练2.tif', bbox_inches='tight', dpi=600)


# In[170]:


plot_confusion_matrix(test_Y,mlp_predict,'mlp','测试集')
# 保存为 PDF 格式
plt.savefig('d:/mlp测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/mlp测试2.tif', bbox_inches='tight', dpi=600)


# In[171]:


metrict_score(test_Y,mlp_predict,test_X,'mlp','测试集',mlp.predict_proba(test_X)[:, 1])


# In[172]:


plot_roc(mlp,mlp.predict_proba(test_X)[:, 1],test_Y,'mlp测试集')
# 保存为 PDF 格式
plt.savefig('d:/mlp测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/mlp测试1.tif', bbox_inches='tight', dpi=600)


# In[167]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = mlp.predict_proba(test_X)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[173]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# In[174]:


# 定义超参数网格
param_grid = {
    'max_depth': np.arange(3, 10, 2),  # 树的最大深度
    'learning_rate': np.arange(0.01, 0.5, 0.2),  # 学习率
    'n_estimators': np.arange(10, 200, 20)  # 迭代次数（弱分类器数量）
}


# In[175]:


# 定义 GBDT 模型
gbdt = GradientBoostingClassifier(random_state=1)

# 使用 GridSearchCV 进行超参数搜索
reg = GridSearchCV(
    gbdt, param_grid, scoring='roc_auc', cv=5, verbose=1
).fit(train_X, train_Y)

# 获取最佳参数和最佳分数
best_params = reg.best_params_
best_score = reg.best_score_


# In[176]:


best_params


# In[177]:


best_score


# In[178]:


gbdt = GradientBoostingClassifier(random_state=1,learning_rate=0.21,max_depth=3,n_estimators=30)
gbdt.fit(train_X, train_Y)
gbdt_predict = gbdt.predict(test_X)
gbdt_predict1 = gbdt.predict(train_X)


# In[179]:


plot_confusion_matrix(train_Y,gbdt_predict1,'gbdt','训练集')
# 保存为 PDF 格式
plt.savefig('d:/gb训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/gb训练2.tif', bbox_inches='tight', dpi=600)


# In[180]:


metrict_score(train_Y,gbdt_predict1,train_X,'gbdt','训练集',gbdt.predict_proba(train_X)[:, 1])


# In[181]:


plot_roc(gbdt,gbdt.predict_proba(train_X)[:, 1],train_Y,'gbdt训练集')
# 保存为 PDF 格式
plt.savefig('d:/gb训练1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/gb训练1.tif', bbox_inches='tight', dpi=600)


# In[182]:


plot_confusion_matrix(test_Y,gbdt_predict,'gbdt','测试集')
# 保存为 PDF 格式
plt.savefig('d:/gb测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/gb测试2.tif', bbox_inches='tight', dpi=600)


# In[183]:


metrict_score(test_Y,gbdt_predict,test_X,'gbdt','测试集',gbdt.predict_proba(test_X)[:, 1])


# In[184]:


plot_roc(gbdt,gbdt.predict_proba(test_X)[:, 1],test_Y,'gbdt测试集')
# 保存为 PDF 格式
plt.savefig('d:/gb测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/gb测试1.tif', bbox_inches='tight', dpi=600)


# In[182]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = gbdt.predict_proba(test_X)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[185]:


from lightgbm import LGBMClassifier 
from sklearn.model_selection import GridSearchCV


# In[186]:


# LightGBM 模型
lgb_estimator = LGBMClassifier(random_state=42)

# 网格搜索参数
param_grid = {
    'n_estimators': [50, 100, 200],     # 弱学习器数量
    'max_depth': [3, 5, 7],             # 树的最大深度
}


# In[187]:


import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(random_state=1,class_weight='balanced')
reg = GridSearchCV(lgb_model,
                   param_grid,
                   scoring='roc_auc',
                   cv=5,verbose = 1).fit(train_scaler,train_Y)
best_alpha = reg.best_params_
best_score = reg.best_score_


# In[188]:


best_params


# In[189]:


best_score


# In[190]:


lgb_model = LGBMClassifier(random_state=1,class_weight='balanced',max_depth=3,n_estimators=50)
lgb_model.fit(train_X, train_Y)
lgb_predict = lgb_model.predict(test_X)
lgb_predict1 = lgb_model.predict(train_X)


# In[191]:


plot_confusion_matrix(train_Y,lgb_predict1,'lightgbm','训练集')
# 保存为 PDF 格式
plt.savefig('d:/light训练1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/light训练1.tif', bbox_inches='tight', dpi=600)


# In[192]:


metrict_score(train_Y,lgb_predict1,train_scaler,'lightgbm','训练集',lgb_model.predict_proba(train_X)[:, 1])


# In[193]:


plot_roc(lgb_model,lgb_model.predict_proba(train_X)[:, 1],train_Y,'lightgbm训练集')
# 保存为 PDF 格式
plt.savefig('d:/light训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/light训练2.tif', bbox_inches='tight', dpi=600)


# In[194]:


plot_confusion_matrix(test_Y,lgb_predict,'lightgbm','测试集')
# 保存为 PDF 格式
plt.savefig('d:/light测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/light测试2.tif', bbox_inches='tight', dpi=600)


# In[195]:


metrict_score(test_Y,lgb_predict,test_X,'lightgbm','测试集',lgb_model.predict_proba(test_X)[:, 1])


# In[196]:


plot_roc(lgb_model,lgb_model.predict_proba(test_X)[:, 1],test_Y,'lightgbm测试集')
# 保存为 PDF 格式
plt.savefig('d:/light测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/light测试1.tif', bbox_inches='tight', dpi=600)


# In[203]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = lgb_model.predict_proba(test_X)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[197]:


from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    random_seed=1,  # 等同于 random_state
    depth=3,  # 控制树的深度
    iterations=50,  # 相当于 n_estimators),  # 平衡类别权重
    verbose=0  # 关闭训练过程中的输出
)

cat_model.fit(train_X, train_Y)

cat_predict = cat_model.predict(test_X)  # 测试集预测
cat_predict1 = cat_model.predict(train_X)  # 训练集预测


# In[198]:


plot_confusion_matrix(train_Y,cat_predict1,'catboost','训练集')
# 保存为 PDF 格式
plt.savefig('d:/cat训练2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/cat训练2.tif', bbox_inches='tight', dpi=600)


# In[199]:


metrict_score(train_Y,cat_predict1,train_scaler,'catboost','训练集',cat_model.predict_proba(train_X)[:, 1])


# In[200]:


plot_roc(cat_model,cat_model.predict_proba(train_X)[:, 1],train_Y,'catboost训练集')
# 保存为 PDF 格式
plt.savefig('d:/cat训练1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/cat训练1.tif', bbox_inches='tight', dpi=600)


# In[201]:


plot_confusion_matrix(test_Y,cat_predict,'catboost','测试集')
# 保存为 PDF 格式
plt.savefig('d:/cat测试1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/cat测试1.tif', bbox_inches='tight', dpi=600)


# In[202]:


metrict_score(test_Y,cat_predict,test_X,'catboost','测试集',cat_model.predict_proba(test_X)[:, 1])


# In[203]:


plot_roc(cat_model,cat_model.predict_proba(test_X)[:, 1],test_Y,'catboost测试集')
# 保存为 PDF 格式
plt.savefig('d:/cat测试2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/cat测试2.tif', bbox_inches='tight', dpi=600)


# In[214]:


from sklearn.metrics import roc_auc_score
import numpy as np

# 获取测试集的预测概率
probs = cat_model.predict_proba(test_X)[:, 1]

# 设置Bootstrap参数
n_bootstraps = 1000  # 建议至少1000次
auc_scores = []

# 进行Bootstrap抽样
for _ in range(n_bootstraps):
    # 生成随机索引（允许重复）
    indices = np.random.choice(len(test_Y), size=len(test_Y), replace=True)
    y_true_bootstrap = test_Y.iloc[indices]  # 假设test_Y是Series或数组
    y_probs_bootstrap = probs[indices]
    
    # 确保子集中存在正负样本
    if len(np.unique(y_true_bootstrap)) == 2:
        auc = roc_auc_score(y_true_bootstrap, y_probs_bootstrap)
        auc_scores.append(auc)

# 计算置信区间
lower_bound = np.percentile(auc_scores, 2.5)
upper_bound = np.percentile(auc_scores, 97.5)

print(f"测试集AUC: {roc_auc_score(test_Y, probs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")


# In[204]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr_rf1,tpr_rf1,thresholds=roc_curve(train_Y.tolist(),
                                       list(logistic_Reg.predict_proba(train_scaler)[:, 1]))
roc_auc_rf1=auc(fpr_rf1,tpr_rf1)


fpr_rf2,tpr_rf2,thresholds=roc_curve(train_Y.tolist(),
                                       list(svc.predict_proba(train_scaler)[:, 1]))
roc_auc_rf2=auc(fpr_rf2,tpr_rf2)


fpr_rf3,tpr_rf3,thresholds=roc_curve(train_Y.tolist(),
                                       list(rf_model.predict_proba(train_X)[:, 1]))
roc_auc_rf3=auc(fpr_rf3,tpr_rf3)
fpr_rf4,tpr_rf4,thresholds=roc_curve(train_Y.tolist(),
                                       list(xgb.predict_proba(train_X)[:, 1]))
roc_auc_rf4=auc(fpr_rf4,tpr_rf4)



fpr_rf5,tpr_rf5,thresholds=roc_curve(train_Y.tolist(),
                                       list(lgb_model.predict_proba(train_X)[:, 1]))
roc_auc_rf5=auc(fpr_rf5,tpr_rf5)

fpr_rf6,tpr_rf6,thresholds=roc_curve(train_Y.tolist(),
                                       list(gbdt.predict_proba(train_X)[:, 1]))
roc_auc_rf6=auc(fpr_rf6,tpr_rf6)


fpr_rf7,tpr_rf7,thresholds=roc_curve(train_Y.tolist(),
                                       list(mlp.predict_proba(train_X)[:, 1]))
roc_auc_rf7=auc(fpr_rf7,tpr_rf7)


fpr_rf8,tpr_rf8,thresholds=roc_curve(train_Y.tolist(),
                                       list(dt.predict_proba(train_X)[:, 1]))
roc_auc_rf8=auc(fpr_rf8,tpr_rf8)

fpr_rf9,tpr_rf9,thresholds=roc_curve(train_Y.tolist(),
                                       list(cat_model.predict_proba(train_X)[:, 1]))
roc_auc_rf9=auc(fpr_rf9,tpr_rf9)

fig, axs = plt.subplots(figsize=(8,6))

colors = ['darkorange', 'cyan', '#7FFFD4', '#A52A2A', '#FF4500', '#32CD32', '#1E90FF', '#FFD700','b']

# 绘制不同模型的 ROC 曲线
plt.plot(fpr_rf1, tpr_rf1, label='LR AUC = %0.3f' % roc_auc_rf1, c=colors[0])
plt.plot(fpr_rf2, tpr_rf2, label='SVM AUC = %0.3f' % roc_auc_rf2, c=colors[1])
plt.plot(fpr_rf3, tpr_rf3, label='RF AUC = %0.3f' % roc_auc_rf3, c=colors[2])
plt.plot(fpr_rf4, tpr_rf4, label='XGB AUC = %0.3f' % roc_auc_rf4, c=colors[3])
plt.plot(fpr_rf5, tpr_rf5, label='LGBM AUC = %0.3f' % roc_auc_rf5, c=colors[4])
plt.plot(fpr_rf6, tpr_rf6, label='GBDT AUC = %0.3f' % roc_auc_rf6, c=colors[5])
plt.plot(fpr_rf7, tpr_rf7, label='MLP AUC = %0.3f' % roc_auc_rf7, c=colors[6])
plt.plot(fpr_rf8, tpr_rf8, label='DT AUC = %0.3f' % roc_auc_rf8, c=colors[7])
plt.plot(fpr_rf9, tpr_rf9, label='Catboost AUC = %0.3f' % roc_auc_rf9, c=colors[8])
plt.legend(loc='lower right',fontsize = 12)
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate',fontsize = 14)
plt.xlabel('Flase Positive Rate',fontsize = 14)
# 保存为 PDF 格式
plt.savefig('d:/roc训练.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/roc训练.tif', bbox_inches='tight', dpi=600)
plt.show()


# In[209]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr_rf1,tpr_rf1,thresholds=roc_curve(test_Y.tolist(),
                                       list(logistic_Reg.predict_proba(test_scaler)[:, 1]))
roc_auc_rf1=auc(fpr_rf1,tpr_rf1)


fpr_rf2,tpr_rf2,thresholds=roc_curve(test_Y.tolist(),
                                       list(svc.predict_proba(test_scaler)[:, 1]))
roc_auc_rf2=auc(fpr_rf2,tpr_rf2)


fpr_rf3,tpr_rf3,thresholds=roc_curve(test_Y.tolist(),
                                       list(rf_model.predict_proba(test_X)[:, 1]))
roc_auc_rf3=auc(fpr_rf3,tpr_rf3)
fpr_rf4,tpr_rf4,thresholds=roc_curve(test_Y.tolist(),
                                       list(xgb.predict_proba(test_X)[:, 1]))
roc_auc_rf4=auc(fpr_rf4,tpr_rf4)



fpr_rf5,tpr_rf5,thresholds=roc_curve(test_Y.tolist(),
                                       list(lgb_model.predict_proba(test_X)[:, 1]))
roc_auc_rf5=auc(fpr_rf5,tpr_rf5)

fpr_rf6,tpr_rf6,thresholds=roc_curve(test_Y.tolist(),
                                       list(gbdt.predict_proba(test_X)[:, 1]))
roc_auc_rf6=auc(fpr_rf6,tpr_rf6)


fpr_rf7,tpr_rf7,thresholds=roc_curve(test_Y.tolist(),
                                       list(mlp.predict_proba(test_X)[:, 1]))
roc_auc_rf7=auc(fpr_rf7,tpr_rf7)


fpr_rf8,tpr_rf8,thresholds=roc_curve(test_Y.tolist(),
                                       list(dt.predict_proba(test_X)[:, 1]))
roc_auc_rf8=auc(fpr_rf8,tpr_rf8)

fpr_rf9,tpr_rf9,thresholds=roc_curve(test_Y.tolist(),
                                       list(cat_model.predict_proba(test_X)[:, 1]))
roc_auc_rf9=auc(fpr_rf9,tpr_rf9)


fig, axs = plt.subplots(figsize=(8,6))

colors = ['darkorange', 'cyan', '#7FFFD4', '#A52A2A', '#FF4500', '#32CD32', '#1E90FF', '#FFD700','b']

# 绘制不同模型的 ROC 曲线
plt.plot(fpr_rf1, tpr_rf1, label='LR AUC = %0.3f' % roc_auc_rf1, c=colors[0])
plt.plot(fpr_rf2, tpr_rf2, label='SVM AUC = %0.3f' % roc_auc_rf2, c=colors[1])
plt.plot(fpr_rf3, tpr_rf3, label='RF AUC = %0.3f' % roc_auc_rf3, c=colors[2])
plt.plot(fpr_rf4, tpr_rf4, label='XGB AUC = %0.3f' % roc_auc_rf4, c=colors[3])
plt.plot(fpr_rf5, tpr_rf5, label='LGBM AUC = %0.3f' % roc_auc_rf5, c=colors[4])
plt.plot(fpr_rf6, tpr_rf6, label='GBDT AUC = %0.3f' % roc_auc_rf6, c=colors[5])
plt.plot(fpr_rf7, tpr_rf7, label='MLP AUC = %0.3f' % roc_auc_rf7, c=colors[6])
plt.plot(fpr_rf8, tpr_rf8, label='DT AUC = %0.3f' % roc_auc_rf8, c=colors[7])
plt.plot(fpr_rf9, tpr_rf9, label='Catboost AUC = %0.3f' % roc_auc_rf9, c=colors[8])
plt.legend(loc='lower right',fontsize = 12)
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate',fontsize = 14)
plt.xlabel('Flase Positive Rate',fontsize = 14)
# 保存为 PDF 格式
plt.savefig('d:/roc测试.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/roc测试.tif', bbox_inches='tight', dpi=600)
plt.show()


# In[232]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

# 定义参数网格
param_grid = {
    'C': np.arange(1, 100, 20),
    'gamma': [0.001, 0.01, 0.1]
}

# 一定要加 probability=True，否则不能正确输出概率
svc = SVC(probability=True)

# 网格搜索
reg = GridSearchCV(
    svc,
    param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1
).fit(train_scaler, train_Y)

# 获取最优参数和分数
best_params = reg.best_params_
best_score = reg.best_score_

print("Best SVM parameters:", best_params)
print("Best CV AUC score:", best_score)

# 使用最佳模型进行测试集预测
best_model = reg.best_estimator_
svm_proba = best_model.predict_proba(test_scaler)[:, 1]

# 测试集 AUC
from sklearn.metrics import roc_auc_score
svm_auc = roc_auc_score(test_Y, svm_proba)
print("Test AUC of best SVM model:", svm_auc)



# In[260]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import itertools

# 计算每个模型在正负样本对上的U统计量
def fastDeLong(predictions, label_1_count):
    m = label_1_count
    n = predictions.shape[1] - m
    positive_examples = predictions[:, :m]
    negative_examples = predictions[:, m:]
    k = predictions.shape[0]

    tx = np.zeros((k, m))
    ty = np.zeros((k, n))

    for r in range(k):
        for i in range(m):
            tx[r, i] = np.sum(positive_examples[r, i] > negative_examples[r, :]) + 0.5 * np.sum(positive_examples[r, i] == negative_examples[r, :])
        tx[r, :] /= n

        for j in range(n):
            ty[r, j] = np.sum(positive_examples[r, :] > negative_examples[r, j]) + 0.5 * np.sum(positive_examples[r, :] == negative_examples[r, j])
        ty[r, :] /= m

    return tx, ty

# 计算AUC和AUC的协方差矩阵
def calc_auc_covariance(tx, ty):
    k = tx.shape[0]
    m = tx.shape[1]
    n = ty.shape[1]

    aucs = np.mean(tx, axis=1)

    v01 = np.var(tx, axis=1, ddof=1) / m
    v10 = np.var(ty, axis=1, ddof=1) / n

    covariance = v01 + v10  # 每个模型AUC方差

    # 计算模型间协方差矩阵
    covar_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            covar_01 = np.cov(tx[i, :], tx[j, :], ddof=1)[0, 1] / m
            covar_10 = np.cov(ty[i, :], ty[j, :], ddof=1)[0, 1] / n
            covar_matrix[i, j] = covar_01 + covar_10

    return aucs, covariance, covar_matrix

def delong_roc_test(y_true, y_pred1, y_pred2, alpha=0.95):
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
        raise ValueError("真实标签和预测概率长度不一致")

    pos_label_count = np.sum(y_true == 1)
    if pos_label_count == 0 or pos_label_count == len(y_true):
        raise ValueError("测试集必须同时包含正负样本")

    # 按标签分组，将正例排前，负例排后，确保计算正确
    order = np.argsort(y_true)[::-1]  # 把正例排在前面
    y_true_sorted = y_true[order]
    preds = np.vstack([y_pred1[order], y_pred2[order]])

    # 计算U统计量
    tx, ty = fastDeLong(preds, pos_label_count)
    aucs, vars_, covar_matrix = calc_auc_covariance(tx, ty)

    auc_diff = aucs[0] - aucs[1]
    var_auc_diff = covar_matrix[0, 0] + covar_matrix[1, 1] - 2 * covar_matrix[0, 1]

    se = np.sqrt(var_auc_diff)
    z = auc_diff / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))

    z_alpha = norm.ppf(1 - (1 - alpha) / 2)
    ci_lower = auc_diff - z_alpha * se
    ci_upper = auc_diff + z_alpha * se

    return p_value, auc_diff, aucs[0], aucs[1], z, ci_lower, ci_upper, vars_[0], vars_[1]

# ----------------- 主流程 -----------------

# 你的模型预测概率，确保都是长度相同
model_preds = {
    'LR': logistic_Reg.predict_proba(test_X)[:, 1],
    'SVM': svm_pred_proba,
    'RF': rf_model.predict_proba(test_X)[:, 1],
    'XGB': xgb.predict_proba(test_X)[:, 1],
    'LGBM': lgb_model.predict_proba(test_X)[:, 1],
    'GBDT': gbdt.predict_proba(test_X)[:, 1],
    'MLP': mlp.predict_proba(test_X)[:, 1],
    'DT': dt.predict_proba(test_X)[:, 1],
    'CatBoost': cat_model.predict_proba(test_X)[:, 1],
}

y_true = test_Y.values if hasattr(test_Y, 'values') else np.array(test_Y)

# 检查长度一致
for k, v in model_preds.items():
    if len(v) != len(y_true):
        raise ValueError(f"模型{k}预测概率长度与真实标签长度不一致")

results = []
for m1, m2 in itertools.combinations(model_preds.keys(), 2):
    try:
        p_val, auc_diff, auc1, auc2, z_score, ci_lower, ci_upper, var1, var2 = delong_roc_test(
            y_true, model_preds[m1], model_preds[m2]
        )
        results.append({
            'Model 1': m1,
            'Model 2': m2,
            'AUC 1': auc1,
            'AUC 2': auc2,
            'AUC Diff': auc_diff,
            'Z-score': z_score,
            'p-value': p_val,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'Var 1': var1,
            'Var 2': var2,
            'Significant (p<0.05)': p_val < 0.05
        })
    except Exception as e:
        print(f"Error comparing {m1} vs {m2}: {e}")

df_results = pd.DataFrame(results).sort_values(by='p-value').reset_index(drop=True)
print(df_results.round(3))

    


# In[257]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# 1. 定义 SVM，开启概率估计
svc = SVC(probability=True, random_state=42)

# 2. 定义参数网格
param_grid = {
    'C': np.arange(1, 100, 20),
    'gamma': [0.001, 0.01, 0.1]
}

# 3. 网格搜索，评估指标用ROC AUC，5折交叉验证
reg = GridSearchCV(svc, param_grid, scoring='roc_auc', cv=5, verbose=1)
reg.fit(train_scaler, train_Y)

print("SVM 最佳参数:", reg.best_params_)
print("SVM 最佳训练AUC:", reg.best_score_)

# 4. 用调优后的最佳模型预测测试集概率
svm_pred_proba = reg.best_estimator_.predict_proba(test_X)[:, 1]

# 5. 其他模型预测概率（假设这些模型已经训练完毕）
lr_pred_proba = logistic_Reg.predict_proba(test_X)[:, 1]
rf_pred_proba = rf_model.predict_proba(test_X)[:, 1]
xgb_pred_proba = xgb.predict_proba(test_X)[:, 1]
lgbm_pred_proba = lgb_model.predict_proba(test_X)[:, 1]
gbdt_pred_proba = gbdt.predict_proba(test_X)[:, 1]
mlp_pred_proba = mlp.predict_proba(test_X)[:, 1]
dt_pred_proba = dt.predict_proba(test_X)[:, 1]
catboost_pred_proba = cat_model.predict_proba(test_X)[:, 1]

# 6. 把预测概率放入字典，用于后续Delong检验
model_preds = {
    'LR': lr_pred_proba,
    'SVM': svm_pred_proba,
    'RF': rf_pred_proba,
    'XGB': xgb_pred_proba,
    'LGBM': lgbm_pred_proba,
    'GBDT': gbdt_pred_proba,
    'MLP': mlp_pred_proba,
    'DT': dt_pred_proba,
    'CatBoost': catboost_pred_proba,
}

# 7. 测试集真实标签
y_true = test_Y.values if hasattr(test_Y, 'values') else test_Y

# 8. 之前你有的 delong_roc_test 函数，接下来用它做模型两两对比（示例略，保持你原代码）

# --- 后面用delong_roc_test比较即可 ---


# In[258]:


# 1. 检查测试集标签是否有正负样本
print('测试集标签分布:')
print(pd.Series(test_Y).value_counts())

# 2. 确保测试集做了和训练集一样的预处理
# 例如如果训练时用了scaler = StandardScaler().fit(train_X)
# 测试时用同一个scaler.transform(test_X)

test_X_scaled = scaler.transform(test_X)  # 如果你训练时用了scaler

# 3. 用调参后最优模型预测，计算测试集AUC
svm_best_model = reg.best_estimator_
svm_pred_proba = svm_best_model.predict_proba(test_X_scaled)[:, 1]

from sklearn.metrics import roc_auc_score
auc_test = roc_auc_score(test_Y, svm_pred_proba)
print(f'SVM 测试集 AUC: {auc_test:.4f}')


# In[238]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 步骤1：定义带 probability 的 SVM
svc = SVC(probability=True)

# 步骤2：参数网格
param_grid = {
    'C': np.arange(1, 100, 20),
    'gamma': [0.001, 0.01, 0.1],
}

# 步骤3：用 GridSearchCV 搜索并拟合
svc_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1
)
svc_search.fit(train_scaler, train_Y)

# 步骤4：获取已经训练好的模型
svc_best = svc_search.best_estimator_

# 步骤5：使用 svc_best 预测概率
y_pred_svm_test = svc_best.predict_proba(test_scaler)[:, 1]


# In[241]:


model_preds = {
    'LR': logistic_Reg.predict_proba(test_X)[:, 1],
    'SVM': reg.best_estimator_.predict_proba(test_X)[:, 1],

    'RF': rf_model.predict_proba(test_X)[:, 1],
    'XGB': xgb.predict_proba(test_X)[:, 1],
    'LGBM': lgb_model.predict_proba(test_X)[:, 1],
    'GBDT': gbdt.predict_proba(test_X)[:, 1],
    'MLP': mlp.predict_proba(test_X)[:, 1],
    'DT': dt.predict_proba(test_X)[:, 1],
    'CatBoost': cat_model.predict_proba(test_X)[:, 1],
}


# In[231]:


from sklearn.metrics import roc_auc_score

print("LR AUC:", roc_auc_score(test_Y, model_preds['LR']))
print("SVM AUC:", roc_auc_score(test_Y, model_preds['SVM']))
print("RF AUC:", roc_auc_score(test_Y, model_preds['RF']))
# ...类似其他模型


# In[213]:


print("训练样本数:", train_X.shape[0])
print("测试样本数:", test_X.shape[0])

# 模型训练
logistic_Reg.fit(train_X, train_Y)

# 预测概率
train_probs = logistic_Reg.predict_proba(train_X)[:, 1]
test_probs = logistic_Reg.predict_proba(test_X)[:, 1]

# 计算AUC
from sklearn.metrics import roc_auc_score
print("训练集LR AUC:", roc_auc_score(train_Y, train_probs))
print("测试集LR AUC:", roc_auc_score(test_Y, test_probs))


# In[205]:


test_X = test_X.iloc[:,:5]


# In[206]:


test_scaler = pd.DataFrame(test_scaler,columns=test_X.columns)


# In[207]:


train_scaler = pd.DataFrame(train_scaler,columns=test_X.columns)


# In[208]:


train_scaler.rename(columns={'Gender': 'Sex','LEG':'Thigh length'}, inplace=True)
test_scaler.rename(columns={'Gender': 'Sex','LEG':'Thigh length'}, inplace=True)


# In[223]:


train_scaler


# In[224]:


import shap
explainer = shap.Explainer(logistic_Reg.predict, train_scaler)
shap_values = explainer(test_scaler)


# In[225]:


plt.clf()
shap.plots.bar(shap_values, max_display=9,show=False)
plt.savefig('d:/0/重要.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/0/重要.tif', bbox_inches='tight', dpi=600)


# In[226]:


plt.clf()
shap.plots.beeswarm(shap_values, max_display=9,show=False)
plt.savefig('d:/0/shap.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/0/shap.tif', bbox_inches='tight', dpi=600)


# In[227]:


plt.clf()
shap.plots.scatter(shap_values[:, "Height"],show=False)
plt.savefig('d:/0/H.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/0/H.tif', bbox_inches='tight', dpi=600)


# In[228]:


logistic_Reg.predict(test_scaler)


# In[229]:


shap.force_plot(shap_values[0,:], test_scaler.iloc[0,:], show=False, matplotlib=True)
plt.savefig('d:/0/单1.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/0/单1.tif', bbox_inches='tight', dpi=600)


# In[230]:


shap.force_plot(shap_values[1,:], test_X.iloc[1,:], show=False,matplotlib=True)
plt.savefig('d:/0/单2.pdf', bbox_inches='tight')
# 保存为 TIFF 格式
plt.savefig('d:/0/单2.tif', bbox_inches='tight', dpi=600)

