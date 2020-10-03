

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(0)

df = pd.read_excel('data_downsampled_03122018.xlsx') 

df = df[df.Case == 1]
df = df[df.responseBP != 'ND' ]

immu_df = df.loc[:, '7_Foxp3NEG Ki67POZ _perc_ of CD3POZ':'5_Tcm/2_Treg']
immu_df = immu_df.dropna(thresh=immu_df.shape[-1] - 10)

print(immu_df.apply(np.isnan).sum(0).value_counts())

# remove any column with more than 1 missing value
immu_df = immu_df.loc[:, immu_df.apply(np.isnan).sum(0) <= 1]

print('After NaN removal we have:')
immu_df.shape

print('Number of string-formated elements:')
# immu_df.apply(lambda x:isinstance(x, basestring)).sum()

df = df[df.index.isin(immu_df.index)]
immu_cols = immu_df.columns
immu_df = StandardScaler().fit_transform(immu_df)




# BPresponse predictability in c and d separately
import statsmodels.api as sm

def fwd_stepwise_selection(X, y, initial_list=[], verbose=True, top_n=None):
    included = list(initial_list)
    while len(included) < X.shape[1]:
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        new_AIC = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]
            new_AIC[new_column] = model.aic
        best_pval = new_pval.min()
        best_feature = new_pval.argmin()
        best_ai = new_AIC[best_feature]
        included.append(best_feature)
        if verbose:
            print('Add  {:30} with p-value {:.6}'.format(
                best_feature, best_pval))
        #     print('Add  {:30} with p-value {:.6} AIC {:.6}'.format(
        #         best_feature, best_pval, best_ai))
        if top_n is not None and top_n == len(included):
            break
    return included


cur_group = 'd'
cur_group = 'c'
group_inds = df['corona/dash'].values == cur_group

y = np.array(df.responseBP.values[group_inds] == 'R', dtype=np.int)
y[y == 0] = -1
X = immu_df[group_inds]

sel_w_pvals = fwd_stepwise_selection(
        pd.DataFrame(X, columns=immu_cols), y, verbose=True, top_n=10)
# print('Forward-stepwise selection: ' +  ' -> '.join(sel_w_pvals))
sel_vars_inds = np.isin(immu_cols, sel_w_pvals)

model = sm.OLS(y, X[:, sel_vars_inds]).fit()
model.summary()


from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import KFold

clf = LogisticRegression(fit_intercept=False)

folder = KFold(n_splits=len(X))  # LOO
accs, coefs = [], []
y_outcome = df.responseBP  # df.responseBP # df.responseLDL
n_fold = 1
all_sel_cols = []
sel_frequencies = []

all_real_y = []
all_pred_y = []
for train_inds, test_inds in folder.split(X):
    print(n_fold)
    n_fold += 1
    sel_w_pvals = fwd_stepwise_selection(
        pd.DataFrame(X[train_inds, :], columns=immu_cols),
            y[train_inds], verbose=True, top_n=10)  # ~1/3 of available samples
    print('Forward-stepwise selection: ' +  ' -> '.join(sel_w_pvals))
    sel_vars_inds = np.isin(immu_cols, sel_w_pvals)

    clf.fit(X[train_inds, :][:, sel_vars_inds], y[train_inds])

    sel_frequencies.append(sel_vars_inds)
    coefs.append(clf.coef_)
    accs.append(clf.score(X[test_inds, :][:, sel_vars_inds], y[test_inds]))
    all_pred_y += list(clf.predict(X[test_inds, :][:, sel_vars_inds]))
    all_real_y += list(y[test_inds])
    all_sel_cols += sel_w_pvals
print(np.mean(accs))
# print(accs)

from sklearn.metrics import f1_score
print(f1_score(y_true=all_real_y, y_pred=all_pred_y))

#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_true=all_real_y, y_pred=all_pred_y)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)


# refit with the top 10 only for visu of clf coefs
top10_cols = pd.Series(all_sel_cols).head(10).values
top10_inds = np.isin(immu_cols, top10_cols)
clf.fit(X[:, top10_inds], y)

import seaborn as sns
from matplotlib import pylab as plt
my_title = 'Single-subject prediction: %2.2f%%' % (np.mean(accs) * 100)
disp_coefs = np.squeeze(clf.coef_)
TH = 1.50
color_order = np.array(["#e74c3c"] * len(disp_coefs))

color_order[disp_coefs < 0] = "#3498db"
my_palette = sns.color_palette(color_order)
plt.figure()
bar_hdl = sns.barplot(np.array(top10_cols), disp_coefs, #ax=axes[i_topic],
                      palette=my_palette, n_boot=100, ci=1.0)
for item in bar_hdl.get_xticklabels():
    item.set_rotation(90)
ticks = plt.xticks()[0]  
sns.despine(bottom=True)
plt.xticks(ticks, np.array(top10_cols))
plt.grid(True, alpha=0.25)
plt.ylabel('Contribution to response prediction')  # (+/- bootstrapped uncertainty')
plt.xlabel('Immunom target')
plt.ylim(-TH, TH)

plt.title(my_title)
plt.savefig('classif_immuno_top10_barplot_.pdf', dpi=600)
plt.savefig('classif_immuno_top10_barplot.png', dpi=600)


top10_cols = pd.Series(all_sel_cols).head(10).values
top10_inds = np.isin(immu_cols, top10_cols)
clf.fit(X[:, top10_inds], y)


andras_ticks = [
    u'Th1/Treg (CXCR3+ CD25-/CD25+)',

    u'Teff % of CD8',

    u'Naïve FoxP3+ % of Treg',

    u'Naïve FoxP3- Treg % of CD4+',

    u'CD31+ CD39- % of Treg',

    u'Memory CD24+ % of CD8+',

    u'Th17-like Treg /µl',

    u'Naïve CD31+ Treg % of CD4+',

    u'Memory FoxP3+ Treg /µl',

    u'IL-17A+ TNFa+ MAIT /µl']

import seaborn as sns
from matplotlib import pylab as plt
my_title = ''
disp_coefs = np.array(np.sum(sel_frequencies, axis=0)[top10_inds], dtype=np.float)
disp_coefs = (disp_coefs / max(disp_coefs)) * 100


# plot the selection frequ

#color_order = np.array(["#e74c3c"] * len(disp_coefs))
#iord = np.argsort(disp_coefs)
color_order[disp_coefs < 0] = "#3498db"
my_palette = sns.color_palette(color_order)
plt.figure()
bar_hdl = sns.barplot(np.array(top10_cols), disp_coefs, #ax=axes[i_topic],
                      color='black', #palette=my_palette,
                      n_boot=100, ci=1.0)
for item in bar_hdl.get_xticklabels():
    item.set_rotation(90)
ticks = plt.xticks()[0]  
sns.despine(bottom=True)
plt.xticks(ticks, np.array(andras_ticks))
# plt.xticks(ticks, np.array(top10_cols))
i_yticks = np.arange(0, 101, 20)
str_yticks = ['%i%%' % perc for perc in i_yticks]
plt.yticks(i_yticks, str_yticks)
plt.grid(True, alpha=0.25)
plt.ylabel('Selection frequency')  # (+/- bootstrapped uncertainty')
plt.xlabel('Immunom target')
plt.tight_layout()
# plt.ylim(-TH, TH)
plt.savefig('classif_immuno_top10_barplot_selfrequ.pdf', dpi=600)
plt.savefig('classif_immuno_top10_barplot_selfrequ.png', dpi=600)


STOP



