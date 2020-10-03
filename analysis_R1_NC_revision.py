import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(0)

df = pd.read_excel('cor.mes.merged_2020.xlsx') 


immu_df = pd.DataFrame(df)

immu_df.drop(labels=['cohort', 'ID', 'r.nr'], axis=1, inplace=True)
immu_df = StandardScaler().fit_transform(immu_df)



# BPresponse predictability in c and d separately
import statsmodels.api as sm

# https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm
def fwd_stepwise_selection(X, y, initial_list=[], verbose=True, top_n=None):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    """
    included = list(initial_list)
    while len(included) < X.shape[1]:
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        new_AIC = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
        #     model = sm.Logit(
        #             y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(disp=0)

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


cur_group = 'c'
# cur_group = 'm'
group_inds = df['cohort'].values == cur_group

y = np.array(df['r.nr'].values[group_inds] == 'R', dtype=np.int)
y[y == 0] = -1
X = immu_df[group_inds]
immu_cols = df.columns[3:]
X[np.isnan(X)] = 0

sel_w_pvals = fwd_stepwise_selection(
        pd.DataFrame(X, columns=immu_cols), y, verbose=True, top_n=5)
# print('Forward-stepwise selection: ' +  ' -> '.join(sel_w_pvals))
sel_vars_inds = np.isin(immu_cols, sel_w_pvals)

model = sm.OLS(y, X[:, sel_vars_inds]).fit()
model.summary()

from sklearn.linear_model.logistic import LogisticRegression
clf = LogisticRegression(fit_intercept=False)

clf.fit(X[:, sel_vars_inds:], y)



# cur_group = 'c'
cur_group = 'm'
group_inds = df['cohort'].values == cur_group

y = np.array(df['r.nr'].values[group_inds] == 'R', dtype=np.int)
y[y == 0] = -1
X = immu_df[group_inds]
immu_cols = df.columns[3:]
X[np.isnan(X)] = 0


print(clf.score(X[:, sel_vars_inds], y)) 

print(np.mean(np.array(clf.predict_proba(X[:, sel_vars_inds])[:, 1] > 0.50, dtype=np.int) == y))


STOP


top5_cols = pd.Series(all_sel_cols).head(5).values
top5_inds = np.isin(immu_cols, top5_cols)
clf.fit(X[:, top5_inds], y)

import seaborn as sns
from matplotlib import pylab as plt
my_title = 'Single-subject prediction: %2.2f%%' % (np.mean(accs) * 100)
disp_coefs = np.squeeze(clf.coef_)
TH = 1.50
color_order = np.array(["#e74c3c"] * len(disp_coefs))
#iord = np.argsort(disp_coefs)
color_order[disp_coefs < 0] = "#3498db"
my_palette = sns.color_palette(color_order)
plt.figure()
bar_hdl = sns.barplot(np.array(top5_cols), disp_coefs, #ax=axes[i_topic],
                      palette=my_palette, n_boot=100, ci=1.0)
for item in bar_hdl.get_xticklabels():
    item.set_rotation(90)
ticks = plt.xticks()[0]  
sns.despine(bottom=True)
plt.xticks(ticks, np.array(top5_cols))
plt.grid(True, alpha=0.25)
plt.ylabel('Contribution to response prediction')  # (+/- bootstrapped uncertainty')

plt.title(my_title)
plt.savefig('classif_top5_barplot_.pdf', dpi=600)
plt.savefig('classif_top5_barplot_.png', dpi=600)
#plt.show()



