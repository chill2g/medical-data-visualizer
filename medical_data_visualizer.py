import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("/content/medical-data-visualizer/medical_examination.csv")
print(df.head())
#print(df.shape)

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)


# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
      df,
      id_vars=['cardio'],
      value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight' ]
    )

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat = df_cat.rename(columns={'size': 'total'})
    #print(df_cat)
    

    # 7
    #sns.catplot(data=df_cat, kind='bar', x='variable', y='total', hue='value', col='cardio')


    # 8
    fig = sns.catplot(data=df_cat, kind='bar', x='variable', y='total', hue='value', col='cardio').fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
      (df['ap_lo'] <= df['ap_hi']) &
      (df['height'] >= df['height'].quantile(0.025)) &
      (df['height'] <= df['height'].quantile(0.975)) &
      (df['weight'] >= df['weight'].quantile(0.025)) &
      (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr().round(1)
    #print(corr)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle of matrix mask


    # 14
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15
    sns.heatmap(
      corr,
      mask=mask,
      annot=True, # for nums
      fmt='.1f',  # 1dp for 0s
      cmap="coolwarm", #red blue colour scheme
      center=0, #center colourmap at 0
      square=True,
      vmin=-0.1, vmax=0.3,
      cbar_kws={'shrink': 0.8},
      ax=ax
    )


    # 16
    fig.savefig('heatmap.png')
    return fig
