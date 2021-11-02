# Build a Random Forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create a DataFrame of Feature/Gini values
features = pd.concat([pd.Series(X_train.columns, name='Features'), 		         	
                	  pd.Series(model.feature_importances_, 
                		        name='Gini')], axis=1 )
features.sort_values(by='Gini', ascending=True, inplace=True)

# Create feature importance plot
import matplotlib.pyplot as plt
plt.figure(figsize=(5,6))
plt.barh(features.Features[:20], features.Gini[:20], color='#F8766D', edgecolor='black', alpha=0.8)
plt.xlabel('Gini', fontsize=14, fontweight='bold', labelpad=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.margins(0.02)
plt.tight_layout()
plt.savefig('Figure_Barplot_feature_importance.pdf')
