"""
Functions to train and evaulate a model on different numbers of training samples
and plot scores for n number of trials
"""

class ModelRepeater(object):
    
    def __init__(self, numbers, model):
        """
        Args:
            numbers: list of numbers
            model: classification model
        """
        self.n = numbers
        self.m = model
    
    def use_n_samples(self, df, score_type, text_col, label_col):
        """
        Try model on different numbers of training samples and get scores
        Args:
            df: Dataframe
            score_type: Metric type- can be 'f1', 'recall', 'precision', 'accuracy'
            text_col: name of column with text (string)
            label_col: name of column with labels (string)
        Returns:
            scores for different numbers of training samples
        """
        scores=[] 
        
        for num in self.n:
            # randomly sample num from data
            sampled_df = df.sample(n=num)
            
            # get features and labels
            X = sampled_df.loc[:, text_col].values
            labels = sampled_df.loc[:, label_col].values

            # Encode labels
            le = LabelEncoder()
            y = le.fit_transform(labels)
            
            # split into train and test sets
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3)
            
            # fit to training data
            self.m.fit(X_train, y_train)
            
            # predict on test set
            y_pred = self.m.predict(X_test)
            
            # get scores
            if score_type == 'f1':
                scores.append(f1_score(y_test, y_pred, average='weighted'))
            elif score_type == 'recall':
                scores.append(recall_score(y_test, y_pred, average='weighted'))
            elif score_type == 'precision':
                scores.append(precision_score(y_test, y_pred, average='weighted'))
            elif score_type == 'accuracy':
                scores.append(accuracy_score(y_test, y_pred))
                
        return scores

    def multiple_plot(self, df, n, score_type, text_col, label_col):
        """
        Plots scores for n trials
        """
        plot_data = []
        n_trials = n 
        
        for trial in range(n_trials):
            scores = self.use_n_samples(df, score_type, text_col, label_col)
            plot_data.append(scores)
            
        for p in plot_data:
            plt.plot(self.n, p, color=sns.color_palette()[0])
            plt.xlabel('Number of training samples')
            plt.ylabel('Score')


