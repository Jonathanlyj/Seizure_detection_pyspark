def evaluateClassifer(result):
    result_label = result.select(['prediction', 'labels']).rdd
    metrics = MulticlassMetrics(result_label)
    precision = metrics.weightedPrecision
    recall = metrics.weightedRecall
    f1Score = metrics.weightedFMeasure()
    accu = metrics.accuracy
    print('Accuracy: ' + str(accu))
    print('Average F1 score: ' + str(precision))
    print('Average Precision: ' + str(precision))
    print('Average Recall: ' + str(recall))
    return metrics

def score_classifier_auc(seizure_prob, early_prob, labels):
    S_predictions = seizure_prob
    E_predictions = early_prob
    S_y_cv = [1.0 if (x == 1.0 or x == 2.0) else 0.0 for x in labels]
    E_y_cv = [1.0 if x == 2.0 else 0.0 for x in labels]


    fpr, tpr, thresholds = roc_curve(S_y_cv, S_predictions)
    S_roc_auc = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(E_y_cv, E_predictions)
    E_roc_auc = auc(fpr, tpr)

    return S_roc_auc, E_roc_auc

def customEvaluate(result):
    def resultRddToDf(x):
        '''Convert rdd to  and pass this function in Row() args'''
        d = {}
        d['seizure_prob'] = float(x[0][1] + x[0][2])
        d['early_prob'] = float(x[0][2])
        return d
    result_prob_df = result.select('probability').rdd.map(lambda x: Row(**resultRddToDf(x))).toDF()
    result_prob_df.cache()
    seizure_probs = [float(row.seizure_prob) for row in result_prob_df.select('seizure_prob').collect()]
    early_probs = [float(row.early_prob) for row in result_prob_df.select('early_prob').collect()]
    labels = [float(row.labels) for row in result.select('labels').collect()]
    

    S_roc_auc, E_roc_auc = score_classifier_auc(seizure_probs, early_probs, labels)  
    return (S_roc_auc + E_roc_auc) / 2





    
    
    

    #print('time consumed: ' + str(end_ts - start_ts))
    
    