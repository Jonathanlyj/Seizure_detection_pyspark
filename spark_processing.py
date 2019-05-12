def sample_transform(x):
    
    full_transform = FreqWithTimeTransform(1, 48, 400)
    return full_transform.apply(x)

def process_raw_sample(sample, with_latency, transforms):

            
    
            data = sample['data']
            transformed_data = transforms(data)
            X = transformed_data            
            if with_latency:
                # this is ictal
                
                latency = sample['latency'][0]
                if latency <= 15:
                    y_value = float(2) # ictal <= 15
                else:
                    y_value = float(1) # ictal > 15
            else:
                # this is interictal
                y_value = float(0)

            return (X, y_value)


def set_model(numTrees, labelCol, maxDepth, seed):
    '''Initialize classifer model and tune parameters with respect to cross-validation result'''
    rf = RandomForestClassifier(numTrees = numTrees, minInstancesPerNode = 2, labelCol = labelCol, maxDepth = maxDepth, seed = seed)

    return rf



def save_model(model, gs_dir, subject):
    json_str_rdd = sc.textFile(gs_dir + '/SETTINGS.json')
    json_str = ''.join(json_str_rdd.collect())
    settings = json.loads(json_str)
    
    model_dir = settings['data-cache-dir']
    
    model.write().overwrite().save('/'.join([gs_dir, model_dir, subject + '_rf']))
    print('Trained classifier saved')

def load_model(gs_dir, subject):
    json_str_rdd = sc.textFile(gs_dir + '/SETTINGS.json')
    json_str = ''.join(json_str_rdd.collect())
    settings = json.loads(json_str)
    model_dir = settings['data-cache-dir']
    try: 
        model = RandomForestClassificationModel.load('/'.join([gs_dir, model_dir, subject + '_rf']))
        print('Trained classifier loaded')
        return model
    except:
        return None



def train_model(gs_dir, subjects, sc, fs, num_nodes):
    '''Train the model with preset params with full scope of labelled data from provided subjects list 
    and save the trained classfier to google cloud storage'''

    json_str_rdd = sc.textFile(gs_dir + '/SETTINGS.json')
    json_str = ''.join(json_str_rdd.collect())
    settings = json.loads(json_str)
    
    proj_name = settings['gcp-project-name']
    proj_dir = settings['gcp-bucket-project-dir']
    dataset_dir = settings['dataset-dir']
    fs = gcsfs.GCSFileSystem(project = proj_name)
    
    models = []
    
    for subject in subjects:
        
        #Load data into rdd
        start_time = time.time()
        loader = dataloader('/'.join([proj_dir,dataset_dir,subject]), fs)
        ictal_raw = loader.load_ictal_data()
        interictal_raw = loader.load_interictal_data()
        partitionNum = num_nodes * 10
        ictal_rdd = sc.parallelize(ictal_raw, partitionNum)
        interictal_rdd = sc.parallelize(interictal_raw, partitionNum)
        end_time = time.time()
        print('--- '+ subject + ": Data Loading %s seconds ---" % (end_time - start_time))
        #Data preprocessing and transformation
        start_time = time.time()
        transformed_ictal_rdd = ictal_rdd.map(lambda x: process_raw_sample(x, True, sample_transform)).cache()
        transformed_interictal_rdd = interictal_rdd.map(lambda x: process_raw_sample(x, False, sample_transform)).cache()

        def rddToDf(x):
            '''Convert rdd to  and pass this function in Row() args'''
            sample_X, sample_y = x
            d = {}
            d['features'] = Vectors.dense(sample_X)
            d['labels'] = sample_y
            return d

        ictal_df = transformed_ictal_rdd.map(lambda x: Row(**rddToDf(x))).toDF()
        interictal_df = transformed_interictal_rdd.map(lambda x: Row(**rddToDf(x))).toDF()
        labeled_df = ictal_df.unionAll(interictal_df)
        labeled_df.cache()
        print(labeled_df.rdd.count())
        end_time = time.time()
        print('--- '+ subject + ": Data Transformation %s seconds ---" % (end_time - start_time))

        #Train and save model
        rf = set_model(3000, 'labels', seed = 130, maxDepth = 5)
        start_time = time.time()
        model = rf.fit(labeled_df)
        end_time = time.time()
        print('--- '+ subject + ": Model Training %s seconds ---" % (end_time - start_time))
        print('--- '+ subject + ": Saving Trained Model ---" )
        save_model(model, gs_dir, subject)
        models.append(model)
        del labeled_df
        del model

    return models
        
        