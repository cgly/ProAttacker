

binary_words = set(['is', 'are', 'was', 'were', "'s", "'re", "'ll", "will",
                    "could", "would", "may", "might", "can", "must",
                    'has', 'have', 'not', 'no', 'nor', "'t", 'wont'])

nclasses = {'ag': 4, 'yelp': 2, 'yahoo': 10, 'dbpedia': 14, 'ag_3': 4,
            'sst-2': 2, 'mnli':3, 'qnli': 2,
            'qqp': 2, 'rte': 2, 'mrpc': 2,'imdb':2,}
class_names = {#1'ag': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci-Tech'},#R原始版本:
                #'ag':{0: "Military", 1: "ESPN", 2: "Commercial", 3: "IT"},
            #'ag':{0: "Religion", 1: "Sports", 2: "Market", 3: "Digital"},
            #'ag':{0: "Oriental", 1: "Sports", 2: "Investment", 3: "Computer"},
            #'ag':{0: "Religion", 1: "Sport", 2: "Market", 3: "Digital"},#best
            'ag': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'},#R:
            #'ag': {0: "MAP", 1: "Sports", 2: "Marketplace", 3: "Microsoft"},
            #'ag':{0: "CNN", 1: "ESPN", 2: "Business", 3: "Wired"},
                #'ag': {0: "Military", 1: "Sport", 2: "Commercial", 3: "IT"},
               'yelp': {0: "Negative", 1: 'Positive'},
            #'yelp': {0: "bad", 1: 'good'},
            'imdb':{0:"negative" ,1:'positive'},
               'dbpedia': {0: 'Company', 1: 'EducationInstitution', 2: "Artist",
                           3: "Athlete", 4: "OfficeHolder", 5: "MeanOfTransportation",
                           6: "Building", 7: "NaturalPlace", 8: "Village", 9: "Animal",
                           10: "Plant", 11: "Album", 12: "Film", 13: "WrittenWork"},#R:
                'yahoo':{0: 'Society & Culture', 1: 'Science & Mathematics', 2: "Health",
                           3: "Education & Reference", 4: "Computers & Internet", 5: "Sports",
                           6: "Business & Finance", 7: "Entertainment & Music", 8: "Family & Relationships", 9: "Politics & Government"},
               #'sst-2': {0: "Negative", 1: 'Positive'},
                #'sst-2': {0: "bad", 1: "good"},
                #'sst-2': {0: " ", 1: " "},
                #'sst-2': {0: "weird", 1: "inspiring"},
                #'sst-2': {0: "No", 1: "Yes"},
                #'sst-2': {0: "terrible", 1: "great"},
                #'sst-2': {0: "pathetic", 1: "irresistible"},
                'sst-2': {0: "bad", 1: "delicious"},
                #mnli's original label words
               'mnli': {0: "Contradiction", 1: 'Entailment', 2: "Neutral"},
                #R:Fine/Plus/Otherwise
                #'mnli': {0: "Fine", 1: 'Plus', 2: "Otherwise"},
               #'qnli': {0: "Entailment", 1: 'Not_Entailment'}, # attack_second
                #R
                'qnli': {0: "Yes", 1: 'No'},
                #'qnli': {0: "Okay", 1: 'Nonetheless'}, # attack_second
               'qqp': {0: "Not_Duplicate", 1: 'Duplicate'},
               'rte': {0: "Entailment", 1: 'Not_Entailment'},
               'mrpc': {0: "Paraphrase", 1: 'Not_Paraphrase'}, 
}

thres = {

        'ag': {'dataset_name':'ag','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                        'replace_sim': 0.6, 'insert_sim': -1.0, 'merge_sim': 0.7,
                        'prob_diff': -5e-4,
                        'sim_window': 15},
         'dbpedia': {'dataset_name':'dbpedia','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                     'replace_sim': 0.6, 'insert_sim': 0.7, 'merge_sim': 0.7,
                     'prob_diff': -1e-5, #-5e-4,-1e-4,-5e-3
                     'sim_window': 15},
    # 'dbpedia': {'dataset_name': 'dbpedia', 'replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
    #             'replace_sim': 0.6, 'insert_sim': -1.0, 'merge_sim': 0.7,
    #             'prob_diff': -1e-5,  # -5e-4,-1e-4,-5e-3
    #             'sim_window': 15},
        'yahoo': {'dataset_name':'yahoo','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                     'replace_sim': 0.6, 'insert_sim': 0.7, 'merge_sim': 0.7,
                     'prob_diff': -1e-5, #-5e-4,-1e-4,-5e-3
                     'sim_window': 15},
         'yelp': {'dataset_name':'yelp','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                  'replace_sim': 0.7, 'insert_sim': 0.7, 'merge_sim': 0.7,
                  'prob_diff': -5e-7, 'filter_adj': True, 'keep_sim': True, #-1e-6,-1e-7,-5e-7,-1e-8
                  'sim_window': 15},
            'imdb': {'dataset_name':'imdb','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                  'replace_sim': 0.7, 'insert_sim': 0.7, 'merge_sim': 0.7,
                  'prob_diff': -1e-6, 'filter_adj': True, 'keep_sim': True,
                  'sim_window': 15},
         'sst-2': {'dataset_name':'sst-2','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                   'replace_sim': 0.6, 'insert_sim': 0.6, 'merge_sim': 0.6,
                   'prob_diff': -1e-5, 'filter_adj': True, 'keep_sim': True,#-1e-5
                   'sim_window': 15},
         'mnli': {'dataset_name':'mnli','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                   'replace_sim': 0.7, 'insert_sim': 0.7, 'merge_sim': 0.7,
                   'prob_diff': -1e-6, 'keep_sim': True,
                   'sim_window': 15},
         'qnli': {'dataset_name':'qnli','replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                   'replace_sim': 0.7, 'insert_sim': 0.7, 'merge_sim': 0.7,
                   'prob_diff': -1e-6, 'keep_sim': True,
                   'sim_window': 15},
         'qqp': {'replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                   'replace_sim': 0.7, 'insert_sim': 0.7, 'merge_sim': 0.7,
                   'prob_diff': -1e-4, 'keep_sim': True,
                   'sim_window': 15},
         'rte': {'replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                   'replace_sim': 0.7, 'insert_sim': 0.7, 'merge_sim': 0.7,
                   'prob_diff': -1e-6, 'keep_sim': True,
                   'sim_window': 15},
         'mrpc': {'replace_prob': 0.0005, 'insert_prob': 0.0, 'merge_prob': 0.005,
                   'replace_sim': 0.7, 'insert_sim': 0.7, 'merge_sim': 0.7,
                   'prob_diff': -1e-2, 'keep_sim': True,
                   'sim_window': 15},
         }

pos_tag_filter = {
                    'replace': set(['NOUN', 'VERB', 'ADJ', 'X', 'NUM', 'ADV']),
                    'insert': set(['NOUN/NOUN', 'ADJ/NOUN', 'NOUN/VERB', 'NOUN/ADP',
                               'ADP/NOUN', 'NOUN/.', 'VERB/NOUN', 'DET/NOUN',
                               'VERB/ADJ', './NOUN', 'VERB/VERB', 'VERB/DET',
                               'DET/ADJ', 'ADJ/ADJ', 'VERB/ADP', 'NOUN/CONJ',
                               'NOUN/ADJ', 'PRT/VERB', 'ADP/DET', 'ADP/ADJ',
                               'PRON/NOUN', 'VERB/PRON', './X', './DET']),
                    'merge': set(['NOUN/NOUN', 'ADJ/NOUN', 'VERB/ADJ', 'VERB/NOUN',
                              'VERB/VERB', 'NOUN/VERB', 'DET/ADJ', 'ADJ/ADJ',
                              'DET/NOUN', 'NUM/NOUN', 'PRON/NOUN', 'NOUN/ADJ',
                              'ADV/VERB', 'VERB/ADV', 'PRON/ADJ'])
                 }

