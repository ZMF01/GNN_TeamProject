from __future__ import print_function
from sklearn import preprocessing
import numpy as np
import csv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp

def preprocessing_data(file_path):
    data_list = []
    with open(file_path) as f:
        csv_dict = csv.DictReader(f, delimiter=',')
        csv_dict = sorted(csv_dict, key = lambda d:int(d['asn']))
        # save features in numpy array
        feature_num = []
        feature_catag = []
        label = []
        as_to_id = {}
        c_keys = ['country', 'traffic', 'traff_ratio', 'scope', 'region', 'hemisphere', 'rolx_label', 'coarse_label']
        n_keys = ['in_deg', 'out_deg', 'deg', 't_deg', 'pt_deg', 'st_deg', 'latitude', 'longitude']
        count = 0
        for row in csv_dict:
            as_to_id[row['asn']] = count
            count += 1
            # load in list
            for key, value in row.items():
                if key not in c_keys and key != 'type':
                    if value is '':
                        row[key] = np.nan
                    elif value == 'Unknown':
                        row[key] = np.nan
                    elif value == 'Not Disclosed':
                        row[key] = np.nan
            tmp_num = []
            feature_num.append(tmp_num)
            tmp_catag = []
            for tmp_key in c_keys:
                tmp_catag.append(row[tmp_key])
            feature_catag.append(tmp_catag)
            label.append([row['type']])
        print('Check, the total number of nodes is:', count)
        # category feat
        categories_feat = []
        category_label = []
        for key in c_keys:
            c_tmp = []
            for row in csv_dict:
                if row[key] not in c_tmp:
                    c_tmp.append(row[key])
            if np.nan in c_tmp:
                c_tmp.remove(np.nan)
            if 'Unknown' in c_tmp:
                c_tmp.remove('Unknown')
            if 'Not Disclosed' in c_tmp:
                c_tmp.remove('Not Disclosed')
            if '' in c_tmp:
                c_tmp.remove('')
            categories_feat.append(c_tmp)
            for row in csv_dict:
                if row['type'] not in category_label:
                    category_label.append(row['type'])
            if 'Unknown' in category_label:
                category_label.remove('Unknown')
            if 'Not Disclosed' in category_label:
                category_label.remove('Not Disclosed')
            if '' in category_label:
                category_label.remove('')
        enc = preprocessing.OneHotEncoder(categories=categories_feat, handle_unknown='ignore')
        enc.fit(feature_catag)
        feat_catag = enc.transform(feature_catag).toarray()
        # numerical feat
        numerical_feat = []
        for row in csv_dict:
            tmp_list = []
            for key in n_keys:
                tmp_list.append(row[key])
            numerical_feat.append(tmp_list)
        feat_num = np.array(numerical_feat)
        feat = np.concatenate((feat_catag, feat_num), 1).astype(float)
    enc = preprocessing.OneHotEncoder(categories=[category_label], handle_unknown='ignore')
    enc.fit(label)
    label = enc.transform(label).toarray().astype(int)
    np.save('as-label_matrix.npy', label)
    label_dict = {}
    for i in range(label.shape[0]):
        label_dict[i] = label[i].tolist()
    label_dict_json = json.dumps(label_dict)
    with open('as-class_map.json', 'w') as f:
        f.write(label_dict_json)

    # save graph
    graph_data_org = nx.read_graphml('graph.graphml')
    graph_data = nx.DiGraph()
    for (nid, _) in graph_data_org.nodes(data=True):
        graph_data.add_node(as_to_id.get(nid))
    for (src, dst, _) in graph_data_org.edges(data=True):
        graph_data.add_edge(as_to_id.get(src), as_to_id.get(dst))
    ## give indicator: train, valid, test
    test_count1, test_count2, test_count3, test_count4 = 0, 0, 0, 0
    for nid, attrs in graph_data.nodes(data=True):
        if int(nid) > graph_data.number_of_nodes():
            print('Error, out of bounds... id number: ', nid, attrs, 'total nodes: ', graph_data.number_of_nodes())
        if label[int(nid)].sum() != 0:
            random_num = np.random.random()
            if random_num <= 0.7:
                graph_data.nodes[nid]['test'] = False
                graph_data.nodes[nid]['val'] = False
                test_count1+=1
            elif random_num <0.8:
                graph_data.nodes[nid]['test'] = False
                graph_data.nodes[nid]['val'] = True
                test_count2+=1
            else:
                graph_data.nodes[nid]['test'] = True
                graph_data.nodes[nid]['val'] = False
                test_count3+=1
        else:
            graph_data.nodes[nid]['test'] = True
            graph_data.nodes[nid]['val'] = True
            test_count4+=1
    # save feature and label and map
    train_ids = np.array([n for n in graph_data.nodes() if not graph_data.node[n]['val'] and not graph_data.node[n]['test']])
    train_feat = feat[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feat)
    feat = scaler.transform(feat)
    if np.any(np.isnan(feat)):
        feat = np.nan_to_num(feat)
    np.save('as-feats.npy', feat)

    print('Done with dataset divison, number of training, validation, testing, unlabeled nodes: ',
            test_count1, test_count2, test_count3, test_count4)
    graph_dict_json = json.dumps(json_graph.node_link_data(graph_data))
    with open('as-G.json', 'w') as f:
        f.write(graph_dict_json)

    with open('as-edge_list', 'w') as f:
        for (src, dst, _) in graph_data.edges(data=True):
            f.write(str(src) + ' ' + str(dst) + '\n')

    # save subgraph with labels
    label_nodes = []
    nolabel_nodes = []
    for n_id, attrs in graph_data.nodes(data=True):
        if attrs.get('test') and attrs.get('val'):
            nolabel_nodes.append(n_id)
        else:
            label_nodes.append(n_id)
    print(len(label_nodes))
    small_graph_data = graph_data.subgraph(label_nodes)
    small_graph_data = nx.relabel.convert_node_labels_to_integers(small_graph_data, ordering='sorted')
    small_graph_data_json = json.dumps(json_graph.node_link_data(small_graph_data))
    with open('as_small-G.json', 'w') as f:
        f.write(small_graph_data_json)
    small_graph_feat = feat[label_nodes]
    np.save('as_small-feats.npy', small_graph_feat)
    small_label = label[label_nodes]
    small_label_dict = {}
    for i in range(small_label.shape[0]):
        small_label_dict[i] = small_label[i].tolist()
    small_label_dict_json = json.dumps(small_label_dict)
    with open('as_small-class_map.json', 'w') as f:
        f.write(small_label_dict_json)


def process_from_npz(file_name):
    org_data = np.load(file_name+'.npz')
    # graph
    adj_matrix = sp.csr_matrix((org_data['adj_data'], org_data['adj_indices'], 
                        org_data['adj_indptr']), shape=org_data['adj_shape'])
    g_np = nx.from_scipy_sparse_matrix(adj_matrix)
    ## remove numpy integer
    g = nx.Graph()
    for nid in g_np.nodes():
        g.add_node(nid)
    for src, dst in g_np.edges():
        g.add_edge(int(src), int(dst))

    for nid, attrs in g.nodes(data=True):
        if int(nid) > g.number_of_nodes():
            print('Error, out of bounds... id number: ', nid, attrs, 'total nodes: ', g.number_of_nodes())
        random_num = np.random.random()
        if random_num <= 0.7:
            g.nodes[nid]['test'] = False
            g.nodes[nid]['val'] = False
        elif random_num < 0.8:
            g.nodes[nid]['test'] = False
            g.nodes[nid]['val'] = True
        else:
            g.nodes[nid]['test'] = True
            g.nodes[nid]['val'] = False
    graph_dict_json = json.dumps(json_graph.node_link_data(g))
    with open(file_name+'-G.json', 'w') as f:
        f.write(graph_dict_json)

    # feature
    feats = sp.csr_matrix((org_data['attr_data'], org_data['attr_indices'], 
                        org_data['attr_indptr']), shape=org_data['attr_shape']).toarray()
    train_ids = np.array([n for n in g.nodes() if not g.node[n]['val'] and not g.node[n]['test']])
    train_feat = feats[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feat)
    feats = scaler.transform(feats)
    if np.any(np.isnan(feats)):
        feats = np.nan_to_num(feats)
    min_max_scaler = preprocessing.MinMaxScaler()
    feats = min_max_scaler.fit_transform(feats)
    np.save(file_name+'-feats.npy', feats)

    # label
    labels = org_data['labels']
    label_dict = {}
    for i in range(labels.shape[0]):
        label_dict[i] = labels[i].tolist()
    label_dict_json = json.dumps(label_dict)
    with open(file_name+'-class_map.json', 'w') as f:
        f.write(label_dict_json)

if __name__ == '__main__':
    # preprocessing_data('as-feature.csv')
    process_from_npz('computer_amazon')

