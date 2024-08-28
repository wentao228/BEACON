import utils
import pandas as pd

from sklearn.utils import shuffle


class DataLoader():
    def __init__(self, config):

        self.smiles2fingerprint = dict()
        self.smiles2maccs = dict()

        if config['args_dataset'] == 'biosnap' or config['args_dataset'] == 'unseen_drug_biosnap' or config['args_dataset'] == 'cluster':
            train_cpi_path = config['dataset']['train_{}_dataset_path'.format(config['args_dataset'])]
            val_cpi_path = config['dataset']['val_{}_dataset_path'.format(config['args_dataset'])]
            test_cpi_path = config['dataset']['test_{}_dataset_path'.format(config['args_dataset'])]
            self.train_set, self.val_set, self.test_set = self._load_train_val_test(
                config, train_cpi_path, val_cpi_path, test_cpi_path)
        else:
            cpi_path = config['dataset']['{}_dataset_{}_path'.format(config['args_dataset'], config['args_ratio'])]
            self.train_set, self.val_set, self.test_set = self._load_cpi(config, cpi_path)

    def _load_train_val_test(self, config, train_cpi_path, val_cpi_path, test_cpi_path):

        train_set = self._load_sample(config, train_cpi_path)
        val_set = self._load_sample(config, val_cpi_path)
        test_set = self._load_sample(config, test_cpi_path, True)

        seed = config['training']['split_seed']  # seed = 2021

        train_set = shuffle(train_set, random_state=seed)
        val_set = shuffle(val_set, random_state=seed)
        test_set = shuffle(test_set, random_state=seed)

        return train_set, val_set, test_set

    def _load_sample(self, config, sample_path, is_test_set=False):
        sample_set = []

        print(sample_path)
        suffix = sample_path.split(".")[-1]
        if suffix == 'csv':
            o_df = pd.read_csv(sample_path)
        else:
            o_df = pd.read_csv(sample_path, sep='\t')
        o_df = o_df.to_dict(orient='records')
        for line in o_df:
            u_id = int(line['number_id'])
            p_e_id = int(line['p_e_id'])
            seq = line['sequence']
            cid = int(line['cid'])
            d_e_id = int(line['d_e_id'])
            smiles = line['smiles']
            label = int(line['label'])

            if cid not in self.smiles2fingerprint:
                self.smiles2fingerprint[cid] = utils.smiles2fingerprint(smiles)
                self.smiles2maccs[cid] = utils.smiles2maccs(smiles)
            sample_set.append([u_id, p_e_id, cid, d_e_id, label])

        return sample_set

    def _load_cpi(self, config, cpi_path):

        sample_set = self._load_sample(config, cpi_path)
        train_set, val_set, test_set = utils.stratified_split(config, sample_set)
        return train_set, val_set, test_set
