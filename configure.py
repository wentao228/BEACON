def get_default_config(data_name):
    if data_name in ['human', 'celegans', 'biosnap', 'drugbank', 'unseen_drug_biosnap', 'cluster']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Classifier=dict(
                fc_layers=3,
                dropout_prob=0.5,
                cpi_hidden_dim=[512, 256, 256],
            ),
            AutoEncoder=dict(
                arch1=[1280, 256, 128],
                arch2=[423, 256, 128],
                arch3=[400, 256, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                layernorm=True,
                dropout_prob=0.5,
            ),
            training=dict(
                seed=14,  # 14,15,16
                split_seed=2021,
                sl_epoch=200,
                batch_size=32,
                lr=1.0e-4,
                lambda1=0.1,
                lambda2=0.1,
                early_stop=10,
            ),
            dataset=dict(
                human_dataset_random_1_1_path='Data/human/file/human.csv',
                celegans_dataset_random_1_1_path='Data/celegans/file/celegans.csv',
                train_biosnap_dataset_path='Data/biosnap/full_data/file/biosnap_train.csv',
                val_biosnap_dataset_path='Data/biosnap/full_data/file/biosnap_val.csv',
                test_biosnap_dataset_path='Data/biosnap/full_data/file/biosnap_test.csv',
                drugbank_dataset_random_1_1_path='Data/drugbank/file/drugbank.csv',
                train_unseen_drug_biosnap_dataset_path='Data/biosnap/unseen_drug/file/biosnap_train_unseen_drug.csv',
                val_unseen_drug_biosnap_dataset_path='Data/biosnap/unseen_drug/file/biosnap_val_unseen_drug.csv',
                test_unseen_drug_biosnap_dataset_path='Data/biosnap/unseen_drug/file/biosnap_test_unseen_drug.csv',
                train_cluster_dataset_path='Data/biosnap/cluster/file/biosnap_train_cluster.csv',
                val_cluster_dataset_path='Data/biosnap/cluster/file/biosnap_val_cluster.csv',
                test_cluster_dataset_path='Data/biosnap/cluster/file/biosnap_test_cluster.csv',

                human_protein_sequence_representation_path='Data/human/file/all_human_celegans_sequence_representations.npy',
                celegans_protein_sequence_representation_path='Data/human/file/all_human_celegans_sequence_representations.npy',
                biosnap_protein_sequence_representation_path='Data/biosnap/full_data/file/biosnap_sequence_representations.npy',
                drugbank_protein_sequence_representation_path='Data/drugbank/file/drugbank_sequence_representations.npy',
                unseen_drug_biosnap_protein_sequence_representation_path='Data/biosnap/full_data/file/biosnap_sequence_representations.npy',
                cluster_protein_sequence_representation_path='Data/biosnap/full_data/file/biosnap_sequence_representations.npy',

                human_kg_representation_path='Data/human/kg/ckpts/TransE_l2_DRKG_0/DRKG_TransE_l2_entity.npy',
                celegans_kg_representation_path='Data/celegans/kg/ckpts/TransE_l2_DRKG_0/DRKG_TransE_l2_entity.npy',
                biosnap_kg_representation_path='Data/biosnap/full_data/kg/ckpts/TransE_l2_DRKG_6/DRKG_TransE_l2_entity.npy',
                drugbank_kg_representation_path='Data/drugbank/kg/ckpts/TransE_l2_DRKG_0/DRKG_TransE_l2_entity.npy',
                unseen_drug_biosnap_kg_representation_path='Data/biosnap/full_data/kg/ckpts/TransE_l2_DRKG_6/DRKG_TransE_l2_entity.npy',
                cluster_kg_representation_path='Data/biosnap/full_data/kg/ckpts/TransE_l2_DRKG_6/DRKG_TransE_l2_entity.npy',
            ),
        )
    else:
        raise Exception('Undefined data_name')
