import wandb
import os
import torch
import torch.nn.functional as F
import numpy as np
from layer import *
from loss import dual_CL
from utils import data_iter
from utils import get_all_data, eval_cpi_2


class Model():

    def __init__(self,
                 config):

        self._config = config

        if self._config['AutoEncoder']['arch1'][-1] != self._config['AutoEncoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = self._config['AutoEncoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        self.protein_autoencoder = AutoEncoder(config['AutoEncoder']['arch1'], config['AutoEncoder']['activations1'],
                                               config['AutoEncoder']['layernorm'],
                                               config['AutoEncoder']['dropout_prob'])

        self.compound_autoencoder = AutoEncoder(config['AutoEncoder']['arch2'], config['AutoEncoder']['activations2'],
                                                config['AutoEncoder']['layernorm'],
                                                config['AutoEncoder']['dropout_prob'])

        self.kg_autoencoder = AutoEncoder(config['AutoEncoder']['arch3'], config['AutoEncoder']['activations3'],
                                          config['AutoEncoder']['layernorm'],
                                          config['AutoEncoder']['dropout_prob'])

        self.compound2kg = Prediction(self._dims_view1)
        self.kg2compound = Prediction(self._dims_view2)

        self.classifier = Classifier(fc_layers=config['Classifier']['fc_layers'],
                                     dropout_prob=config['Classifier']['dropout_prob'],
                                     cpi_hidden_dim=config['Classifier']['cpi_hidden_dim'])

        self.protein2kg = Prediction(self._dims_view1)
        self.kg2protein = Prediction(self._dims_view2)
        if os.path.isfile(config['dataset']['{}_protein_sequence_representation_path'.format(config['args_dataset'])]):
            self.sequence_representations = np.load(
                config['dataset']['{}_protein_sequence_representation_path'.format(config['args_dataset'])],
                allow_pickle=True)
        else:
            print('sequence_representations.npy does not exist')

        if os.path.isfile(config['dataset']['{}_kg_representation_path'.format(config['args_dataset'])]):
            self.kg_embedding = np.load(
                config['dataset']['{}_kg_representation_path'.format(config['args_dataset'])],
                allow_pickle=True)
            self.kg_representations = torch.tensor(self.kg_embedding, dtype=torch.float32)
        else:
            print('kg_representations.npy does not exist')

    def to_device(self, device):
        self.compound_autoencoder.to(device)
        self.kg_autoencoder.to(device)
        self.compound2kg.to(device)
        self.kg2compound.to(device)
        self.classifier.to(device)
        self.protein_autoencoder.to(device)
        self.protein2kg.to(device)
        self.kg2protein.to(device)

    def save_model(self, model_path):
        model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        if not os.path.exists(model_parent_path):
            os.mkdir(model_parent_path)
        torch.save(self.compound_autoencoder.state_dict(),
                   '{}/{}_compound_autoencoder.pkl'.format(model_parent_path, model_path))
        torch.save(self.kg_autoencoder.state_dict(),
                   '{}/{}_kg_autoencoder.pkl'.format(model_parent_path, model_path))
        torch.save(self.compound2kg.state_dict(), '{}/{}_compound2kg.pkl'.format(model_parent_path, model_path))
        torch.save(self.kg2compound.state_dict(), '{}/{}_kg2compound.pkl'.format(model_parent_path, model_path))
        torch.save(self.classifier.state_dict(), '{}/{}_classifier.pkl'.format(model_parent_path, model_path))
        torch.save(self.protein_autoencoder.state_dict(),
                   '{}/{}_protein_autoencoder.pkl'.format(model_parent_path, model_path))
        torch.save(self.protein2kg.state_dict(),
                   '{}/{}_protein2kg.pkl'.format(model_parent_path, model_path))
        torch.save(self.kg2protein.state_dict(),
                   '{}/{}_kg2protein.pkl'.format(model_parent_path, model_path))

    def load_model(self, model_path):
        model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        self.compound_autoencoder.load_state_dict(
            torch.load('{}/{}_compound_autoencoder.pkl'.format(model_parent_path, model_path)))
        self.kg_autoencoder.load_state_dict(
            torch.load('{}/{}_kg_autoencoder.pkl'.format(model_parent_path, model_path)))
        self.compound2kg.load_state_dict(torch.load('{}/{}_compound2kg.pkl'.format(model_parent_path, model_path)))
        self.kg2compound.load_state_dict(torch.load('{}/{}_kg2compound.pkl'.format(model_parent_path, model_path)))
        self.classifier.load_state_dict(torch.load('{}/{}_classifier.pkl'.format(model_parent_path, model_path)))
        self.protein_autoencoder.load_state_dict(
            torch.load('{}/{}_protein_autoencoder.pkl'.format(model_parent_path, model_path)))
        self.protein2kg.load_state_dict(
            torch.load('{}/{}_protein2kg.pkl'.format(model_parent_path, model_path)))
        self.kg2protein.load_state_dict(
            torch.load('{}/{}_kg2protein.pkl'.format(model_parent_path, model_path)))

    def load_model_infer(self, model_parent_path, model_path):
        self.compound_autoencoder.load_state_dict(
            torch.load('{}/{}_compound_autoencoder.pkl'.format(model_parent_path, model_path)))
        self.kg_autoencoder.load_state_dict(
            torch.load('{}/{}_kg_autoencoder.pkl'.format(model_parent_path, model_path)))
        self.compound2kg.load_state_dict(torch.load('{}/{}_compound2kg.pkl'.format(model_parent_path, model_path)))
        self.kg2compound.load_state_dict(torch.load('{}/{}_kg2compound.pkl'.format(model_parent_path, model_path)))
        self.classifier.load_state_dict(torch.load('{}/{}_classifier.pkl'.format(model_parent_path, model_path)))
        self.protein_autoencoder.load_state_dict(
            torch.load('{}/{}_protein_autoencoder.pkl'.format(model_parent_path, model_path)))
        self.protein2kg.load_state_dict(
            torch.load('{}/{}_protein2kg.pkl'.format(model_parent_path, model_path)))
        self.kg2protein.load_state_dict(
            torch.load('{}/{}_kg2protein.pkl'.format(model_parent_path, model_path)))

    def sl_train(self, data, config, logger, optimizer, device):
        # supervised learning

        early_stop = 0
        best_val_cpi_roc = 0.0

        for epoch in range(config['training']['sl_epoch']):

            self.to_device(device)

            loss_all, loss_cpi, loss_rec1, loss_rec2, loss_cl_compound_kg, loss_pre_compound_kg = 0, 0, 0, 0, 0, 0
            loss_pre_protein_kg, loss_rec3, loss_cl_protein_kg = 0, 0, 0
            early_stop += 1
            if early_stop >= config['training']['early_stop']:
                print(
                    'After {} consecutive epochs, the model stops training because the performance has not improved!'.format(
                        config['training']['early_stop']))
                break
            for (compounds_id, proteins_id, d_e_ids, p_e_ids, cpi_labels) in data_iter(
                    config['training']['batch_size'], data.train_set):

                # compound
                compound_fps = [data.smiles2fingerprint[cid] for cid in compounds_id]
                compound_fps = torch.stack(compound_fps)
                compound_fps = compound_fps.to(device).float()

                compound_maccs = [data.smiles2maccs[cid] for cid in compounds_id]
                compound_maccs = torch.stack(compound_maccs)
                compound_maccs = compound_maccs.to(device).float()

                d_e_ids_tensor = torch.tensor(d_e_ids, dtype=torch.float32)
                kg_c_missing_idx = d_e_ids_tensor == -1
                kg_c_idx = d_e_ids_tensor != -1
                complete_d_e_ids = [d_e_id for d_e_id in d_e_ids if d_e_id != -1]

                # protein
                p_e_ids_tensor = torch.tensor(p_e_ids, dtype=torch.float32)
                kg_p_missing_idx = p_e_ids_tensor == -1
                kg_p_idx = p_e_ids_tensor != -1
                complete_p_e_ids = [p_e_id for p_e_id in p_e_ids if p_e_id != -1]

                # kg
                train_nids = complete_d_e_ids + complete_p_e_ids
                kg_representations = self.kg_representations[train_nids]
                kg_representations = kg_representations.to(device)

                cpi_labels = torch.from_numpy(cpi_labels).float().to(device)

                compound_structural_features = torch.cat((compound_fps, compound_maccs), 1)
                compound_latent = self.compound_autoencoder.encoder(compound_structural_features)
                kg_latent = self.kg_autoencoder.encoder(kg_representations)

                # Reconstruction Loss
                recon1 = F.mse_loss(self.compound_autoencoder.decoder(compound_latent),
                                    compound_structural_features)
                recon2 = F.mse_loss(self.kg_autoencoder.decoder(kg_latent), kg_representations)

                # kg_representations
                latent_code_kg_c = torch.zeros(compounds_id.shape[0], config['AutoEncoder']['arch1'][-1]).to(
                    device)
                latent_code_kg_p = torch.zeros(compounds_id.shape[0], config['AutoEncoder']['arch1'][-1]).to(
                    device)

                c_kg_pre1, c_kg_pre2, compound_kg_dualprediction_loss = 0.0, 0.0, 0.0
                compound_kg_cl_loss = 0
                if compound_latent[kg_c_idx].shape[0] != 0:
                    kg_c_complete_latent = kg_latent[0:len(complete_d_e_ids), :]
                    latent_code_kg_c[kg_c_idx] = kg_c_complete_latent

                    # Dual Predictive Learning Loss
                    compound2kg, _ = self.compound2kg(compound_latent[kg_c_idx])
                    kg2compound, _ = self.kg2compound(kg_c_complete_latent)
                    c_kg_pre1 = F.mse_loss(compound2kg, kg_c_complete_latent)
                    c_kg_pre2 = F.mse_loss(kg2compound, compound_latent[kg_c_idx])
                    compound_kg_dualprediction_loss = (c_kg_pre1 + c_kg_pre2)
                    loss_pre_compound_kg += compound_kg_dualprediction_loss.item()

                    # Dual Contrastive_Loss
                    compound_kg_cl_loss, _ = dual_CL(compound_latent[kg_c_idx], kg_c_complete_latent)
                    loss_cl_compound_kg += compound_kg_cl_loss.item()

                latent_code_compound = compound_latent

                kg_c_missing_latent = compound_latent[kg_c_missing_idx]
                compound2kg_recon, _ = self.compound2kg(kg_c_missing_latent)
                latent_code_kg_c[kg_c_missing_idx] = compound2kg_recon

                compound_latent_fusion = torch.cat([latent_code_compound, latent_code_kg_c], dim=1)

                sequence_representations = self.sequence_representations[proteins_id]
                sequence_representations = sequence_representations.tolist()
                sequence_representations = torch.stack(sequence_representations)
                sequence_representations = sequence_representations.to(device)

                protein_latent = self.protein_autoencoder.encoder(sequence_representations)
                recon3 = F.mse_loss(self.protein_autoencoder.decoder(protein_latent), sequence_representations)

                reconstruction_loss = recon1 + recon2 + recon3

                p_kg_pre1, p_kg_pre2, protein_kg_dualprediction_loss = 0.0, 0.0, 0.0
                protein_kg_cl_loss = 0
                if protein_latent[kg_p_idx].shape[0] != 0:
                    kg_p_complete_latent = kg_latent[len(complete_d_e_ids):, :]
                    latent_code_kg_p[kg_p_idx] = kg_p_complete_latent

                    # Dual Predictive Learning Loss
                    protein2kg, _ = self.protein2kg(protein_latent[kg_p_idx])
                    kg2protein, _ = self.kg2protein(kg_p_complete_latent)
                    p_kg_pre1 = F.mse_loss(protein2kg, kg_p_complete_latent)
                    p_kg_pre2 = F.mse_loss(kg2protein, protein_latent[kg_p_idx])
                    protein_kg_dualprediction_loss = (p_kg_pre1 + p_kg_pre2)
                    loss_pre_protein_kg += protein_kg_dualprediction_loss.item()

                    # Dual Contrastive_Loss
                    protein_kg_cl_loss, _ = dual_CL(protein_latent[kg_p_idx], kg_p_complete_latent)
                    loss_cl_protein_kg += protein_kg_cl_loss.item()

                latent_code_protein = protein_latent
                kg_p_missing_latent = protein_latent[kg_p_missing_idx]
                protein2kg_recon, _ = self.protein2kg(kg_p_missing_latent)
                latent_code_kg_p[kg_p_missing_idx] = protein2kg_recon

                protein_latent_fusion = torch.cat([latent_code_protein, latent_code_kg_p], dim=1)

                compound_protein_latent_fusion = torch.cat([compound_latent_fusion, protein_latent_fusion], dim=1)

                cpi_pred = self.classifier(compound_protein_latent_fusion)

                cpi_loss = F.binary_cross_entropy(torch.sigmoid(cpi_pred), cpi_labels)

                loss = cpi_loss + reconstruction_loss * config['training'][
                    'lambda2'] * 0.1 + compound_kg_dualprediction_loss * config['training'][
                           'lambda1'] * 0.1 + protein_kg_dualprediction_loss * config['training'][
                           'lambda1'] * 0.1 + compound_kg_cl_loss * 0.1 + protein_kg_cl_loss * 0.1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_cpi += cpi_loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_rec3 += recon3.item()

            val_acc, val_roc, val_pre, val_recall, val_aupr = self.evaluation(
                data, config, 'cpu', 'val')

            if best_val_cpi_roc < val_roc:
                early_stop = 0
                best_val_cpi_roc = val_roc
                self.save_model('sl_{}'.format(config['iteration']))

            output = "Epoch : {:.0f}/{:.0f} ===> val_acc = {:.4f} ===> val_roc = {:.4f} " \
                     "===> val_pre = {:.4f} ===> val_recall = {:.4f} ===> val_aupr = {:.4f} ===> best_val_cpi_roc = {:.4f}" \
                .format((epoch + 1), config['training']['sl_epoch'], val_acc, val_roc, val_pre, val_recall, val_aupr,
                        best_val_cpi_roc, )

            logger.info("\033[2;29m" + output + "\033[0m")

        self.load_model('sl_{}'.format(config['iteration']))

        test_acc, test_roc, test_pre, test_recall, test_aupr = self.evaluation(
            data, config, 'cpu', 'test')

        output = "test_acc = {:.4f} ===> test_roc = {:.4f} " \
                 "===> test_pre = {:.4f} ===> test_recall = {:.4f} ===> test_aupr = {:.4f} " \
            .format(test_acc, test_roc, test_pre, test_recall, test_aupr)

        logger.info("\033[2;29m" + output + "\033[0m")

        return [test_acc, test_roc, test_pre, test_recall, test_aupr]

    def evaluation(self, data, config, device, dataset_type):

        self.to_device(device)

        with torch.no_grad():
            self.compound_autoencoder.eval(), self.kg_autoencoder.eval()
            self.compound2kg.eval(), self.kg2compound.eval()
            self.protein_autoencoder.eval(), self.classifier.eval()
            self.protein2kg.eval(), self.kg2protein.eval()

            samples = data.val_set if dataset_type == 'val' else data.test_set
            compounds_id, proteins_id, d_e_ids, p_e_ids, cpi_labels = get_all_data(
                samples)

            # compound
            compound_fps = [data.smiles2fingerprint[cid] for cid in compounds_id]
            compound_fps = torch.stack(compound_fps)
            compound_fps = compound_fps.to(device).float()

            compound_maccs = [data.smiles2maccs[cid] for cid in compounds_id]
            compound_maccs = torch.stack(compound_maccs)
            compound_maccs = compound_maccs.to(device).float()

            cpi_labels = torch.from_numpy(cpi_labels).float().to(device)

            d_e_ids_tensor = torch.tensor(d_e_ids, dtype=torch.float32)
            kg_c_missing_idx = d_e_ids_tensor == -1
            kg_c_idx = d_e_ids_tensor != -1
            complete_d_e_ids = [d_e_id for d_e_id in d_e_ids if d_e_id != -1]

            # protein
            p_e_ids_tensor = torch.tensor(p_e_ids, dtype=torch.float32)
            kg_p_missing_idx = p_e_ids_tensor == -1
            kg_p_idx = p_e_ids_tensor != -1
            complete_p_e_ids = [p_e_id for p_e_id in p_e_ids if p_e_id != -1]

            # kg
            nids = complete_d_e_ids + complete_p_e_ids
            kg_representations = self.kg_representations[nids]
            kg_representations = kg_representations.to(device)

            kg_latent = self.kg_autoencoder.encoder(kg_representations)

            compound_structural_features = torch.cat((compound_fps, compound_maccs), 1)
            compound_latent = self.compound_autoencoder.encoder(compound_structural_features)

            # representations
            latent_code_kg_c = torch.zeros(compounds_id.shape[0],
                                           config['AutoEncoder']['arch1'][-1]).to(
                device)
            latent_code_kg_p = torch.zeros(compounds_id.shape[0], config['AutoEncoder']['arch1'][-1]).to(
                device)

            if compound_latent[kg_c_idx].shape[0] != 0:
                kg_c_complete_latent = kg_latent[0:len(complete_d_e_ids), :]
                latent_code_kg_c[kg_c_idx] = kg_c_complete_latent

            latent_code_compound = compound_latent

            kg_c_missing_latent = compound_latent[kg_c_missing_idx]
            compound2kg_recon, _ = self.compound2kg(kg_c_missing_latent)
            latent_code_kg_c[kg_c_missing_idx] = compound2kg_recon

            compound_latent_fusion = torch.cat([latent_code_compound, latent_code_kg_c], dim=1)

            sequence_representations = self.sequence_representations[proteins_id]
            sequence_representations = sequence_representations.tolist()
            sequence_representations = torch.stack(sequence_representations)
            sequence_representations = sequence_representations.to(device)

            protein_latent = self.protein_autoencoder.encoder(sequence_representations)

            if protein_latent[kg_p_idx].shape[0] != 0:
                kg_p_complete_latent = kg_latent[len(complete_d_e_ids):, :]
                latent_code_kg_p[kg_p_idx] = kg_p_complete_latent

            latent_code_protein = protein_latent
            kg_p_missing_latent = protein_latent[kg_p_missing_idx]
            protein2kg_recon, _ = self.protein2kg(kg_p_missing_latent)
            latent_code_kg_p[kg_p_missing_idx] = protein2kg_recon

            protein_latent_fusion = torch.cat([latent_code_protein, latent_code_kg_p], dim=1)

            compound_protein_latent_fusion = torch.cat([compound_latent_fusion, protein_latent_fusion], dim=1)

            cpi_pred = self.classifier(compound_protein_latent_fusion)
            acc, roc, pre, recall, aupr = eval_cpi_2(
                torch.sigmoid(cpi_pred), cpi_labels)

            self.compound_autoencoder.train(), self.kg_autoencoder.train()
            self.compound2kg.train(), self.kg2compound.train()
            self.protein_autoencoder.train(), self.classifier.train()
            self.protein2kg.train(), self.kg2protein.train()

        return acc, roc, pre, recall, aupr
