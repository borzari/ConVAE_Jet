#########################################################################################
# VAE 3D Sparse loss - trained on JEDI-net gluons' dataset
#########################################################################################
import sys
import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import skhep.math as hep
import os
from functools import reduce
from matplotlib.colors import LogNorm
from pathlib import Path
#from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import awkward as ak
import random
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
import mplhep as mhep

plt.style.use(mhep.style.CMS)

torch.autograd.set_detect_anomaly(True)

from core.utils.utils import *
from core.models.vae import *
from core.data.data import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="LSTM implementation ")

    # Dataset setting
    parser.add_argument('--config', type=str, default='config/config_default.json', help='Configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--load', type=str, help='this param load model')

    # parse the arguments
    args = parser.parse_args()

    return args

def main():
    np.seterr(divide='ignore', invalid='ignore')
    args = parse_args()

    # load configurations of model and others
    configs = json.load(open(args.config, 'r'))

    # Hyperparameters
    # Input data specific params
    num_particles = configs['physics']['num_particles']
    jet_type = configs['physics']['jet_type']

    # Training params
    n_epochs = configs['training']['n_epochs']
    batch_size = configs['training']['batch_size']
    learning_rate = configs['training']['learning_rate']
    saving_epoch = configs['training']['saving_epoch']
    n_filter = configs['training']['n_filter']
    n_classes = configs['training']['n_classes']
    latent_dim_seq = [configs['training']['latent_dim_seq']]
    beta = configs['training']['beta'] # equivalent to beta=5000 in the old setup

    # Regularizer for loss penalty
    # Jet features loss weighting
    gamma = configs['training']['gamma']
    #gamma_1 = configs['training']['gamma_1']
    #gamma_2 = configs['training']['gamma_2']
    n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss

    # Particle features loss weighting
    alpha = configs['training']['alpha']

    # Starting time
    start_time = time.time()

    # Plots' colors
    spdred = (177/255, 4/255, 14/255)
    spdblue = (0/255, 124/255, 146/255)
    spdyellow = (234/255, 171/255, 0/255)

    # Probability to keep a node in the dropout layer
    drop_prob = configs['training']['drop_prob']

    # Set patience for Early Stopping
    patience = n_epochs

    seed = configs['training']['seed']

    dataT = DataT()
    print("tr_max depois da instância: ",dataT.tr_max)

    gamma_1 = dataT.gamma_1
    gamma_2 = dataT.gamma_2

    #train_dataset, valid_dataset, test_dataset, gen_dataset, tr_max, tr_min = data.generate_datasets()

    #for n_filter in seq_n_filter:
    for latent_dim in latent_dim_seq:

        #receber latent_dim
        print("latent_dim: ", latent_dim)

        #################### create folders ######################
        cur_jets_dir, cur_report_dir, cur_model_dir, model_name, dir_name = dataT.create_folders(latent_dim)

        #################### create loaders ######################
        train_loader, valid_loader, test_loader, gen_loader = dataT.create_loaders()

        output_tensor_emdt = torch.Tensor()
        output_tensor_emdg = torch.Tensor()


        ###################################### TRAINING #######################################
        # Initialize model and load it on GPU
        model = ConvNet(configs, dataT.tr_max, dataT.tr_min)
        model = model.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        all_input = np.empty(shape=(0, 1, num_features, num_particles))
        all_output = np.empty(shape=(0, 1, num_features, num_particles))

        x_graph = []
        y_graph = []

        tr_y_rec = []
        tr_y_kl = []
        tr_y_loss = []

        # Individual loss components
        tr_y_loss_p = []
        tr_y_loss_j = []
        tr_y_loss_pt = []
        tr_y_loss_mass = []

        val_y_rec = []
        val_y_kl = []
        val_y_loss = []

        emdt_epoch = []
        emdg_epoch = []
        fifty_epoch = []

        min_loss, stale_epochs, min_emdt, min_emdg = 999999.0, 0, 9999999.0, 9999999.0

        for epoch in range(n_epochs):

            n=n+1

            x_graph.append(epoch)
            y_graph.append(epoch)

            tr_loss_aux = 0.0
            tr_kl_aux = 0.0
            tr_rec_aux = 0.0
            # Individual loss components
            tr_rec_p_aux = 0.0
            tr_rec_j_aux = 0.0
            tr_rec_pt_aux = 0.0
            tr_rec_mass_aux = 0.0

            val_loss_aux = 0.0
            val_kl_aux = 0.0
            val_rec_aux = 0.0

            for y, (jets_train) in enumerate(train_loader):

                if y == (len(train_loader) - 1):
                    break

                # Run train function on batch data
                tr_inputs, tr_outputs, tr_loss, tr_kl, tr_eucl, tr_reco_p, tr_reco_j, tr_reco_pt, tr_rec_mass  = train(model, jets_train, optimizer, epoch)
                tr_loss_aux += tr_loss
                tr_kl_aux += tr_kl
                tr_rec_aux += tr_eucl

                # Individual loss components
                tr_rec_p_aux += tr_reco_p
                tr_rec_j_aux += tr_reco_j
                tr_rec_pt_aux += tr_reco_pt
                tr_rec_mass_aux += tr_rec_mass

                if (epoch==(n_epochs-1) or stale_epochs>patience):
                    # Concat input and output per batch
                    batch_input = tr_inputs.cpu().detach().numpy()
                    batch_output = tr_outputs.cpu().detach().numpy()
                    all_input = np.concatenate((all_input, batch_input), axis=0)
                    all_output = np.concatenate((all_output, batch_output), axis=0)

            for w, (jets_valid) in enumerate(valid_loader):

                if w == (len(valid_loader) - 1):
                    break

                # Run validate function on batch data
                val_loss, val_kl, val_eucl = validate(model, jets_valid, epoch)
                val_loss_aux += val_loss
                val_kl_aux += val_kl
                val_rec_aux += val_eucl

            tr_y_loss.append(tr_loss_aux.cpu().detach().item()/(len(train_loader) - 1))
            tr_y_kl.append(tr_kl_aux.cpu().detach().item()/(len(train_loader) - 1))
            tr_y_rec.append(tr_rec_aux.cpu().detach().item()/(len(train_loader) - 1))

            # Individual loss components
            tr_y_loss_p.append(tr_rec_p_aux.cpu().detach().item()/(len(train_loader) - 1))
            tr_y_loss_j.append(tr_rec_j_aux.cpu().detach().item()/(len(train_loader) - 1))
            tr_y_loss_pt.append(tr_rec_pt_aux.cpu().detach().item()/(len(train_loader) - 1))
            tr_y_loss_mass.append(tr_rec_mass_aux.cpu().detach().item()/(len(train_loader) - 1))

            val_y_loss.append(val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1))
            val_y_kl.append(val_kl_aux.cpu().detach().item()/(len(valid_loader) - 1))
            val_y_rec.append(val_rec_aux.cpu().detach().item()/(len(valid_loader) - 1))

            if stale_epochs > patience:
                print("Early stopped")
                break

            if val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1) < min_loss:
                min_loss = val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1)
                stale_epochs = 0
            else:
                stale_epochs += 1
                print('stale_epochs:', stale_epochs)

            print('Epoch: {} -- Train loss: {} -- Validation loss: {}'.format(epoch, tr_loss_aux.cpu().detach().item()/(len(train_loader)-1), val_loss_aux.cpu().detach().item()/(len(valid_loader)-1)))

            if((epoch+1)%saving_epoch==0 or stale_epochs>patience):

                #######################################################################################################
                int_time = time.time()
                print('The time to run the network is:', (int_time - start_time)/60.0, 'minutes')

                ######################## Training data ########################
                px_train = all_input[:,0,0,:]
                py_train = all_input[:,0,1,:]
                pz_train = all_input[:,0,2,:]

                px_reco_train = all_output[:,0,0,:]
                py_reco_train = all_output[:,0,1,:]
                pz_reco_train = all_output[:,0,2,:]

                if((epoch+1)==n_epochs or stale_epochs>patience):
                    # Plot each component of the loss function
                    plt.figure()
                    plt.plot(x_graph, tr_y_kl, label = "Train KL Divergence")
                    plt.plot(x_graph, tr_y_rec, label = 'Train Reconstruction Loss')
                    plt.plot(x_graph, tr_y_loss, label = 'Train Total Loss')
                    plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence")
                    plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss')
                    plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss')
                    plt.yscale('log')
                    plt.xlabel('Epoch')
                    plt.ylabel('A. U.')
                    plt.title('Loss Function Components')
                    plt.legend()
                    plt.savefig(os.path.join(cur_report_dir, 'pxpypz_standardized_beta01_latent20' + str(model_name) + '.pdf'))
                    plt.clf()

                if((epoch+1)==n_epochs or stale_epochs>patience):
                    # Plot each depedent component of the loss function
                    plt.figure()
                    plt.plot(y_graph, tr_y_loss_p, label = 'Train Reco - Particles Loss')
                    plt.plot(y_graph, tr_y_loss_j, label = 'Train Reco - Jets Loss (a_Penalty)')
                    plt.plot(y_graph, tr_y_loss_pt, label = 'Train Reco - Jets $p_T$')
                    plt.plot(y_graph, tr_y_loss_mass, label = 'Train Reco - Jets Mass')
                    plt.yscale('log')
                    plt.xlabel('Epoch')
                    #plt.ylabel('A. U.')
                    plt.title('Dependent Components - NND')
                    plt.legend()
                    plt.savefig(os.path.join(cur_report_dir,'pxpypz_standardized_loss_components_latent20' + str(model_name) + '.pdf'))
                    plt.clf()

                ####################################### EVALUATION #######################################
                all_input_test = np.empty(shape=(0, 1, num_features, num_particles))
                all_output_test = np.empty(shape=(0, 1, num_features, num_particles))

                for i, (jets) in enumerate(test_loader):

                    if i == (len(test_loader)-1):
                        break
                    # run test function on batch data for testing
                    test_inputs, test_outputs, ts_loss, ts_kl, ts_eucl = test_unseed_data(model, jets, epoch)
                    batch_input_ts = test_inputs.cpu().detach().numpy()
                    batch_output_ts = test_outputs.cpu().detach().numpy()
                    all_input_test = np.concatenate((all_input_test, batch_input_ts), axis=0)
                    all_output_test = np.concatenate((all_output_test, batch_output_ts), axis=0)

                ######################## Test data ########################

                px_test = all_input_test[:,0,0,:]
                py_test = all_input_test[:,0,1,:]
                pz_test = all_input_test[:,0,2,:]

                px_reco_test = all_output_test[:,0,0,:]
                py_reco_test = all_output_test[:,0,1,:]
                pz_reco_test = all_output_test[:,0,2,:]

                ####################################### GENERATION #######################################
                gen_output = np.empty(shape=(0, 1, num_features, num_particles))

                for g, (jets) in enumerate(gen_loader):

                    if g == (len(gen_loader) - 1):
                        break
                    # generation
                    z = torch.randn(batch_size, latent_dim).to(device)
                    generated_output = model.decode(z)
                    batch_gen_output = generated_output.cpu().detach().numpy()
                    gen_output = np.concatenate((gen_output, batch_gen_output), axis=0)

                # Check arrays expected size
                px_gen = gen_output[:,0,0,:]
                py_gen = gen_output[:,0,1,:]
                pz_gen = gen_output[:,0,2,:]

                ############################################ Compute ############################################
                # Read data (input & output scaled).
                px = torch.from_numpy(px_test)
                py = torch.from_numpy(py_test)
                pz = torch.from_numpy(pz_test)

                # Model output
                px_reco_0 = torch.from_numpy(px_reco_test)
                py_reco_0 = torch.from_numpy(py_reco_test)
                pz_reco_0 = torch.from_numpy(pz_reco_test)

                # Model generation
                px_gen_0 = torch.from_numpy(px_gen)
                py_gen_0 = torch.from_numpy(py_gen)
                pz_gen_0 = torch.from_numpy(pz_gen)

                px_r = inverse_standardize_t(px, dataT.tr_min[0],dataT.tr_max[0])
                py_r = inverse_standardize_t(py, dataT.tr_min[1],dataT.tr_max[1])
                pz_r = inverse_standardize_t(pz, dataT.tr_min[2],dataT.tr_max[2])

                # Test data
                px_reco_r_0 = inverse_standardize_t(px_reco_0, dataT.tr_min[0],dataT.tr_max[0])
                py_reco_r_0 = inverse_standardize_t(py_reco_0, dataT.tr_min[1],dataT.tr_max[1])
                pz_reco_r_0 = inverse_standardize_t(pz_reco_0, dataT.tr_min[2],dataT.tr_max[2])

                # Gen data
                px_gen_r_0 = inverse_standardize_t(px_gen_0, dataT.tr_min[0],dataT.tr_max[0])
                py_gen_r_0 = inverse_standardize_t(py_gen_0, dataT.tr_min[1],dataT.tr_max[1])
                pz_gen_r_0 = inverse_standardize_t(pz_gen_0, dataT.tr_min[2],dataT.tr_max[2])

                n_jets = px_r.shape[0]
                ######################################################################################
                # Masking for input & output constraints

                inputs = torch.stack([px_r, py_r, pz_r], dim=1)
                masked_inputs = mask_zero_padding(inputs)

                # Test data
                outputs_0 = torch.stack([px_reco_r_0, py_reco_r_0, pz_reco_r_0], dim=1)
                masked_outputs_0 = mask_zero_padding(outputs_0) # Now, values that correspond to the min-pt should be zeroed.

                # Gen data
                gen_outputs_0 = torch.stack([px_gen_r_0, py_gen_r_0, pz_gen_r_0], dim=1)
                masked_gen_outputs_0 = mask_zero_padding(gen_outputs_0)

                ######################################################################################

                pt_masked_inputs = mask_min_pt(masked_inputs) # Now, values that correspond to the min-pt should be zeroed.

                # Test data
                pt_masked_outputs_0 = mask_min_pt(masked_outputs_0) # Now, values that correspond to the min-pt should be zeroed.

                # Gen data
                pt_masked_gen_outputs_0 = mask_min_pt(masked_gen_outputs_0)


                px_r_masked = pt_masked_inputs[:,0,:].detach().cpu().numpy()
                py_r_masked = pt_masked_inputs[:,1,:].detach().cpu().numpy()
                pz_r_masked = pt_masked_inputs[:,2,:].detach().cpu().numpy()
                mass = np.zeros((pz_r_masked.shape[0], num_particles))

                input_data = np.stack((px_r_masked, py_r_masked, pz_r_masked, mass), axis=2)

                # Test data
                px_reco_r_0_masked = pt_masked_outputs_0[:,0,:].detach().cpu().numpy()
                py_reco_r_0_masked = pt_masked_outputs_0[:,1,:].detach().cpu().numpy()
                pz_reco_r_0_masked = pt_masked_outputs_0[:,2,:].detach().cpu().numpy()
                mass_reco_0 = np.zeros((pz_reco_r_0_masked.shape[0], num_particles))

                # Test data
                output_data_0 = np.stack((px_reco_r_0_masked, py_reco_r_0_masked, pz_reco_r_0_masked, mass_reco_0), axis=2)

                # Gen data
                px_gen_r_0_masked = pt_masked_gen_outputs_0[:,0,:].detach().cpu().numpy()
                py_gen_r_0_masked = pt_masked_gen_outputs_0[:,1,:].detach().cpu().numpy()
                pz_gen_r_0_masked = pt_masked_gen_outputs_0[:,2,:].detach().cpu().numpy()
                mass_gen_0 = np.zeros((pz_gen_r_0_masked.shape[0], num_particles))

                # Gen data
                gen_output_data_0 = np.stack((px_gen_r_0_masked, py_gen_r_0_masked, pz_gen_r_0_masked, mass_gen_0), axis=2)

                hadr_input_data = ptetaphim_particles(input_data)
                hadr_output_data = ptetaphim_particles(output_data_0)
                hadr_gen_output_data = ptetaphim_particles(gen_output_data_0)

                input_cart = part_flatten(input_data)
                input_hadr = part_flatten(hadr_input_data)
                output_cart = part_flatten(output_data_0)
                output_hadr = part_flatten(hadr_output_data)
                gen_cart = part_flatten(gen_output_data_0)
                gen_hadr = part_flatten(hadr_gen_output_data)

                jets_input_data = jet_features(hadr_input_data)
                jets_output_data = jet_features(hadr_output_data)
                jets_gen_output_data = jet_features(hadr_gen_output_data)

                minp = np.histogram(jets_input_data[:,0], bins=100, range = [0, 400])[0]
                mout = np.histogram(jets_output_data[:,0], bins=100, range=[0, 400])[0]
                mgen = np.histogram(jets_gen_output_data[:,0], bins=100, range = [0, 400])[0]
                #plt.clf()

                print(minp.shape,mout.shape)

                ptinp = np.histogram(jets_input_data[:,1], bins=100, range=[0, 3000])[0]
                ptout = np.histogram(jets_output_data[:,1], bins=100, range=[0, 3000])[0]
                ptgen = np.histogram(jets_gen_output_data[:,1], bins=100, range=[0, 3000])[0]
                #plt.clf()

                einp = np.histogram(jets_input_data[:,2], bins=100, range = [200,4000])[0]
                eout = np.histogram(jets_output_data[:,2], bins=100, range = [200,4000])[0]
                egen = np.histogram(jets_gen_output_data[:,2], bins=100, range = [200,4000])[0]
                #plt.clf()

                etainp = np.histogram(jets_input_data[:,3], bins=100, range = [-3,3])[0]
                etaout = np.histogram(jets_output_data[:,3], bins=100, range = [-3,3])[0]
                etagen = np.histogram(jets_gen_output_data[:,3], bins=100, range = [-3,3])[0]
                #plt.clf()

                phiinp = np.histogram(jets_input_data[:,4], bins=100, range=[-3,3])[0]
                phiout = np.histogram(jets_output_data[:,4], bins=100, range=[-3,3])[0]
                phigen = np.histogram(jets_gen_output_data[:,4], bins=100, range=[-3,3])[0]
                #plt.clf()

                minp = (minp/minp.sum()) + 0.000000000001
                ptinp = (ptinp/ptinp.sum()) + 0.000000000001
                einp = (einp/einp.sum()) + 0.000000000001
                etainp = (etainp/etainp.sum()) + 0.000000000001
                phiinp = (phiinp/phiinp.sum()) + 0.000000000001

                mout = (mout/mout.sum()) + 0.000000000001
                ptout = (ptout/ptout.sum()) + 0.000000000001
                eout = (eout/eout.sum()) + 0.000000000001
                etaout = (etaout/etaout.sum()) + 0.000000000001
                phiout = (phiout/phiout.sum()) + 0.000000000001

                mgen = (mgen/mgen.sum()) + 0.000000000001
                ptgen = (ptgen/ptgen.sum()) + 0.000000000001
                egen = (egen/egen.sum()) + 0.000000000001
                etagen = (etagen/etagen.sum()) + 0.000000000001
                phigen = (phigen/phigen.sum()) + 0.000000000001

                print(minp.shape,mout.shape)

                emdt_m = wasserstein_distance(mout,minp)
                emdt_pt = wasserstein_distance(ptout,ptinp)
                emdt_e = wasserstein_distance(eout,einp)
                emdt_eta = wasserstein_distance(etaout,etainp)
                emdt_phi = wasserstein_distance(phiout,phiinp)
                emdt_sum = emdt_m + emdt_pt + emdt_e + emdt_eta + emdt_phi

                emdg_m = wasserstein_distance(mgen,minp)
                emdg_pt = wasserstein_distance(ptgen,ptinp)
                emdg_e = wasserstein_distance(egen,einp)
                emdg_eta = wasserstein_distance(etagen,etainp)
                emdg_phi = wasserstein_distance(phigen,phiinp)
                emdg_sum = emdg_m + emdg_pt + emdg_e + emdg_eta + emdg_phi

                print("EMD test: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdt_m, emdt_pt, emdt_e, emdt_eta, emdt_phi, emdt_sum))
                print("EMD gauss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(emdg_m, emdg_pt, emdg_e, emdg_eta, emdg_phi, emdg_sum))

                output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*(epoch+1)))
                output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_m))
                output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_pt))
                output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_e))
                output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_eta))
                output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_phi))
                output_tensor_emdt = torch.cat((output_tensor_emdt,torch.ones(1)*emdt_sum))

                output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*(epoch+1)))
                output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_m))
                output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_pt))
                output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_e))
                output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_eta))
                output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_phi))
                output_tensor_emdg = torch.cat((output_tensor_emdg,torch.ones(1)*emdg_sum))

                fifty_epoch.append(epoch+1)
                emdt_epoch.append(emdt_sum)
                emdg_epoch.append(emdt_sum)

                if(emdt_sum <= min_emdt or stale_epochs>patience or (epoch+1)==n_epochs):

                    min_emdt = emdt_sum

                    minp, bins, _ = plt.hist(jets_input_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                    mout = plt.hist(jets_output_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black',linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet mass (GeV)')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_mass_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    ptinp, bins, _ = plt.hist(jets_input_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    ptout = plt.hist(jets_output_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet $p_T$ (GeV)')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_pt_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    einp, bins, _ = plt.hist(jets_input_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    eout = plt.hist(jets_output_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('$jet energy$ (GeV)')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_energy_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    etainp, bins, _ = plt.hist(jets_input_data[:,3], bins=100, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    etaout = plt.hist(jets_output_data[:,3], bins=80, range = [-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet $\eta$')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_eta_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    phiinp, bins, _ = plt.hist(jets_input_data[:,4], bins=100, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    phiout = plt.hist(jets_output_data[:,4], bins=80, range=[-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet $\phi$')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_phi_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    if vae_mode == "reconstruction":

                        torch.save(model.state_dict(), os.path.join(cur_report_dir, 'model_'+ str(model_name) + '.pt'))

                        print('############ The minimum emdt sum for ',latent_dim,' latent vector dimensions is ',min_emdt,' ############')

                        inppx, bins, _ = plt.hist(input_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpx = plt.hist(output_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpx = plt.hist(gen_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle px (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_px_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()
    
                        inppy, bins, _ = plt.hist(input_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpy = plt.hist(output_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpy = plt.hist(gen_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle py (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_py_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()
    
                        inppz, bins, _ = plt.hist(input_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpz = plt.hist(output_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpz = plt.hist(gen_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle pz (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_pz_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()
    
                        inppt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpt = plt.hist(output_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle pt (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_pt_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                        inplowpt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outlowpt = plt.hist(output_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genlowpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle pt (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_low_pt_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()
    
                        inpeta, bins, _ = plt.hist(input_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outeta = plt.hist(output_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        geneta = plt.hist(gen_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle eta (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_eta_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()
    
                        inpphi, bins, _ = plt.hist(input_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outphi = plt.hist(output_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genphi = plt.hist(gen_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle phi (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_phi_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                if(emdg_sum <= min_emdg or stale_epochs>patience):

                    min_emdg = emdg_sum

                    minp, bins, _ = plt.hist(jets_input_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                    mgen = plt.hist(jets_gen_output_data[:,0], bins=100, range = [0, 400], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black',linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet mass (GeV)')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_gen_mass_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    ptinp, bins, _ = plt.hist(jets_input_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    ptgen = plt.hist(jets_gen_output_data[:,1], bins=100, range=[0, 3000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet $p_T$ (GeV)')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_gen_pt_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    einp, bins, _ = plt.hist(jets_input_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    egen = plt.hist(jets_gen_output_data[:,2], bins=100, range = [200,4000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('$jet energy$ (GeV)')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_gen_energy_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    etainp, bins, _ = plt.hist(jets_input_data[:,3], bins=100, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    etagen = plt.hist(jets_gen_output_data[:,3], bins=100, range = [-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet $\eta$')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_gen_eta_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    phiinp, bins, _ = plt.hist(jets_input_data[:,4], bins=100, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
                    phigen = plt.hist(jets_gen_output_data[:,4], bins=100, range=[-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
                    plt.ylabel("Probability (a.u.)")
                    plt.xlabel('jet $\phi$')
                    plt.yscale('linear')
                    plt.legend(loc='lower right', prop={'size': 16})
                    plt.savefig(os.path.join(cur_report_dir,'jet_gen_phi_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                    plt.clf()

                    if vae_mode == "generation":

                        torch.save(model.state_dict(), os.path.join(cur_report_dir, 'model_'+ str(model_name) + '.pt'))

                        print('############ The minimum emdg sum for ',latent_dim,' latent vector dimensions is ',min_emdg,' ############')

                        inppx, bins, _ = plt.hist(input_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpx = plt.hist(output_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpx = plt.hist(gen_cart[:,0], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle px (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_px_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                        inppy, bins, _ = plt.hist(input_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpy = plt.hist(output_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpy = plt.hist(gen_cart[:,1], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle py (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_py_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                        inppz, bins, _ = plt.hist(input_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpz = plt.hist(output_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpz = plt.hist(gen_cart[:,2], bins=100, range = [-400, 400], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle pz (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_pz_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                        inppt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outpt = plt.hist(output_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 1500], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle pt (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_pt_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                        inplowpt, bins, _ = plt.hist(input_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outlowpt = plt.hist(output_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genlowpt = plt.hist(gen_hadr[:,0], bins=100, range = [0, 2], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle pt (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_low_pt_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                        inpeta, bins, _ = plt.hist(input_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outeta = plt.hist(output_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        geneta = plt.hist(gen_hadr[:,1], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle eta (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_eta_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                        inpphi, bins, _ = plt.hist(input_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
                        outphi = plt.hist(output_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Reco Test', color = spdblue,linewidth=1.5)
                        genphi = plt.hist(gen_hadr[:,2], bins=100, range = [-4, 4], histtype = 'step', density=False, label='Generated', color = spdyellow,linewidth=1.5)
                        plt.ylabel("Probability (a.u.)")
                        plt.xlabel('Particle phi (GeV)')
                        plt.yscale('log')
                        plt.legend(loc='upper right', prop={'size': 16})
                        plt.savefig(os.path.join(cur_report_dir,'part_phi_GeV'+str(model_name)+'.pdf'), format='pdf', bbox_inches='tight')
                        plt.clf()

                torch.save(output_tensor_emdt, os.path.join(cur_model_dir, 'emdt'+str(model_name)+'.pt'))
                torch.save(output_tensor_emdg, os.path.join(cur_model_dir, 'emdg'+str(model_name)+'.pt'))

    end_time = time.time()
    print("The total time is ",((end_time-start_time)/60.0)," minutes.")

if __name__=='__main__':
    main()
