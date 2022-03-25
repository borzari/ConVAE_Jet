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
import optuna
from optuna.trial import TrialState

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="LSTM implementation ")

    # Dataset setting
    parser.add_argument('--config', type=str, default='config/config_opt.json', help='Configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--load', type=str, help='this param load model')

    # parse the arguments
    args = parser.parse_args()

    return args

def objective(trial):
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
    num_features = configs['training']['num_features']
    batch_size = trial.suggest_int('batch_size', configs['training']['batch_size_min'],configs['training']['batch_size_max'])
    learning_rate = trial.suggest_float("learning_rate", configs['training']['learning_rate_min'], configs['training']['learning_rate_max'], log=True)
    saving_epoch = configs['training']['saving_epoch']
    n_filter = configs['training']['n_filter']
    n_classes = configs['training']['n_classes']
    latent_dim_seq = [configs['training']['latent_dim_seq']]
    beta = trial.suggest_float("beta", configs['training']['beta_min'], configs['training']['beta_max'])
    gamma = trial.suggest_float("gamma", configs['training']['gamma_min'], configs['training']['gamma_max'])
    gamma_1 = trial.suggest_float("gamma_1", configs['training']['gamma_1_min'], configs['training']['gamma_1_max'])
    gamma_2 = trial.suggest_float("gamma_2", configs['training']['gamma_2_min'], configs['training']['gamma_2_max'])
    #gamma_2 = 1.0
    n = 0 # this is to count the epochs to turn on/off the jet pt contribution to the loss
    min_emd = 0.

    # Particle features loss weighting
    alpha = trial.suggest_float("alpha", configs['training']['alpha_min'], configs['training']['alpha_max'])
    # Starting time
    start_time = time.time()

    # Probability to keep a node in the dropout layer
    drop_prob = trial.suggest_float("drop_prob", configs['training']['drop_prob_min'], configs['training']['drop_prob_max'])

    # Set patience for Early Stopping
    patience = n_epochs

    seed = configs['training']['seed']

    # Define max_accuracy and norm_emdg for each experiment with different hyperparameter values
    max_accuracy = 0.0
    norm_emdg = 1.0

    dataT = DataT(trial)
    print("tr_max depois da instância: ",dataT.tr_max)

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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        ###################################### TRAINING #######################################
        # Initialize model and load it on GPU
        model = ConvNet(configs, dataT.tr_max, dataT.tr_min, trial)
        #model = model.cuda()
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

        min_loss, stale_epochs, min_emdt, min_emdg = 999999.0, 0, 9999999.0, np.pinf

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
                tr_inputs, tr_outputs, tr_loss, tr_kl, tr_eucl, tr_reco_p, tr_reco_j, tr_reco_pt, tr_rec_mass  = train(model, jets_train, optimizer)
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
                val_loss, val_kl, val_eucl = validate(model, jets_valid)
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
                #print('stale_epochs:', stale_epochs)

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

                ####################################### EVALUATION #######################################
                all_input_test = np.empty(shape=(0, 1, num_features, num_particles))
                all_output_test = np.empty(shape=(0, 1, num_features, num_particles))

                for i, (jets) in enumerate(test_loader):

                    if i == (len(test_loader)-1):
                        break
                    # run test function on batch data for testing
                    test_inputs, test_outputs, ts_loss, ts_kl, ts_eucl = test_unseed_data(model, jets)
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

                def inverse_standardize_t(X, tmin, tmax):
                    mean = tmin
                    std = tmax
                    original_X = ((X * (std - mean)) + mean)
        #            original_X = ((X * std) + mean)
                    return original_X

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

                # Input constraints
                def mask_zero_padding(input_data):
                    # Mask input for zero-padded particles. Set to zero values between -10^-4 and 10^-4
                    px = input_data[:,0,:]
                    py = input_data[:,1,:]
                    pz = input_data[:,2,:]
                    mask_px = ((px <= -0.0001) | (px >= 0.0001))
                    mask_py = ((py <= -0.0001) | (py >= 0.0001))
                    mask_pz = ((pz <= -0.0001) | (pz >= 0.0001))
                    masked_px = (px * mask_px) + 0.0
                    masked_py = (py * mask_py) + 0.0
                    masked_pz = (pz * mask_pz) + 0.0
                    data = torch.stack([masked_px, masked_py, masked_pz], dim=1)
                    return data

                inputs = torch.stack([px_r, py_r, pz_r], dim=1)
                masked_inputs = mask_zero_padding(inputs)

                # Test data
                outputs_0 = torch.stack([px_reco_r_0, py_reco_r_0, pz_reco_r_0], dim=1)
                masked_outputs_0 = mask_zero_padding(outputs_0) # Now, values that correspond to the min-pt should be zeroed.

                # Gen data
                gen_outputs_0 = torch.stack([px_gen_r_0, py_gen_r_0, pz_gen_r_0], dim=1)
                masked_gen_outputs_0 = mask_zero_padding(gen_outputs_0)

                ######################################################################################
                # Output constraints
                def mask_min_pt(output_data):
                    # Mask output for min-pt
                    min_pt_cut = 0.25
                    mask =  output_data[:,0,:] * output_data[:,0,:] + output_data[:,1,:] * output_data[:,1,:] > min_pt_cut**2
                    # Expand over the features' dimension
                    mask = mask.unsqueeze(1)
                    # Then, you can apply the mask
                    data_masked = mask * output_data
                    return data_masked

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

                def compute_eta(pz, pt):
                    eta = np.nan_to_num(np.arcsinh(pz/pt))
                    return eta
                def compute_phi(px, py):
                    phi = np.arctan2(py, px)
                    return phi
                def particle_pT(p_part): # input of shape [n_jets, 3_features, n_particles]
                    p_px = p_part[:, :, 0]
                    p_py = p_part[:, :, 1]
                    p_pt = np.sqrt(p_px*p_px + p_py*p_py)
                    return p_pt
                def ptetaphim_particles(in_dataset):
                    part_pt = particle_pT(in_dataset)
                    part_eta = compute_eta(in_dataset[:,:,2],part_pt)
                    part_phi = compute_phi(in_dataset[:,:,0],in_dataset[:,:,1])
                    part_mass = in_dataset[:,:,3]
                    return np.stack((part_pt, part_eta, part_phi, part_mass), axis=2)

                hadr_input_data = ptetaphim_particles(input_data)
                hadr_output_data = ptetaphim_particles(output_data_0)
                hadr_gen_output_data = ptetaphim_particles(gen_output_data_0)

                def jet_features(jets, mask_bool=False, mask=None):
                    vecs = ak.zip({
                            "pt": jets[:, :, 0],
                            "eta": jets[:, :, 1],
                            "phi": jets[:, :, 2],
                            "mass": jets[:, :, 3],
                            }, with_name="PtEtaPhiMLorentzVector")

                    sum_vecs = vecs.sum(axis=1)

                    jf = np.stack((ak.to_numpy(sum_vecs.mass), ak.to_numpy(sum_vecs.pt), ak.to_numpy(sum_vecs.energy), ak.to_numpy(sum_vecs.eta), ak.to_numpy(sum_vecs.phi)), axis=1)

                    return ak.to_numpy(jf)

                jets_input_data = jet_features(hadr_input_data)
                jets_output_data = jet_features(hadr_output_data)
                jets_gen_output_data = jet_features(hadr_gen_output_data)

                minp, bins = np.histogram(jets_input_data[:,0], bins=100, range=[0,400])
                mout, bins = np.histogram(jets_output_data[:,0], bins=100, range=[0,400])
                mgen, bins = np.histogram(jets_gen_output_data[:,0], bins=100, range=[0,400])

                ptinp, bins = np.histogram(jets_input_data[:,1], bins=100, range=[0,3000])
                ptout, bins = np.histogram(jets_output_data[:,1], bins=100, range=[0,3000])
                ptgen, bins = np.histogram(jets_gen_output_data[:,1], bins=100, range=[0,3000])

                einp, bins = np.histogram(jets_input_data[:,2], bins=100, range=[200,4000])
                eout, bins = np.histogram(jets_output_data[:,2], bins=100, range=[200,4000])
                egen, bins = np.histogram(jets_gen_output_data[:,2], bins=100, range=[200,4000])

                etainp, bins = np.histogram(jets_input_data[:,3], bins=100, range=[-3,3])
                etaout, bins = np.histogram(jets_output_data[:,3], bins=100, range=[-3,3])
                etagen, bins = np.histogram(jets_gen_output_data[:,3], bins=100, range=[-3,3])

                phiinp, bins = np.histogram(jets_input_data[:,4], bins=100, range=[-3,3])
                phiout, bins = np.histogram(jets_output_data[:,4], bins=100, range=[-3,3])
                phigen, bins = np.histogram(jets_gen_output_data[:,4], bins=100, range=[-3,3])

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
                emdg_sum = float(emdg_m + emdg_pt + emdg_e + emdg_eta + emdg_phi)

                #print(emdg_sum,norm_emdg)

                if emdg_sum == np.nan:
                    norm_emdg = 1.0
                else:
                    norm_emdg = emdg_sum

                if norm_emdg < min_emdg: min_emdg = norm_emdg

                print("################################")
                print("Current EMD:",norm_emdg)
                print("Min. EMD:   ",min_emdg)
                print("################################")

                '''
                accuracy = 1 - (emdg_sum/norm_emdg)

                if accuracy >= max_accuracy:
                    max_accuracy = accuracy

                print("The accuracy is {:.4f} and the max_accuracy is {:.4f}. The emd is {:.4f} and the norm_emd is {:.4f}".format(accuracy,max_accuracy,emdg_sum,norm_emdg))

                ##### send accuracy from current epoch to optimizer
                trial.report(accuracy, epoch)
                '''
                ##### send accuracy from current epoch to optimizer
                trial.report(norm_emdg, epoch)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    end_time = time.time()
    print("The total time is ",((end_time-start_time)/60.0)," minutes.")
    print("###########################################################")
    print()

    #return max_accuracy
    return min_emdg

def main():
    study = optuna.create_study(
    study_name='opt_convae_v2',
    storage='mysql://usr_optuna:sY8d%5kq@top01/db_optuna',
    load_if_exists=True,
    direction="minimize")
    study.optimize(objective, n_trials=None, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__=='__main__':
    main()
