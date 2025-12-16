import numpy as np
import h5py 

def get_synthesis_dataset_both(phase="val", dataset="ixi", domain_1="T1", domain_2="T2", norm=False):

    target_file="datasets/" + dataset + "/" + domain_1 + "_" + "4_multi_synth_recon_" +str(phase)+'.mat'
    f = h5py.File(target_file,'r')
    data_fs_t1=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs_t1=data_fs_t1.astype(np.float32) 
    
    target_file="datasets/" + dataset + "/" + domain_2 + "_"  + "4_multi_synth_recon_" +str(phase)+'.mat'
    f = h5py.File(target_file,'r')
    data_fs_t2=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs_t2=data_fs_t2.astype(np.float32) 

    data_fs_t1 = np.squeeze(data_fs_t1)
    data_fs_t2 = np.squeeze(data_fs_t2)
    
    return data_fs_t1, data_fs_t2



def get_single_domain_dataset(phase="train", contrast="T1", dataset = "ixi", norm=False):
    if dataset == "ct_to_t1" or dataset == "ct_to_t2":
        target_file = "datasets/" + dataset + "/" + "train" + "/" + contrast + "/" + contrast + ".npy"
        data = np.load(target_file)[:,np.newaxis,:,:]
        
        label = np.tile(np.asarray("this is pelvis " + contrast)[np.newaxis], [data.shape[0],1])

        if contrast == "ct":
            ys = np.zeros(shape=data.shape[0],dtype=np.int64)
        elif contrast == "mri":
            ys = np.ones(shape=data.shape[0],dtype=np.int64)
        else:
            ValueError("undefined contrast")
    else:
        target_file = "datasets/" + dataset + "/"  + contrast + "_" + "4_multi_synth_recon_" +str(phase)+'.mat'
        f = h5py.File(target_file,'r')
        data=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
        data=data.astype(np.float32) 

        label = np.tile(np.asarray("this MRI is " + contrast + "-weighted")[np.newaxis], [data.shape[0],1])

        if contrast == "T1":
            ys = np.zeros(shape=data.shape[0],dtype=np.int64)
        elif contrast == "T2":
            ys = np.ones(shape=data.shape[0],dtype=np.int64)
        else:
            ys = np.ones(shape=data.shape[0],dtype=np.int64)*2

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    label = label[indices]
    ys = ys[indices]

    if norm:
        data = (data-0.5)/0.5

    return data, label, ys

def get_multi_domain_dataset(phase="train", dataset="ixi", norm=False):

    target_file = "datasets/" + dataset + "/"  + "T1" + "_" + "4_multi_synth_recon_" +str(phase)+'.mat'
    f = h5py.File(target_file,'r')
    data_t1=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_t1=data_t1.astype(np.float32)     

    target_file = "datasets/" + dataset + "/"  + "T2" + "_" + "4_multi_synth_recon_" +str(phase)+'.mat'
    f = h5py.File(target_file,'r')
    data_t2=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_t2=data_t2.astype(np.float32)     

    target_file = "datasets/" + dataset + "/"  + "PD" + "_" + "4_multi_synth_recon_" +str(phase)+'.mat'
    f = h5py.File(target_file,'r')
    data_pd=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_pd=data_pd.astype(np.float32)     

    label_t1 = np.tile(np.asarray("this MRI is T1-weighted")[np.newaxis], [data_t1.shape[0],1])
    label_t2 = np.tile(np.asarray("this MRI is T2-weighted")[np.newaxis], [data_t2.shape[0],1])
    label_pd = np.tile(np.asarray("this MRI is PD-weighted")[np.newaxis], [data_pd.shape[0],1])
    
    y_t1 = np.zeros(shape=data_t1.shape[0],dtype=np.int64)
    y_t2 = np.ones(shape=data_t1.shape[0],dtype=np.int64)
    y_pd = np.ones(shape=data_t1.shape[0],dtype=np.int64)*2

    data = np.concatenate([data_t1, data_t2, data_pd])
    labels = np.concatenate([label_t1, label_t2, label_pd])
    ys = np.concatenate([y_t1, y_t2, y_pd])

    indices = np.arange(data.shape[0])

    np.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices]
    ys = ys[indices]


    if norm:
        data = (data-0.5)/0.5

    return data, labels, ys