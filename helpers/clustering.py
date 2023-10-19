import faiss
import torch
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as adjusted_nmi
from sklearn.metrics import adjusted_rand_score as adjusted_rand_index
from sklearn.metrics import fowlkes_mallows_score as fms
from sklearn.metrics import homogeneity_completeness_v_measure as hcm
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def kmeans_classifier(train_features, val_features, targets, f, args):
    # fit based on train set
    print('=> fitting K-Means classifier..')

    d = train_features.shape[1]
    niter = 100
    nredo = 10

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    
    kmeans = faiss.Kmeans(d, args.num_classes, update_index=False, spherical=True, nredo=nredo, niter=niter, verbose=True, gpu=True)
    kmeans.train(train_features)

    val_features = scaler.transform(val_features)
    D, preds = kmeans.index.search(val_features, 1)
    preds = preds.squeeze(1)

    # evaluate
    val_nmi = nmi(targets, preds)
    val_adjusted_nmi = adjusted_nmi(targets, preds)
    val_adjusted_rand_index = adjusted_rand_index(targets, preds)
    val_fms = fms(targets, preds)
    val_homogeneity, val_completeness, val_v_measure = hcm(targets, preds)

    print(f'=> number of samples: {len(targets)}')
    print(f'=> number of unique assignments: {len(set(preds))}')
    print("NMI\tANMI\tARI\tFMS\tHOMO\tCOMPL\tVM")
    print(f"{val_nmi * 100}\t{val_adjusted_nmi * 100}\t{val_adjusted_rand_index * 100}\t{val_fms * 100}\t{val_homogeneity * 100}\t{val_completeness * 100}\t{val_v_measure * 100}")

    return val_nmi, val_adjusted_nmi, val_adjusted_rand_index, val_fms, val_homogeneity, val_completeness, val_v_measure