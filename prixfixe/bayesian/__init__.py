import numpy as np
import re

from .final_layers_block import BayesianFinalLayersBlock
from .trainer import BayesianTrainer, setup_pyro_model

__all__ = ["BayesianFinalLayersBlock", "BayesianTrainer", "setup_pyro_model", "get_jaspar_motifs", "motif_dict_to_array"]

def read_meme_file(filepath):
    """
    Read meme file from https://github.com/jvierstra/motif-clustering/ project.
    Copied from https://github.com/jvierstra/motif-clustering/.

    Parameters
    ----------
    filepath

    Returns
    -------

    """
    pwms = {}

    data = open(filepath).read()

    pos = 0
    while 1:
        rec_loc = data.find("\nMOTIF", pos)

        if rec_loc < 0:
            break

        nl = data.find("\n", rec_loc + 1)

        motif = data[rec_loc + 6 : nl].strip()
        # print motif

        mat_header_start = data.find("letter-probability", rec_loc)
        mat_header_end = data.find("\n", mat_header_start + 1)

        i = data.find(":", mat_header_start)
        kp = data[i + 1 : mat_header_end]
        r = re.compile(r"(\w+)=\s?([\w.\+]+)\s?")
        attribs = dict(r.findall(kp))

        ist = mat_header_end + 1
        iend = -1

        j = 0
        z = []
        while j < int(attribs["w"]):
            iend = data.find("\n", ist)
            z.append(list(map(float, data[ist:iend].strip().split())))
            ist = iend + 1
            j += 1

        pwms[motif] = np.array(z)

        pos = ist

    return pwms


def motif_ppm_to_pwm(
    motif,
    pseudocount=1e-3,
    background_counts=[0.25, 0.25, 0.25, 0.25],
):
    r"""
    Converts numpy array of PPMs to numpy array of PWMs
    """
    return np.log2((motif + pseudocount) / np.array(background_counts))


def motif_dict_ppm_to_pwm(
    motif_dict,
    **kwargs,
):
    r"""
    Converts numpy array of PPMs to numpy array of PWMs
    """
    return {k: motif_ppm_to_pwm(v, **kwargs) for k, v in motif_dict.items()}

def get_jaspar_motifs(fixed_motifs_path, genome):
    jaspar = read_meme_file(fixed_motifs_path)

    jaspar_unique = dict()
    jaspar_names = list()
    from re import sub

    for k in jaspar.keys():
        k_unique = sub("^.+ ", "", k)
        if genome == "mm10":
            k_unique = k_unique.lower().capitalize()
        elif genome == "hg38":
            k_unique = k_unique.upper()
        else:
            k_unique = k_unique.lower().capitalize()
        if k_unique not in jaspar_names:
            jaspar_unique[k_unique] = jaspar[k]
            jaspar_names = jaspar_names + [k_unique]
    jaspar_unique = motif_dict_ppm_to_pwm(jaspar_unique)
    # get rid of motifs for TF pairs
    jaspar_unique_single = {k: v for k, v in jaspar_unique.items() if ":" not in k}
    # make similar to MotifCentral (max nucleotide = 1)
    # CISBP motifs need to be renormalized to have 0 value for the top nucleotide per position
    jaspar_unique_single = {
        tf: (motif.T - motif.max(1)).T for tf, motif in jaspar_unique_single.items()
    }
    return jaspar_unique_single


def motif_dict_to_array(motif_dict, motif_length=19) -> np.ndarray:
    r"""
    Converts motif dictionary to numpy array of PPMs [n_motifs, motif_length, 4].
    """
    n_motifs = len(motif_dict)
    motif_tensor = np.zeros((n_motifs, motif_length, 4))
    for idx, motif_name in enumerate(motif_dict.keys()):
        motif = motif_dict[motif_name]
        curr_motif_length = motif.shape[0]
        if curr_motif_length < motif_length:
            # if provided motif is shorter than requested -> add padding
            expand_start = round(((motif_length - curr_motif_length) / 2) + 0.1)
            expand_end = expand_start
            if (motif_length - curr_motif_length) / 2 % 1 != 0:
                expand_end = expand_end - 1
            pad_start = -motif_ppm_to_pwm(0.25 * np.ones((expand_start, 4)))
            pad_end = -motif_ppm_to_pwm(0.25 * np.ones((expand_end, 4)))
            motif_tensor[idx, :, :] = np.concatenate((pad_start, motif, pad_end))
        elif curr_motif_length > motif_length:
            # if provided motif is longer than requested -> crop
            cut_start = round(((curr_motif_length - motif_length) / 2) + 0.1)
            cut_end = cut_start
            if (curr_motif_length - motif_length) / 2 % 1 != 0:
                cut_start = cut_start - 1
            motif_tensor[idx, :, :] = motif[cut_start : motif.shape[0] - cut_end, :]
        else:  # same length
            motif_tensor[idx, :, :] = motif
    return motif_tensor