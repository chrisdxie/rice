import sys, os
import numpy as np
import cv2

from . import constants
from .util import utilities as util_
from .util import munkres


# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def boundary_overlap(predicted_mask, gt_mask, bound_th=0.003):
    """Compute IoU of boundaries of GT/predicted mask, using dilated GT boundary.

    Args:
        predicted_mask  (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        
    Returns:
        overlap (float): IoU overlap of boundaries
    """
    assert np.atleast_3d(predicted_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(predicted_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = util_.seg2bmap(predicted_mask);
    gt_boundary = util_.seg2bmap(gt_mask);

    from skimage.morphology import disk

    # Dilate segmentation boundaries
    bp = disk(bound_pix)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), bp, iterations=1)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), bp, iterations=1)

    # Get the intersection (true positives). Calculate true positives differently for
    #   precision and recall since we have to dilated the boundaries
    fg_match = np.logical_and(fg_boundary, gt_dil)
    gt_match = np.logical_and(gt_boundary, fg_dil)

    # Return precision_tps, recall_tps (tps = true positives)
    return np.sum(fg_match), np.sum(gt_match)

# This function is modeled off of P/R/F measure as described by Dave et al. (arXiv19)
def multilabel_metrics(prediction, gt, i, N, 
                       obj_detect_threshold=0.75,
                       compute_boundary_stuff=True,
                       verbose=False):
    """Compute Metrics.

    Compute F-Measure, Precision, Recall, F@.75 metrics. Compute them normally
        and the Object Size Normalized (OSN) variants as well.
        It computes these measures only of objects, not background (0) / table (1).

    Args:
        gt: a [H x W] numpy.ndarray with ground truth masks
        prediction: a [H x W] numpy.ndarray with predicted masks
        i: int. For parallel computing outputs
        N: int. For parallel computing outputs

    Returns:
        a dictionary with the metrics
    """
    if verbose:
        print(f"Running example {i}/{N}...")

    ### Compute F, TP matrices ###

    # Get unique OBJECT labels from GT and prediction
    labels_gt = np.unique(gt)
    labels_gt = labels_gt[~np.isin(labels_gt, [constants.BACKGROUND_LABEL, constants.TABLE_LABEL])]
    num_labels_gt = labels_gt.shape[0]

    labels_pred = np.unique(prediction)
    labels_pred = labels_pred[~np.isin(labels_pred, [constants.BACKGROUND_LABEL, constants.TABLE_LABEL])]
    num_labels_pred = labels_pred.shape[0]

    # F-measure, True Positives, Boundary stuff
    obj_F = np.zeros((num_labels_gt, num_labels_pred))
    obj_P = np.zeros((num_labels_gt, num_labels_pred))
    obj_R = np.zeros((num_labels_gt, num_labels_pred))
    obj_tps = np.zeros((num_labels_gt, num_labels_pred))
    if compute_boundary_stuff:
        bound_F = np.zeros((num_labels_gt, num_labels_pred))
        bound_P = np.zeros((num_labels_gt, num_labels_pred))
        bound_R = np.zeros((num_labels_gt, num_labels_pred))
        bound_tps = np.zeros((num_labels_gt, num_labels_pred, 2)) 
        # Each item of "coundary_stuff" contains: precision_tps, recall_tps

    # Some edge case stuff
    # Edge cases are similar to here: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
    if (num_labels_pred == 0 and num_labels_gt > 0 ): # all false negatives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 1.,
                'Objects Recall' : 0.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 0.,
                'Objects OSN F-measure' : 0.,
                'Objects OSN Precision' : 1.,
                'Objects OSN Recall' : 0.,
                'Boundary OSN F-measure' : 0.,
                'Boundary OSN Precision' : 1.,
                'Boundary OSN Recall' : 0.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                'obj_detected_075_percentage_normalized' : 0.,
                }
    elif (num_labels_pred > 0 and num_labels_gt == 0 ): # all false positives
        return {'Objects F-measure' : 0.,
                'Objects Precision' : 0.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 0.,
                'Boundary Precision' : 0.,
                'Boundary Recall' : 1.,
                'Objects OSN F-measure' : 0.,
                'Objects OSN Precision' : 0.,
                'Objects OSN Recall' : 1.,
                'Boundary OSN F-measure' : 0.,
                'Boundary OSN Precision' : 0.,
                'Boundary OSN Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 0.,
                'obj_detected_075_percentage_normalized' : 0.,
                }
    elif (num_labels_pred == 0 and num_labels_gt == 0 ): # correctly predicted nothing
        return {'Objects F-measure' : 1.,
                'Objects Precision' : 1.,
                'Objects Recall' : 1.,
                'Boundary F-measure' : 1.,
                'Boundary Precision' : 1.,
                'Boundary Recall' : 1.,
                'Objects OSN F-measure' : 1.,
                'Objects OSN Precision' : 1.,
                'Objects OSN Recall' : 1.,
                'Boundary OSN F-measure' : 1.,
                'Boundary OSN Precision' : 1.,
                'Boundary OSN Recall' : 1.,
                'obj_detected' : num_labels_pred,
                'obj_detected_075' : 0.,
                'obj_gt' : num_labels_gt,
                'obj_detected_075_percentage' : 1.,
                'obj_detected_075_percentage_normalized' : 1.,
                }

    if compute_boundary_stuff:
        # Preprocess boundary counts
        bound_counts_pred = np.zeros(num_labels_pred)
        for j, pred_j in enumerate(labels_pred):
            pred_mask = (prediction == pred_j)
            bound_counts_pred[j] = np.sum(util_.seg2bmap(pred_mask))
        bound_counts_gt = np.zeros(num_labels_gt)
        for i, gt_i in enumerate(labels_gt):
            gt_mask = (gt == gt_i)
            bound_counts_gt[i] = np.sum(util_.seg2bmap(gt_mask))

    # For every pair of GT label vs. predicted label, calculate stuff
    for i, gt_i in enumerate(labels_gt):

        gt_i_mask = (gt == gt_i)

        for j, pred_j in enumerate(labels_pred):
            
            pred_j_mask = (prediction == pred_j)
            
            ### Overlap Stuff ###
            A = np.logical_and(pred_j_mask, gt_i_mask)
            obj_tp = np.int64(np.count_nonzero(A)) # Cast this to numpy.int64 so 0/0 = nan
            obj_tps[i,j] = obj_tp 
            
            obj_P[i,j] = obj_tp/np.count_nonzero(pred_j_mask)
            obj_R[i,j] = obj_tp/np.count_nonzero(gt_i_mask)
            obj_F[i,j] = (2 * obj_P[i,j] * obj_R[i,j]) / (obj_P[i,j] + obj_R[i,j])

            ### Boundary Stuff ###
            if compute_boundary_stuff:
                bound_tps[i,j] = boundary_overlap(pred_j_mask, gt_i_mask)

                bound_P[i,j] = bound_tps[i,j][0] / bound_counts_pred[j]
                bound_R[i,j] = bound_tps[i,j][1] / bound_counts_gt[i]
                bound_F[i,j] = (2 * bound_P[i,j] * bound_R[i,j]) / (bound_P[i,j] + bound_R[i,j])


    ### Compute the Hungarian assignment ###
    obj_F[np.isnan(obj_F)] = 0
    m = munkres.Munkres()
    assignments = m.compute(obj_F.max() - obj_F.copy()) # list of (y,x) indices into F (these are the matchings)
    idx = tuple(np.array(assignments).T)

    ### Compute the number of "detected objects" ###
    num_obj_detected = 0
    for a in assignments:
        if obj_F[a] > obj_detect_threshold:
            num_obj_detected += 1

    ### Compute Overlap measures ###
    precision = np.sum(obj_tps[idx]) / np.sum(prediction.clip(0,2) == constants.OBJECTS_LABEL)
    recall = np.sum(obj_tps[idx]) / np.sum(gt.clip(0,2) == constants.OBJECTS_LABEL)
    F_measure = (2 * precision * recall) / (precision + recall)
    if np.isnan(F_measure): # b/c precision = recall = 0
        F_measure = 0

    # Object Size Normalized measures
    obj_F_osn = np.sum(obj_F[idx])/max(num_labels_pred,num_labels_gt)
    obj_P_osn = np.sum(obj_P[idx])/num_labels_pred
    obj_R_osn = np.sum(obj_R[idx])/num_labels_gt

    ### Compute Boundary measures ###
    if compute_boundary_stuff:
        bound_F[np.isnan(bound_F)] = 0
        boundary_precision = np.sum(bound_tps[idx][:,0]) / np.sum(bound_counts_pred)
        boundary_recall = np.sum(bound_tps[idx][:,1]) / np.sum(bound_counts_gt)
        boundary_F_measure = (2 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
        if np.isnan(boundary_F_measure): # b/c/ precision = recall = 0
            boundary_F_measure = 0

        # Object Size Normalize measures
        bound_F_osn = np.sum(bound_F[idx])/max(num_labels_pred,num_labels_gt)
        bound_P_osn = np.sum(bound_P[idx])/num_labels_pred
        bound_R_osn = np.sum(bound_R[idx])/num_labels_gt
    else:
        boundary_F_measure = None
        boundary_precision = None
        boundary_recall = None
        bound_F_osn = None
        bound_P_osn = None
        bound_R_osn = None


    return {'Objects F-measure' : F_measure,
            'Objects Precision' : precision,
            'Objects Recall' : recall,
            'Boundary F-measure' : boundary_F_measure,
            'Boundary Precision' : boundary_precision,
            'Boundary Recall' : boundary_recall,
            'Objects OSN F-measure' : obj_F_osn,
            'Objects OSN Precision' : obj_P_osn,
            'Objects OSN Recall' : obj_R_osn,
            'Boundary OSN F-measure' : bound_F_osn,
            'Boundary OSN Precision' : bound_P_osn,
            'Boundary OSN Recall' : bound_R_osn,
            'obj_detected' : num_labels_pred,
            'obj_detected_075' : num_obj_detected,
            'obj_gt' : num_labels_gt,
            'obj_detected_075_percentage' : num_obj_detected / num_labels_gt,
            'obj_detected_075_percentage_normalized' : num_obj_detected / max(num_labels_gt, num_labels_pred),
            }

