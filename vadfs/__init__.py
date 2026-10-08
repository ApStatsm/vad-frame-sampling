"""VAD Frame Sampling research code: shared pipeline used by scripts/.

Modules
    paths    project root, path resolution and file hashing
    runs     run directories, provenance and summary writing
    ucf      UCF-Crime labels and fixed input locations
    vadclip  fixed pretrained VadCLIP loading and upstream-equivalent inference
    color    x86 libswscale YUV420p -> RGB emulation
    extract  raw-video reference extraction (decode, select, crop, CLIP encode)
    metrics  frame-level AUC/AP/Ano-AUC, original fine-grained mAP, run comparison
"""
