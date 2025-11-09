__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '9.5'
__status__ = "Research"
__date__ = "30/1/2018"
__license__= "MIT License"

import os
import numpy as np
import torch
from torch.autograd import Variable
import pkg_resources


# ============================================================================
# Memorability Thresholds and Categories
# ============================================================================

# Memorability classification thresholds
MEMORABILITY_THRESHOLD_HIGH = 0.6  # Threshold for high memorability (requires perturbation)
MEMORABILITY_THRESHOLD_VERY_HIGH = 0.85  # Threshold for very high memorability

# Memorability categories with ranges
MEMORABILITY_CATEGORIES = {
    'very_low': (0.0, 0.3),
    'low': (0.3, 0.5),
    'medium': (0.5, 0.7),
    'high': (0.7, 0.85),
    'very_high': (0.85, 1.0)
}


# ============================================================================
# Perturbation Methods Configuration
# ============================================================================

# Available perturbation methods
PERTURBATION_METHODS = {
    'sticker': 'neighbor',      # For stickers: use neighbor fill
    'bumper': 'neighbor',       # For bumpers: use neighbor fill
    'door': 'blur',            # For doors: use blur
    'hood': 'blur',            # For hood: use blur
    'windshield': 'neighbor',  # For windshield: use neighbor fill
    'lights': 'blur',          # For lights: use blur
    'default': 'neighbor'      # Default fallback method
}

# Perturbation blend ratios (how much of the perturbation to apply)
PERTURBATION_BLEND_RATIOS = {
    'sticker': 1.0,      # Full replacement for stickers
    'default': 0.7       # 70% blend for other parts
}


# ============================================================================
# Diffusion Model Configuration (Optional)
# ============================================================================

# Diffusion model settings
DIFFUSION_CONFIG = {
    'enabled': False,  # Set to True to enable diffusion (requires diffusers package)
    'model_id': 'stabilityai/stable-diffusion-2-inpainting',
    'num_inference_steps': 20,
    'guidance_scale': 7.5,
    'use_fp16': True  # Use float16 for faster inference (GPU only)
}

# Diffusion prompts for each class
DIFFUSION_PROMPTS = {
    'sticker': 'plain car surface, no text, no stickers, smooth paint',
    'bumper': 'generic car bumper, plain, no logos, standard design',
    'door': 'generic car door, plain surface, no distinctive features',
    'hood': 'generic car hood, smooth surface, standard paint',
    'windshield': 'generic car windshield, clear glass, no markings',
    'lights': 'generic car headlight, standard design, no custom elements',
    'default': 'generic car part, plain, no distinctive features'
}


# ============================================================================
# Attention Visualization Configuration
# ============================================================================

# Attention map settings
ATTENTION_CONFIG = {
    'enabled': True,  # Enable attention map generation
    'alpha': 0.6,     # Blend factor for attention overlay (0.0-1.0)
    'colormap': 'COLORMAP_JET',  # OpenCV colormap name
    'show_only_high_mem': True   # Only show attention for high memorability detections
}


# ============================================================================
# HParameters Class (Original)
# ============================================================================

class HParameters:

    def __init__(self):
        self.use_cuda = True
        self.cuda_device = 0

        self.use_attention = True
        self.last_step_prediction = False
        self.train_split = 'train_1'
        self.val_split = 'val_1'
        self.front_end_cnn = 'ResNet18FC'

        self.epoch_start = 0
        self.epoch_max = 100

        self.l2_req = 0.00001
        self.mem_loc_w = None
        self.seq_steps = 3

        self.lr_epochs = [0]
        self.lr = [1e-5]

        # alpha map cost weight
        # hps.gamma = 0.001
        self.gamma = 0.00001

        # memorability-location cost weight
        self.omega = 0

        self.train_batch_size = 128
        self.test_batch_size = 128

        self.torch_version_major, self.torch_version_minor = [int(v) for v in torch.__version__.split('.')[:2]]
        torchvision_version = pkg_resources.get_distribution("torchvision").version
        self.torchvision_version_major, self.torchvision_version_minor = [int(v) for v in torchvision_version.split('.')[:2]]

        return

    @property
    def seq_steps(self):
        return self._seq_steps

    @seq_steps.setter
    def seq_steps(self, value):
        self._seq_steps = value
        if value <= 0:
            return

        mem_loc_w = (0.1 ** (np.arange(0, self._seq_steps)))
        self.mem_loc_w = Variable(torch.from_numpy(np.array([mem_loc_w]))).float()

    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str


# ============================================================================
# Configuration Function (Original + Updated)
# ============================================================================

def get_amnet_config(args):

    hps = HParameters()

    hps.dataset_name = args.dataset
    hps.experiment_name = args.experiment
    hps.front_end_cnn = args.cnn
    hps.model_weights = args.model_weights
    hps.dataset_root = args.dataset_root
    hps.images_dir = args.images_dir
    hps.splits_dir = args.splits_dir
    hps.eval_images = args.eval_images
    hps.test_split = args.test_split
    hps.val_split = args.val_split
    hps.train_split = args.train_split
    hps.epoch_max = args.epoch_max
    hps.epoch_start = args.epoch_start
    hps.train_batch_size = args.train_batch_size
    hps.test_batch_size = args.test_batch_size

    # Default configuration
    hps.cuda_device = args.gpu
    hps.seq_steps = args.lstm_steps
    hps.last_step_prediction = args.last_step_prediction
    hps.use_attention = not args.att_off

    hps.use_cuda = hps.cuda_device > -1

    # Create experiment name
    if hps.experiment_name == '':
        hps.experiment_name = hps.dataset_name + '_' + hps.front_end_cnn
        hps.experiment_name += '_lstm' + str(hps.seq_steps)

        if hps.last_step_prediction:
            hps.experiment_name += '_last'

        if not hps.use_attention:
            hps.experiment_name += '_noatt'


#----------------------------------------------------------------------------------
# Dataset specific configurations

    if hps.dataset_name == 'lamem':

        if hps.front_end_cnn == '':
            hps.front_end_cnn = 'ResNet50FC'

        if hps.dataset_root == '':
            hps.dataset_root = 'datasets/lamem/'

        # Set default validation split filename
        if hps.val_split == '':
            if hps.train_split != '':
                hps.val_split = 'val_' + hps.train_split.split('_')[1]

        if hps.epoch_max < 0:
            hps.epoch_max = 55


        if hps.train_batch_size < 0:
            hps.train_batch_size = 222
            #hps.train_batch_size = 128

        if hps.test_batch_size < 0:
            hps.test_batch_size = 370

        hps.l2_req = 0.000001

        hps.target_mean = 0.754
        hps.target_scale = 2.0

        hps.img_mean = [0.485, 0.456, 0.406]
        hps.img_std = [0.229, 0.224, 0.225]



    elif hps.dataset_name == 'sun':
        # SUN memorability dataset

        if hps.val_split == '':
            if hps.train_split != '':
                hps.val_split = 'test_' + hps.train_split.split('_')[1]

        if hps.dataset_root == '':
            hps.dataset_root = 'datasets/SUN_memorability/'

        if hps.epoch_max < 0:
            hps.epoch_max = 50

        if hps.train_batch_size < 0:
            hps.train_batch_size = 222

        if hps.test_batch_size < 0:
            hps.test_batch_size = 370

        hps.l2_req = 0.0001

        # TODO: Should be updated for the SUN dataset!
        hps.target_mean = 0.754
        hps.target_scale = 2.0
        hps.img_mean = [0.485, 0.456, 0.406]
        hps.img_std = [0.229, 0.224, 0.225]



    elif hps.dataset_name == 'ava':
        # AVA image aesthetic dataset

        if hps.val_split == '':
            if hps.train_split != '':
                hps.val_split = 'test_' + hps.train_split.split('_')[1]

        if hps.dataset_root == '':
            hps.dataset_root = 'datasets/ava/'

        if hps.epoch_max < 0:
            hps.epoch_max = 150

        if hps.train_batch_size < 0:
            hps.train_batch_size = 370

        if hps.test_batch_size < 0:
            hps.test_batch_size = 370

        hps.l2_req = 0.000001
        hps.lr = [0.0001]

        hps.target_mean = 0.538388987454
        hps.target_scale = 2.0
        hps.img_mean = [0.485, 0.456, 0.406]
        hps.img_std = [0.229, 0.224, 0.225]

    elif hps.dataset_name == 'memcat_vehicle':
        # MemCat Vehicle dataset configuration
        
        if hps.val_split == '':
            if hps.train_split != '':
                hps.val_split = 'val_' + hps.train_split.split('_')[0]  # Adjust as needed

        if hps.epoch_max < 0:
            hps.epoch_max = 30  # Default epochs for fine-tuning

        if hps.train_batch_size < 0:
            hps.train_batch_size = 128

        if hps.test_batch_size < 0:
            hps.test_batch_size = 128

        hps.l2_req = 0.000001
        hps.lr = [1e-5]  # Fine-tuning learning rate

        # Use the same normalization as ImageNet/LaMem
        hps.target_mean = 0.754  # You may want to compute this for your dataset
        hps.target_scale = 2.0
        hps.img_mean = [0.485, 0.456, 0.406]  # Standard ImageNet mean
        hps.img_std = [0.229, 0.224, 0.225]   # Standard ImageNet std

    else:
        print("ERROR Unknown dataset:", hps.dataset_name)

    if hps.front_end_cnn == 'VGG16FC':
        hps.train_batch_size = 138
        hps.test_batch_size = 138

    if hps.train_split != '':
        hps.experiment_name += '_' + hps.train_split

    return hps


# ============================================================================
# Utility Functions for Memorability Integration
# ============================================================================

def get_memorability_category(score: float) -> str:
    """
    Get memorability category from score
    
    Args:
        score: Memorability score (0-1)
        
    Returns:
        Category name
    """
    for category, (low, high) in MEMORABILITY_CATEGORIES.items():
        if low <= score < high:
            return category
    return 'very_high'


def get_perturbation_strategy(class_name: str, mem_score: float) -> str:
    """
    Determine perturbation strategy based on class and memorability
    
    Args:
        class_name: Detected class name
        mem_score: Memorability score
        
    Returns:
        Strategy name: 'remove', 'genericize', 'subtle', or 'none'
    """
    # Always remove stickers
    if class_name.lower() == 'sticker':
        return 'remove'
    
    # Very high memorability - aggressive perturbation
    if mem_score > MEMORABILITY_THRESHOLD_VERY_HIGH:
        return 'genericize'
    
    # High memorability - moderate perturbation
    if mem_score > MEMORABILITY_THRESHOLD_HIGH:
        return 'subtle'
    
    # Medium and below - no perturbation needed
    return 'none'


def get_perturbation_method(class_name: str) -> str:
    """
    Get perturbation method for a specific class
    
    Args:
        class_name: Detected class name
        
    Returns:
        Method name: 'blur', 'neighbor', 'inpaint', or 'generic'
    """
    return PERTURBATION_METHODS.get(class_name.lower(), PERTURBATION_METHODS['default'])


def get_perturbation_blend_ratio(class_name: str) -> float:
    """
    Get blend ratio for perturbation
    
    Args:
        class_name: Detected class name
        
    Returns:
        Blend ratio (0.0-1.0)
    """
    return PERTURBATION_BLEND_RATIOS.get(class_name.lower(), PERTURBATION_BLEND_RATIOS['default'])


def get_diffusion_prompt(class_name: str) -> str:
    """
    Get diffusion inpainting prompt for a specific class
    
    Args:
        class_name: Detected class name
        
    Returns:
        Prompt string for diffusion model
    """
    return DIFFUSION_PROMPTS.get(class_name.lower(), DIFFUSION_PROMPTS['default'])


def is_diffusion_enabled() -> bool:
    """
    Check if diffusion-based perturbation is enabled
    
    Returns:
        True if diffusion is enabled and available
    """
    if not DIFFUSION_CONFIG['enabled']:
        return False
    
    try:
        import diffusers
        return True
    except ImportError:
        print("⚠ Warning: Diffusion enabled in config but 'diffusers' package not installed.")
        print("  Install with: pip install diffusers transformers accelerate")
        return False


def get_memorability_color(score: float) -> tuple:
    """
    Get BGR color for memorability score
    
    Args:
        score: Memorability score (0-1)
        
    Returns:
        BGR color tuple (b, g, r)
    """
    # Green (low mem) -> Yellow -> Red (high mem)
    if score < 0.5:
        # Green to yellow
        r = int(score * 2 * 255)
        g = 255
        b = 0
    else:
        # Yellow to red
        r = 255
        g = int((1 - (score - 0.5) * 2) * 255)
        b = 0
    
    return (b, g, r)  # BGR format for OpenCV


def get_config():
    """
    Get default configuration for standalone usage (without args)
    This is useful when using the memorability integration without training
    """
    hps = HParameters()
    
    # Set default values for standalone usage
    hps.dataset_name = 'lamem'
    hps.experiment_name = 'memorability_analysis'
    hps.front_end_cnn = 'ResNet50FC'
    hps.model_weights = ''
    hps.dataset_root = 'datasets/lamem/'
    hps.images_dir = 'images'
    hps.splits_dir = 'splits'
    
    # Default image normalization
    hps.target_mean = 0.754
    hps.target_scale = 2.0
    hps.img_mean = [0.485, 0.456, 0.406]
    hps.img_std = [0.229, 0.224, 0.225]
    
    return hps


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test the configuration
    print("Memorability Configuration Test")
    print("=" * 70)
    
    # Test categories
    print("\nMemorability Categories:")
    test_scores = [0.2, 0.45, 0.6, 0.75, 0.9]
    for score in test_scores:
        category = get_memorability_category(score)
        strategy = get_perturbation_strategy('bumper', score)
        method = get_perturbation_method('bumper')
        color = get_memorability_color(score)
        print(f"  Score: {score:.2f} -> {category:12s} | Strategy: {strategy:12s} | Method: {method:10s} | Color: {color}")
    
    # Test perturbation methods
    print("\nPerturbation Methods by Class:")
    test_classes = ['sticker', 'bumper', 'door', 'hood', 'lights', 'unknown_class']
    for cls in test_classes:
        method = get_perturbation_method(cls)
        ratio = get_perturbation_blend_ratio(cls)
        prompt = get_diffusion_prompt(cls)
        print(f"  {cls:15s} -> Method: {method:10s} | Ratio: {ratio:.2f} | Prompt: {prompt[:50]}...")
    
    # Test diffusion config
    print(f"\nDiffusion Enabled: {is_diffusion_enabled()}")
    print(f"Attention Enabled: {ATTENTION_CONFIG['enabled']}")
    
    # Test default config
    print("\nDefault Configuration:")
    hps = get_config()
    print(f"  Dataset: {hps.dataset_name}")
    print(f"  CNN: {hps.front_end_cnn}")
    print(f"  Attention: {hps.use_attention}")
    print(f"  CUDA: {hps.use_cuda}")
