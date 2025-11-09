# Privacy-Preserving Traffic Surveillance

A video processing pipeline that automatically detects and obscures identifying vehicle features in surveillance footage while keeping the overall scene intact and useful.

## What's This About?

Traffic cameras catch everything—bumper stickers, custom paint jobs, dents, basically anything that makes a car recognizable. We built this to automatically find those identifying details and tone them down using AI, so surveillance footage can still be useful without being creepy.

The system watches videos, figures out which parts of vehicles are most memorable, and then subtly modifies those regions. It does this smartly across frames so you don't get weird flickering or inconsistencies.

## How It Works

**Step 1: Detection**  
We use YOLOv8 to spot 21 different car parts—hoods, bumpers, doors, mirrors, headlights, you name it.

**Step 2: Tracking**  
DeepSORT keeps tabs on these parts as they move through the video, giving each one a persistent ID.

**Step 3: Memorability Scoring**  
This is where AMNet comes in. It's a neural network that's pretty good at predicting which visual elements stick in your memory. We run it on each detected region and get a score.

**Step 4: Selective Editing**  
For regions that score high on memorability, we use Stable Diffusion's inpainting model to replace them with generic, bland versions. Think of it like Photoshop's content-aware fill, but for video.

**Step 5: Consistency Check**  
Here's the clever bit—we do a two-pass process. First pass analyzes everything, second pass applies edits using a cache so the same car part looks the same across all frames. No flickering weirdness.

## The Tech Stack

```
┌─────────────────────────────────────────────────┐
│  Input: Traffic Surveillance Video              │
└─────────────────┬───────────────────────────────┘
                  │
          ┌───────▼────────┐
          │  YOLOv8 Object │
          │    Detection   │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │    DeepSORT    │
          │    Tracking    │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │  AMNet Memory  │
          │     Scoring    │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │    Diffusion   │
          │   Inpainting   │
          └───────┬────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  Output: Privacy-Protected Video                │
└─────────────────────────────────────────────────┘
```

## About AMNet

We're using AMNet (Attention-based Memorability Network) from a 2018 CVPR paper by Fajtl et al. The original work focused on predicting how memorable images are. We've adapted it to work on vehicle regions in real-time video.

Our tweaks:
- Made it work region-by-region instead of whole images
- Fine-tuned it on vehicle data (got Spearman correlation around 0.56, which is decent)
- Hooked up the attention maps so you can see what it's focusing on
- Optimized for batch processing so it doesn't take forever

## File Structure

**object_detector.py**  
YOLOv8 wrapper that finds car parts. Can process single frames or batches, gives you nice visualizations, and spits out detection stats.

**object_tracker.py**  
DeepSORT implementation. Assigns IDs to detected objects and follows them through the video. Draws colored boxes and motion trails for debugging.

**amnet_integration.py**  
The memorability brain. Takes detection crops, runs them through AMNet, and decides what needs editing. Also handles the edit cache to keep things consistent.

**diffusion_editor.py**  
Stable Diffusion 2 inpainting wrapper. Generates bland, logo-free replacements for high-memorability regions. Falls back to Gaussian blur if something breaks.

**temporal_video_pipeline.py**  
The orchestrator. Runs the two-pass process, manages output files, and ties everything together.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/AMNet-Memorability-Privacy.git
cd AMNet-Memorability-Privacy

# Set up environment
conda create -n privacy_pipeline python=3.10
conda activate privacy_pipeline
pip install -r requirements.txt
```

### What You Need

- Python 3.8 or newer
- PyTorch 2.0+ (with CUDA if you want speed)
- OpenCV, NumPy, Pillow
- Ultralytics (for YOLOv8)
- deep-sort-realtime
- diffusers, transformers, accelerate
- tqdm (because progress bars are nice)

## Running It

Quick memorability test on a single image:
```bash
python amnet_integration.py
```

Full pipeline on a video:
```bash
python temporal_video_pipeline.py --video path/to/traffic.mp4 --device cuda
```

### Command Options

| Flag | What It Does | Default |
|------|--------------|---------|
| `--video` | Path to input video | Required |
| `--threshold` | Memorability cutoff (0-1) | 0.6 |
| `--frames` | Max frames to process | All |
| `--device` | Use `cpu` or `cuda` | cuda |
| `--no-comparison` | Skip side-by-side output | Off |
| `--no-metadata` | Don't save tracking JSON | Off |

### What You Get

After processing, check the `results/` folder:

```
results/
├── yourname_temporal_edited.mp4    # The privacy-protected version
├── yourname_comparison.mp4         # Original vs edited side-by-side
├── yourname_metadata.json          # All the tracking and score data
└── memorability_test_*.jpg         # Test outputs if you ran standalone tests
```

## Tweaking It

**Use your own detector**  
Swap out `models/best.pt` with your own YOLOv8 weights if you've trained on different objects or regions.

**Change the inpainting style**  
Edit `DiffusionEditor.CLASS_PROMPT_MAP` in diffusion_editor.py to customize how replaced regions look.

**Adjust privacy level**  
Lower `--threshold` to be more aggressive (edit more stuff), raise it to be conservative (only edit really memorable things).

## How Well Does It Work?

The memorability network correlates decently with human judgments (ρ ≈ 0.56). In practice, it catches most obvious identifiers like bumper stickers and custom decals. Sometimes it misses subtle stuff, sometimes it flags boring things. It's not perfect, but it's pretty good.

Processing speed depends on your hardware. On a decent GPU (RTX 3090), expect around 5-10 fps for the full pipeline with diffusion inpainting. CPU-only is much slower.

## Credits

This builds on some excellent prior work:

**AMNet** - Fajtl, J., Argyriou, V., Monekosso, D., & Remagnino, P. (2018). AMNet: Memorability Estimation with Attention. CVPR 2018.

**YOLOv8** - [Ultralytics](https://github.com/ultralytics/ultralytics) for the detection framework

**DeepSORT** - [Deep SORT Realtime](https://github.com/levan92/deep_sort_realtime) for tracking

**Stable Diffusion** - [Stability AI](https://stability.ai/) for the inpainting model

If you use this work, please cite the original AMNet paper:
```bibtex
@inproceedings{fajtl2018amnet,
  title={AMNet: Memorability Estimation with Attention},
  author={Fajtl, Jiri and Argyriou, Vasileios and Monekosso, Dorothy and Remagnino, Paolo},
  booktitle={CVPR},
  pages={6363--6372},
  year={2018}
}
```

## Issues?

If something breaks, check:
- GPU memory (diffusion models are hungry)
- OpenCV video codec compatibility
- Model file paths (especially models/best.pt)
- CUDA version matching PyTorch

Still stuck? Open an issue with your error log and we'll take a look.

## License

Check LICENSE file for details. Third-party models (YOLOv8, Stable Diffusion, AMNet) have their own licenses.

---

Built because surveillance doesn't have to mean zero privacy.