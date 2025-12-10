print("Testing diffusers import...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    print(f"❌ torch: {e}")

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"❌ transformers: {e}")

try:
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    print(f"✓ diffusers imported successfully")
    print(f"  StableDiffusionInpaintPipeline: {StableDiffusionInpaintPipeline}")
    print(f"  DDIMScheduler: {DDIMScheduler}")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Other error: {type(e).__name__}: {e}")

print("\nDone!")