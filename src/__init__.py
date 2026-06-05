"""TraceFlow — Traceable Rectified Flow Transformers Against Generative Model Inversion.

Active modules
--------------
- src.models       : AutoencoderBackend, FlowTransformer
- src.generation   : rectified_flow (sample_euler, sample_heun, flow_loss)
- src.security     : IdentityLatentTransform, KeyedLatentBottleneck, build_latent_transform
- src.data         : build_dataset, build_dataloader
- src.utils        : checkpoint, seed, image, metrics
"""
