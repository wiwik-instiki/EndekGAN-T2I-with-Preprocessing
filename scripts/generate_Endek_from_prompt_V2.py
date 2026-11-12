# -*- coding: utf-8 -*-
"""Generate 256px Endek motif from an arbitrary user prompt using a NEW V2 checkpoint."""
from pathlib import Path
import argparse,json,sys,importlib.util
ROOT=Path(__file__).resolve().parent
CLASSES={"flora":0,"fauna":1,"dekoratif":2,"geometris":3}
ARMS={"A":"full_image_augmented_A","B":"full_plus_crop_ablation_B"}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prompt",type=str,required=True)
    p.add_argument("--class",dest="motif_class",choices=list(CLASSES),required=True)
    p.add_argument("--arm",choices=list(ARMS),default="B")
    p.add_argument("--model",default="seam_proposed")
    p.add_argument("--seed",type=int,default=42,help="Model training seed")
    p.add_argument("--latent-seed",type=int,default=12345)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    output=ROOT/"runs_preprocessed"/ARMS[args.arm]/f"{args.model}_seed{args.seed}"
    state=output/"run_state.json"
    if not state.is_file():
        raise FileNotFoundError("No completed V2 checkpoint; training must finish first.")
    s=json.loads(state.read_text(encoding="utf-8"))
    if not s.get("completed"):
        raise RuntimeError("This V2 run did not finish.")
    checkpoint=Path(s.get("best_checkpoint") or "")
    if not checkpoint.is_file():
        raise FileNotFoundError("New V2 best checkpoint missing")
    if ROOT.resolve() not in checkpoint.resolve().parents:
        raise RuntimeError("Refusing to use checkpoint outside V2 project")
    print("MODEL:",checkpoint)
    print("PROMPT:",args.prompt)
    print("MOTIF CLASS:",args.motif_class)
    if not args.execute:
        print("DRY RUN: pass --execute for image generation.")
        return
    if not args.prompt.strip():raise ValueError("Empty text prompt")
    import torch
    from torchvision.utils import save_image
    spec=importlib.util.spec_from_file_location("engine",ROOT/"endek_v2_preprocessing_training_engine.py")
    engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
    G,encoder,cfg,dev=engine.generator_from_checkpoint(checkpoint,device=args.device,trusted=True)
    text=encoder.encode([args.prompt],[args.motif_class]).to(dev)
    gen=torch.Generator(device=dev);gen.manual_seed(args.latent_seed)
    noise=torch.randn((1,cfg.latent_dim),generator=gen,device=dev)
    category=torch.tensor([CLASSES[args.motif_class]],device=dev)
    with torch.inference_mode():
        image=(G(noise,text,category).cpu()+1).mul(.5).clamp(0,1)
    out=output/"prompt_generation";out.mkdir(parents=True,exist_ok=True)
    dest=out/(f"{args.motif_class}_z{args.latent_seed}_modelseed{args.seed}.png")
    if dest.exists():raise FileExistsError("Change latent seed; refusing to overwrite existing generated image")
    save_image(image,dest)
    meta={"prompt":args.prompt,"class":args.motif_class,"latent_seed":args.latent_seed,
          "checkpoint":str(checkpoint),"output":str(dest),"generated_by":"NEW EndekGAN-T2I V2"}
    dest.with_suffix(".json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print("GENERATED:",dest)
if __name__=="__main__":
    main()
