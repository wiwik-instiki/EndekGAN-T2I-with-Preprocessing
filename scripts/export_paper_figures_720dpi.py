"""Export 13 source-traceable manuscript figures at 720 dpi; avoid raw private data."""
from pathlib import Path
from PIL import Image, PngImagePlugin
import numpy as np, cv2, csv, hashlib, json, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=R/"figures"/"paper_IEEE_Access_720dpi_20261120"
O.mkdir(parents=True,exist_ok=True)
paths=["figures/manuscript_methods_2026/figure_01_arm_b_actual_motif_examples.png",
"figures/manuscript_methods_2026/figure_02_source_aware_preprocessing.png",
"figures/manuscript_methods_2026/figure_03_experimental_framework.png",
"figures/manuscript_methods_2026/figure_04_conditional_gan_architecture.png",
"figures/manuscript_methods_2026/figure_05_seam_penalty_edge_gap.png",
"figures/fig21_preprocessing_A_B_validation_kid.png",
"figures/fig23_ieee_results_four_metrics_3seeds.png",
"figures/fig24_ieee_results_seedwise_comparison.png",
"figures/fig25_ieee_actual_prompt_matched_three_models.png",
"figures/fig26_ieee_actual_2x2_tiling_three_models.png",
"figures/fig27_ieee_tipgan_training_examples_only.png"]
width=[3.36111,4.49359,5.96795,6.97222,5.92453,3.34210,4.74123,4.38410,4.75641,4.74359,6.48611,3.32,3.32]
captions=["Representative original-source-derived motifs","Source-aware preprocessing arms A/B",
"Research framework for RQ1 and RQ2","Conditional GAN architecture",
"Training seam penalty and held-out edge gap","Three-seed checkpoint-validation KID",
"Three-seed held-out FID, KID, edge gap and LPIPS","Per-seed held-out FID, KID, edge gap and LPIPS",
"Prompt-matched generated samples (seed 42)","Two-by-two tiling of Figure 9",
"TiPGAN training-only exemplars","Class-wise fidelity/continuity trade-off","Per-class LPIPS diversity diagnostics"]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
meta=[]
for n,rel in enumerate(paths,1):
 src=R/rel; target=O/f"Fig_{n:02d}_Endek_T2I_720dpi.png"
 with Image.open(src) as im:
  image=im.convert('RGB'); iw,ih=image.size
  ow=max(iw,int(np.ceil(width[n-1]*720)))
  oh=int(round(ih*ow/iw))
  if ow>iw:image=Image.fromarray(cv2.resize(np.asarray(image),(ow,oh),interpolation=cv2.INTER_LANCZOS4))
  info=PngImagePlugin.PngInfo();info.add_text("Original public figure",rel)
  info.add_text("Resolution caution","Resampling does not add native source-image detail")
  image.save(target,dpi=(720,720),compress_level=1,pnginfo=info)
 meta.append({"figure":n,"file":target.name,"caption":captions[n-1],"source":rel,
  "source_pixels":f"{iw}x{ih}","source_px_per_paper_in":round(iw/width[n-1],1),
  "output_pixels":f"{image.width}x{image.height}","dpi":720,
  "method":"Lanczos resampling" if ow>iw else "Retag original","sha256":digest(target)})
with (R/"results"/"heldout_B_transformer_vs_seam_all3seeds_allclasses.csv").open(encoding="utf-8-sig",newline="") as f:
 data=[x for x in csv.DictReader(f) if x['class_label'] in ('flora','fauna','dekoratif','geometris')]
cats=['flora','fauna','dekoratif','geometris']
def m(model,cat,field):
 vals=[float(x[field]) for x in data if x['model']==model and x['class_label']==cat]
 assert len(vals)==3;return sum(vals)/3
colors=['#24796C','#B56631','#7855A6','#376DA9']
plt.rcParams.update({'font.size':8,'axes.titlesize':9})
fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
for ax,col in zip(axes,['FID','KID']):
 for cat,color in zip(cats,colors):
  a=(m('transformer',cat,'seam_gap'),m('transformer',cat,col))
  b=(m('seam_proposed',cat,'seam_gap'),m('seam_proposed',cat,col))
  ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.4})
  ax.scatter(*a,color=color,marker='o',s=20,label=cat.title())
  ax.scatter(*b,facecolors='none',edgecolors=color,marker='s',s=25)
 ax.set_xlabel('Opposing-edge gap (lower better)')
 ax.set_ylabel(col+' (lower better)');ax.set_title(col+' / border trade-off');ax.grid(alpha=.25)
axes[0].legend(fontsize=6,loc='best')
fig.suptitle('Class-wise fidelity and border continuity (three-seed means)',fontsize=10)
dest=O/'Fig_12_Endek_T2I_720dpi.png';fig.savefig(dest,dpi=720);plt.close(fig)
meta.append({'figure':12,'file':dest.name,'caption':captions[11],'source':'results/heldout_B_transformer_vs_seam_all3seeds_allclasses.csv','source_pixels':'vector-generated','source_px_per_paper_in':'n/a','output_pixels':str(Image.open(dest).size),'dpi':720,'method':'Native 720dpi metric plot','sha256':digest(dest)})
fig,axes=plt.subplots(2,2,figsize=(7,4.45),layout='constrained')
for ax,cat in zip(axes.flat,cats):
 for model,label,color,mark in [('transformer','No-seam','#376DA9','o'),('seam_proposed','Seam proposed','#B56631','s')]:
  y=[float(next(x['LPIPS_diversity'] for x in data if x['model']==model and x['class_label']==cat and int(x['seed'])==seed)) for seed in (42,123,2025)]
  ax.plot(range(3),y,color=color,marker=mark,label=label,lw=1.5)
 ax.set_title(cat.title());ax.set_xticks(range(3),['42','123','2025']);ax.set_xlabel('Training seed');ax.set_ylabel('LPIPS diversity ↑');ax.grid(alpha=.25)
axes[0,0].legend(fontsize=7)
fig.suptitle('Within-prompt perceptual diversity across classes and seeds',fontsize=10)
dest=O/'Fig_13_Endek_T2I_720dpi.png';fig.savefig(dest,dpi=720);plt.close(fig)
meta.append({'figure':13,'file':dest.name,'caption':captions[12],'source':'results/heldout_B_transformer_vs_seam_all3seeds_allclasses.csv','source_pixels':'vector-generated','source_px_per_paper_in':'n/a','output_pixels':str(Image.open(dest).size),'dpi':720,'method':'Native 720dpi metric plot','sha256':digest(dest)})
assert len(meta)==13
for it in meta:
 with Image.open(O/it['file']) as im:
  assert min(im.info.get('dpi',(0,0)))>600
  im.verify()
with (O/'FIGURE_MANIFEST.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(meta[0]));w.writeheader();w.writerows(meta)
(O/'FIGURE_MANIFEST.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
lines=['# IEEE Access Endek T2I: Figure package (720 dpi)','',
'All 13 figures correspond to the November 20, 2026 manuscript and contain no private model checkpoints, raw datasets or source-linked prompt metadata.','',
'**Image quality disclosure.** Figures 1–11 reuse existing published source image panels; most have been Lanczos-upscaled to 720 pixels per paper inch. This does not create additional optical detail. Figures 12–13 are native 720dpi matplotlib re-renders of the paper metric plots. They reproduce numeric evidence but not necessarily byte-identical styling of the manuscript embedded raster figures.','',
'**Rights and scope.** Figure 1 shows already-public original-source textile samples; publication rights still need author confirmation. Figure 11 contains image-conditioned training-only results, not held-out text-to-image tests. Neither figure set is artisan-certified.','',
'## Figure index','']
for x in meta:lines.append(f"- [Fig. {x['figure']}: {x['caption']}]({x['file']})")
lines+=['','See [FIGURE_MANIFEST.csv](FIGURE_MANIFEST.csv) for dimensions, sources and SHA-256 checksums.']
(O/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'figure_count':len(meta),'folder':str(O),'max_bytes':max((O/x['file']).stat().st_size for x in meta),'source_dpi_below_600':sum(isinstance(x['source_px_per_paper_in'],float) and x['source_px_per_paper_in']<600 for x in meta)},indent=2))
