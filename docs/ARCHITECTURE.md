# Architecture from my MTP

![Redrawn architecture](../assets/architecture.svg)

I based this vector figure on the model diagram included in my LaTeX source: `source/Chapters/method.tex`, which references `Images/Complete Architercture.png`. I kept its Stage A/B/C organization, shared CLIP conditioning, C-to-B feature cascade and iterative masked-token sampling loop.

I redrew the figure with consistent typography, aligned components, a restrained color palette and separate routes for text, image features and token flow. I distinguish the encoded training targets with dashed arrows and label frozen components explicitly. The SVG stays sharp when enlarged.

For this repository, I label the reconstructed implementation: RGB256 images, a 16×16 VQ token map, 192×7×7 image features, 12 masked refinement steps and 50 DDIM steps. I also show the selected grown ranks, B=8 and C=10. These ranks describe the reported experiment rather than a requirement of the architecture. I leave the original diagram's historical parameter counts and latent-shape annotations in the preserved original; the measured parameter accounting is documented in [evaluation details](EVALUATION.md).

The encoder branch creates training tokens. At text-only inference, I generate features with C, refine tokens with B and decode them with the frozen VQ decoder. I cache the frozen text and VQ outputs during training. The complete objectives and stage dependencies are in my [training protocol](TRAINING.md).

<details>
<summary>Original diagram from my LaTeX source</summary>

![Original MTP architecture](../assets/thesis/architecture-original.png)

I preserve this PNG byte for byte. The [original LaTeX figure fragment](../assets/thesis/architecture-source.tex) and [source hashes](../assets/thesis/provenance.json) record where it came from. The new figure is a presentation redraw with reconstruction-specific labels; it does not alter my thesis source.

</details>
