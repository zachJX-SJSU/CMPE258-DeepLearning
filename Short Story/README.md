# CMPE258-DeepLearning
# Short stories
Medium Article:[Article](https://medium.com/@zach.xie/from-2d-images-to-3d-world-models-a-survey-of-3d-scenario-generation-with-ai-ae8601f49965)   

Slides: [Google Slides](https://docs.google.com/presentation/d/17CbMD9PSL52nJ9qGUWyaueNgSIKAEnu51MQ9pAQh2I0/edit?usp=sharing) (also in git repo)  

Video Walkthru:[Video](https://youtu.be/ICvCp2AAzqA?si=zP8QWitW29PUUTHs)    


# Hybrid Mesh/Depth/Point-Cloud + 3D Gaussian Splatting for Single-Image 3D World Generation

## 1. Introduction: from view synthesis to world generation

The goal of **3D world generation from a 2D image** is deceptively simple: given one image, generate a navigable three-dimensional environment that plausibly extends beyond the camera's original field of view. This is harder than ordinary novel-view synthesis. Novel-view synthesis usually assumes that the target camera remains near observed content; world generation must invent unseen regions, preserve spatial coherence, and create a representation that can be rendered from many new viewpoints.

The field has evolved through several stages. Early view-synthesis methods focused on reconstructing or interpolating views within a bounded camera range. Later generative approaches began asking a more ambitious question: can a system extrapolate a scene indefinitely from one image? **Infinite Nature** framed this as *perpetual view generation*: long-range synthesis of novel views along an arbitrary camera trajectory from a single input image. The authors argued that standard view synthesis fails under large camera motion, while generic video generation often lacks explicit scene geometry; their answer was a hybrid "render, refine, repeat" strategy combining geometry and image synthesis.  
Source: Liu et al., *Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image*, ICCV 2021.  
https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Infinite_Nature_Perpetual_View_Generation_of_Natural_Scenes_From_a_ICCV_2021_paper.pdf

Recent methods have shifted from producing only a video or view sequence toward producing a **persistent 3D scene representation**. This is where **3D Gaussian Splatting**, or **3DGS**, becomes important. 3DGS represents a scene using many explicit radiance primitives--Gaussian "splats"--that can be rendered efficiently and interactively. For 3D worlds, this is attractive because the final output is not merely a generated video; it is an explorable asset.

The technical route examined here is:

> **Hybrid mesh/depth/point-cloud + 3DGS**: use depth, point clouds, or mesh-like geometry to ground the world spatially, use diffusion/inpainting models to hallucinate missing content, then use 3D Gaussian Splatting as the final real-time rendering representation.

The main paper for this route is:

> **Schwarz et al., "A Recipe for Generating 3D Worlds From a Single Image," ICCV 2025.**

This paper is a strong representative because it directly targets **single-image-to-3D-world generation**, rather than only object generation, local reconstruction, or video synthesis. The ICCV record summarizes the method as generating coherent panoramas with a pretrained diffusion model, lifting them into 3D with a metric depth estimator, filling unobserved regions with point-cloud-conditioned inpainting, and producing high-quality 3D environments suitable for VR display.  
Source: Schwarz et al., *A Recipe for Generating 3D Worlds from a Single Image*, ICCV 2025.  
https://openaccess.thecvf.com/content/ICCV2025/html/Schwarz_A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025_paper.html

---

## 2. Foundation paper: *Infinite Nature* and the render-refine-repeat idea

A useful foundation paper for this topic is **Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image** by Liu et al., presented at ICCV 2021. It does not use 3D Gaussian Splatting, because 3DGS had not yet become the dominant representation. But it established a conceptual pattern that later world-generation pipelines still follow: use geometry to render a new view, use a generative model to repair or extend the image, then repeat.

**Suggested illustration:** Use **Figure 1** from *Infinite Nature*, which shows training videos, a single test input image, and generated output frames along a long camera trajectory. The figure visually captures the central problem: starting from one frame and producing a long fly-through sequence.

The key insight of *Infinite Nature* is that world generation should not be treated as pure video synthesis. A video model may generate plausible frames, but without geometry it can ignore the actual camera motion and 3D structure. Conversely, classical view synthesis is geometrically grounded but fails when the camera moves beyond observed regions. *Infinite Nature* therefore uses disparity maps as a geometric proxy and repeatedly renders, refines, and updates both image and geometry. The paper reports that this recursive framework can generate hundreds of frames from a single input image, especially for aerial coastal scenes.

This paper laid the foundation in three ways.

First, it defined a more ambitious task than local view interpolation. The goal was not just to synthesize a nearby unseen view, but to support long-range camera motion through an imagined scene. Second, it showed the value of a **hybrid pipeline**: geometry alone is too sparse, while generation alone is too unconstrained. Third, it anticipated the modern "world model" framing: a generated environment should maintain enough spatial structure that an observer, camera, or agent can move through it.

Its main limitation is global consistency. The generated world is plausible along a trajectory, but it is not necessarily a persistent, globally coherent 3D asset. That limitation motivates the move toward explicit 3D representations such as 3DGS.

---

## 3. The main paper: *A Recipe for Generating 3D Worlds From a Single Image*

Schwarz et al.'s **A Recipe for Generating 3D Worlds From a Single Image** is a natural successor to the *Infinite Nature* line of thinking. It keeps the hybrid philosophy--combine 2D generative priors with explicit geometry--but changes the output target. Instead of producing only a long video, it constructs a **360-degree 3D world** represented by **Gaussian Splats** and designed for VR exploration. The paper's first figure states that the generated scene can be explored in a VR headset within a cube of 2 meters per side.

**Suggested illustration:** Use **Figure 1** from the ICCV 2025 paper as the main pipeline illustration. It shows the single input image on the left and multiple rendered views from the resulting 3DGS world, including VR headset orientation indicators.

The paper's core claim is that the extremely hard problem of single-image 3D world generation can be decomposed into easier subproblems. Instead of training a monolithic image-to-world model, the authors use existing models strategically. Their method has two major stages: **panorama synthesis** and **3D lifting/completion**. The paper explicitly lists its contributions as decomposing 3D scene synthesis into panorama synthesis and point-cloud-conditional inpainting, proposing a zero-shot in-context approach to panorama generation, fine-tuning a ControlNet for point-cloud-conditioned inpainting, and augmenting 3DGS with a distortion correction mechanism.

Source: Schwarz et al., *A Recipe for Generating 3D Worlds from a Single Image*, ICCV 2025.  
https://openaccess.thecvf.com/content/ICCV2025/papers/Schwarz_A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025_paper.pdf

---

## 4. Technical pipeline

### 4.1 Panorama generation

The first problem is field-of-view expansion. A normal image sees only a narrow slice of the world. To create an explorable environment, the method first expands the input into a coherent **360-degree panorama**.

This is an important design choice. If the system immediately lifts the narrow image into 3D, it has almost no information about what surrounds the viewer. The generated world would be incomplete and would quickly expose missing regions under camera rotation. By contrast, a panorama provides a full surrounding visual shell. It does not yet solve 3D geometry, but it gives the later stages a much wider visual context.

The paper frames panorama generation as an **in-context learning problem for a pretrained inpainting model**. It compares different panorama-generation strategies, including direct ad-hoc outpainting, sequential camera rotation, and an anchored strategy that duplicates the input to stabilize sky and ground synthesis. The method is training-free at this panorama stage and relies on the pretrained generative model's capacity to complete visual context.

**Suggested illustration:** Use the paper's panorama-synthesis figure on page 3, which compares ad-hoc, sequential, and anchored generation strategies. This is useful because it shows that the paper's novelty is not just "use diffusion," but how the diffusion model is prompted and spatially arranged for global consistency.

### 4.2 Metric depth and point-cloud lifting

Once the panorama exists, the system estimates **metric depth** and lifts the panoramic image into 3D. This produces a point-cloud-like geometric scaffold: pixels become spatial samples with color and position.

This stage is the bridge between 2D generation and 3D representation. The generated panorama is visually rich but physically shallow; depth estimation gives it spatial structure. The resulting point cloud allows the system to render the current partial world from new viewpoints and detect holes or disoccluded regions.

This is where the route differs from pure diffusion or pure video generation. The method is not merely asking a model to imagine the next frame. It explicitly renders from an evolving 3D structure, then asks the generative model to fill what is missing.

### 4.3 Point-cloud-conditioned inpainting

After lifting, the system still has missing geometry and unseen surfaces. When the viewer moves, some previously hidden regions become visible. The paper addresses this through **point-cloud-conditioned inpainting**.

The idea is straightforward but powerful. The partial 3D world is rendered from a new viewpoint. Some pixels are known because the point cloud can project them into the new camera; other pixels are unknown because they correspond to disoccluded regions. The inpainting model receives both visual context and geometric conditioning, so the generated content is more spatially anchored than ordinary image inpainting.

The paper states that it fine-tunes a ControlNet using a forward-backward warping strategy for point-cloud-conditioned inpainting, requiring minimal training effort. This is important for the survey's technical route: the system uses a point cloud not as the final representation, but as a **conditioning mechanism** for generating missing content.

### 4.4 3D Gaussian Splatting as the final representation

The final scene is represented using **3D Gaussian Splatting**. This is what makes the output an explorable 3D world rather than a static panorama or generated video.

3DGS gives the method a practical rendering substrate. Once the Gaussian scene is built, it can be rendered in real time on a VR device, although the paper notes that the full scene-synthesis process itself is not real time because diffusion inference remains computationally expensive.

The paper also introduces a distortion correction mechanism to handle residual inconsistencies between generated multiview images. This matters because generative inpainting is not perfectly 3D-consistent. Even if the point cloud provides geometry, hallucinated regions can disagree across views. A learned correction over the Gaussian representation improves sharpness and detail.

**Suggested illustration:** Use **Figure 2** from the paper, which shows multiple examples of single input images and rendered views from the generated 3DGS worlds. This figure is useful for demonstrating the method's output diversity: stylized landscapes, snowy scenes, fantasy forests, and indoor-like environments.

---

## 5. Why this route is promising

The hybrid mesh/depth/point-cloud + 3DGS route is promising because it combines the strengths of three otherwise incomplete paradigms.

First, **2D generative models** provide strong visual priors. They know how skies, forests, rooms, paths, buildings, and lighting should look. This is essential because the input image severely underdetermines the unseen world.

Second, **depth and point clouds** provide geometric grounding. They prevent the system from becoming a pure hallucination engine. Rendered point clouds preserve already generated content, expose missing regions, and give the inpainting model a structured signal.

Third, **3D Gaussian Splatting** provides a real-time, explicit rendering representation. This makes the output more usable than a generated video. The viewer can move inside the scene, and the world can be rendered from new camera poses.

This combination directly addresses the weaknesses of earlier approaches. Compared with *Infinite Nature*, the ICCV 2025 paper produces a persistent 3DGS environment rather than only a generated fly-through. Compared with pure video-diffusion approaches, it models 3D structure from the beginning. The authors report that this explicit 3D modeling allows their method to outperform state-of-the-art video-synthesis-based methods on several image-quality metrics.

---

## 6. Limitations

Despite its strength, this route is not yet a complete solution to physical, large-scale world generation.

The paper itself identifies several limitations. The navigable area is constrained: the generated world is designed around a 2-meter cube, and the point-cloud-conditioned inpainting problem becomes significantly harder beyond that range. The method also cannot reliably generate the backsides of fully occluded objects. Finally, synthesis is not real time because large diffusion models are computationally expensive, even though the final 3DGS scene can be displayed interactively.

There is also a deeper limitation: **visual 3D does not equal physical 3D**. A Gaussian-splat world can look convincing, but it does not automatically provide object identities, watertight meshes, collision proxies, material parameters, or dynamics. For VR viewing, this may be acceptable. For robotics, manipulation, or game-like physical interaction, additional structure is needed.

So the route is best understood as a strong path toward **immersive visual world generation**, not yet a full physical world model.

---

## 7. Relationship to the broader field

This paper sits at an important midpoint in the evolution of world generation.

Earlier work such as *Infinite Nature* showed that long-range generation needs both geometry and synthesis. NeRF-era methods later emphasized differentiable neural rendering, but often suffered from slow optimization and limited scalability. Video-diffusion methods improved generative capacity but remained geometrically fragile. The 3DGS era shifts attention toward explicit, fast-rendering scene representations.

The ICCV 2025 paper's contribution is not a single new model architecture. Its value is more architectural: it gives a practical recipe for connecting existing components into a world-generation pipeline. The core decomposition is:

> **single image -> 360 panorama -> metric depth -> point-cloud-conditioned completion -> 3D Gaussian Splat world**

That decomposition is likely to remain influential because it matches the structure of the problem. A single image must be expanded in angular coverage, lifted into geometry, completed under novel views, and rendered interactively.

---

## 8. Conclusion

The hybrid mesh/depth/point-cloud + 3DGS route is currently one of the most promising technical directions for 3D world generation from a single 2D image. It avoids the brittleness of pure geometry, the inconsistency of pure 2D generation, and the inefficiency of many implicit neural-rendering pipelines.

**Infinite Nature** laid an early foundation by framing single-image world extrapolation as a geometry-aware generative process: render, refine, and repeat. **A Recipe for Generating 3D Worlds From a Single Image** modernizes that idea with diffusion-based panorama synthesis, metric depth lifting, point-cloud-conditioned inpainting, and 3D Gaussian Splatting. The result is a practical pipeline for generating immersive, navigable 3D environments from one image.

For a survey, this paper is a strong main reference because it clearly represents the field's current direction: not a monolithic world model, but a **hybrid system that uses 2D generative priors for imagination, geometric representations for consistency, and 3DGS for interactive rendering**. Its limitations also make the next research frontier clear: expanding navigable scale, improving occluded geometry, supporting real-time generation, and moving from visually plausible worlds toward physically interactive worlds.

---

## References

1. Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, Angjoo Kanazawa. **Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image.** ICCV 2021.  
   https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Infinite_Nature_Perpetual_View_Generation_of_Natural_Scenes_From_a_ICCV_2021_paper.pdf

2. Katja Schwarz, Denys Rozumnyi, Samuel Rota Bulo, Lorenzo Porzi, Peter Kontschieder. **A Recipe for Generating 3D Worlds from a Single Image.** ICCV 2025.  
   https://openaccess.thecvf.com/content/ICCV2025/html/Schwarz_A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025_paper.html

3. Katja Schwarz, Denys Rozumnyi, Samuel Rota Bulo, Lorenzo Porzi, Peter Kontschieder. **A Recipe for Generating 3D Worlds from a Single Image.** ICCV 2025 PDF.  
   https://openaccess.thecvf.com/content/ICCV2025/papers/Schwarz_A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025_paper.pdf

