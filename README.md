# RNACG
<!-- RNACG (RNA Conditional  Generator), which is based on Flow Matching -->
This is the official implementation of RNACG, we would like to release the paramters of the model and all the output file mentioned in the paper.

<!-- We are still working on the code cleaning and will update this repository frequently. Please stay tuned. -->

# Easy Start

We provide rfamflow model on RF00001 as an example. You can find the model in `ckpts/bestmodel_RF00001.pth` and the output file in `data/examples/RF00001test.out`.

```bash
python cli.py --input RF00001 --output data/examples/RF00001test.out --model ./ckpts/bestmodel_RF00001.pth
```

Similarly, you can run the following command to generate the output file for the inverse folding task. We also provide the model in `ckpts/best_inv3dflow_ribodiffusion_0.pth` and the input file in `data/examples/inv3d_example.pdb`.


```bash
python cli.py --task inversefold --input .\data\examples\inv3d_example.pdb --output .\data\examples\inv3d.out --model .\ckpts\best_inv3dflow_ribodiffusion_0.pth --device cuda:0
```
