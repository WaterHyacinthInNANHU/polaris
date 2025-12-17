# Creating Custom Environments

The environments we provide were scanned using ZED cameras, but the reconstruction pipeline is camera agnostic. 

Capture a dense view video of a scene without motion blur, and run it through [COLMAP](https://colmap.github.io/install.html)

Once you have your COLMAP dataset, follow the instructions in [2DGS](https://github.com/hbb1/2d-gaussian-splatting) to obtain a splat and corresponding extracted mesh.

Turn the `fuse_post.ply` mesh into a USD, and create an asset directory that follows this structure.
```
new_asset/
├── mesh.usd
├── splat.ply
├── textures/ (optional, if USD requires textures)
└── config.yaml (optional, USD parameter configuratoin)
```

Using the [online scene composition GUI](https://polaris-evals.github.io/compose-environments/), create a USD stage that composes the objects in the scene. Export and unzip the USD with the command below.
```
unzip scene.zip -d PolaRiS-environments/new_env/
```

You should now have a directory that looks something like this:
```
PolaRiS-environments/
└── new_env/
    ├── assets/
    │   ├── object_1/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   ├── object_2/
    │   │   └── mesh.usd
    │   │   └── textures/
    │   └── scene_splat/
    │       ├── config.yaml
    │       └── splat.ply
    ├── scene.usda             # Main USD stage file
    └── initial_conditions.json  (defined via GUI)
```

Add the new environment to the [environments file](../src/polaris/environments/__init__.py), following the same pattern as the default 6 environments. You can also see how to define a rubric to score rollouts with just a few lines of code.

<!-- ```bash
sudo apt install colmap ffmpeg

# Split video into frames at desired FPS
ffmpeg -i dense_view.mp4 -vf "fps=10" frames/dense_view_%04d.png
``` -->


## Uploading Environments to HuggingFace

Share your custom environments with the community by uploading them to the [PolaRiS-Evals/PolaRiS-Hub](https://huggingface.co/datasets/PolaRiS-Evals/PolaRiS-Hub) dataset. Uploads are submitted as pull requests for review.

### Environment Structure

Your environment folder must contain:
```
my_environment/
├── assets/
│   ├── object_1/
│   │   └── mesh.usdz
│   ├── object_2/
│   │   └── mesh.usdz
│   └── scene_splat/
│       ├── config.yaml
│       └── splat.ply
├── scene.usd              # Main USD stage file
└── initial_conditions.json
```

### Upload Commands

```bash
# Install the package (adds polaris CLI to PATH)
pip install -e .

# Dry-run validation only (no upload)
polaris upload ./PolaRiS-environments/my_environment --dry-run

# Upload and create a PR (uses HF_TOKEN env var)
export HF_TOKEN=your_huggingface_write_token
polaris upload ./PolaRiS-environments/my_environment \
  --pr-title "Add my_environment" \
  --pr-description "Description of the environment"

# Upload to a different repo
polaris upload ./PolaRiS-environments/my_environment \
  --repo-id your-org/your-dataset \
  --pr-title "Add my_environment"
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Validate only, don't upload |
| `--pr-title` | Title for the pull request |
| `--pr-description` | Description/body for the PR |
| `--repo-id` | Target HF dataset (default: `PolaRiS-Evals/PolaRiS-Hub`) |
| `--branch` | Target branch (default: `main`) |
| `--token` | HF token (or use `HF_TOKEN` env var) |
| `--strict` | Treat validation warnings as errors |
| `--require-pxr` | Fail if USD files can't be opened (requires pxr) |
| `--skip-validation` | Skip validation (not recommended) |

### Managing Your PR Locally

After creating a PR, you can check it out locally to make changes:

```bash
# Clone the dataset repo
git clone https://huggingface.co/datasets/PolaRiS-Evals/PolaRiS-Hub
cd PolaRiS-Hub

# Fetch and checkout PR (replace <PR_NUMBER> with your PR number)
git fetch origin refs/pr/<PR_NUMBER>:pr/<PR_NUMBER>
git checkout pr/<PR_NUMBER>

# Make edits, then push back
git add .
git commit -m "Update environment"
git push origin pr/<PR_NUMBER>:refs/pr/<PR_NUMBER>
```
