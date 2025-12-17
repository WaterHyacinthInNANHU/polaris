# Creating Custom Environments

The environments we provide were scanned using ZED cameras, but the reconstruction pipeline is camera agnostic.

Capture a dense view video of a scene without motion blur, and run it through [COLMAP](https://colmap.github.io/install.html)

```bash
sudo apt install colmap ffmpeg

# Split video into frames at desired FPS
ffmpeg -i dense_view.mp4 -vf "fps=10" frames/dense_view_%04d.png
```

## Uploading Environments to HuggingFace

Share your custom environments with the community by uploading them to the [owhan/PolaRiS-environments](https://huggingface.co/datasets/owhan/PolaRiS-environments) dataset. **All uploads are automatically submitted as pull requests** (not direct commits) for review and quality control.

### Environment Structure

Your environment folder must contain:

```text
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
| `--repo-id` | Target HF dataset (default: `owhan/PolaRiS-environments`) |
| `--branch` | Target branch (default: `main`) |
| `--token` | HF token (or use `HF_TOKEN` env var) |
| `--strict` | Treat validation warnings as errors |
| `--require-pxr` | Fail if USD files can't be opened (requires pxr) |
| `--skip-validation` | Skip validation (not recommended) |

### How PRs Work for HuggingFace Datasets

When you run `polaris upload`, the tool automatically:

1. Validates your environment structure locally
2. Creates a pull request (not a direct commit) to the target dataset
3. Returns the PR URL or instructions to view it

**Viewing Your PR:**

- After upload, the CLI will print the PR URL (e.g., `https://huggingface.co/datasets/owhan/PolaRiS-environments/discussions/<PR_NUMBER>`)
- You can also view all PRs at: `https://huggingface.co/datasets/<repo-id>/discussions`
- PRs must be reviewed and merged by dataset maintainers before your environment appears in the dataset

**Merging Your PR:**

- Navigate to the PR URL in your browser
- Review the changes in the "Files" tab
- Click "Publish" when ready to merge (requires write access to the dataset)

### Managing Your PR Locally

After creating a PR, you can check it out locally to make changes:

```bash
# Clone the dataset repo
git clone https://huggingface.co/datasets/owhan/PolaRiS-environments
cd PolaRiS-environments

# Fetch and checkout PR (replace <PR_NUMBER> with your PR number from the upload output)
git fetch origin refs/pr/<PR_NUMBER>:pr/<PR_NUMBER>
git checkout pr/<PR_NUMBER>

# Make edits, then push back
git add .
git commit -m "Update environment"
git push origin pr/<PR_NUMBER>:refs/pr/<PR_NUMBER>
```
