# fraud-detection Helm Chart

Packages MLflow, BentoML, and the Flask app into a single deployable unit.
ArgoCD uses this chart as its source — the CD pipeline only updates `values-prod.yaml`
(image tag), commits back to Git, and ArgoCD syncs the cluster.

## Directory layout

```
helm/
├── fraud-detection/
│   ├── Chart.yaml
│   ├── values.yaml            ← shared defaults
│   ├── values-dev.yaml        ← dev overrides (lower resources, :latest tag)
│   ├── values-prod.yaml       ← prod overrides (pinned tag, higher resources)
│   └── templates/
│       ├── _helpers.tpl
│       ├── namespace.yaml
│       ├── mlflow.yaml        ← PVC + Deployment + Service
│       ├── bentoml.yaml       ← PVC + Deployment + Service
│       ├── flask.yaml         ← Deployment + Service
│       └── httproute.yaml     ← Gateway API HTTPRoute
└── argocd-application.yaml    ← replaces k8s/argoCD/automatic_insurance_claim_application.yaml
```

## Local dev (Minikube)

```bash
# Dry-run to verify rendering
helm template fraud-detection ./helm/fraud-detection -f helm/fraud-detection/values.yaml -f helm/fraud-detection/values-dev.yaml

# Install / upgrade
helm upgrade --install fraud-detection ./helm/fraud-detection -f helm/fraud-detection/values.yaml -f helm/fraud-detection/values-dev.yaml --namespace fraud-detection --create-namespace
```

## Production (via ArgoCD GitOps)

The CD pipeline patches `global.imageTag` in `values-prod.yaml` and commits:

```bash
# In your CD job (after building & pushing the image):
sed -i "s/^  imageTag:.*/  imageTag: sha-${GITHUB_SHA}/" \
  helm/fraud-detection/values-prod.yaml

git add helm/fraud-detection/values-prod.yaml
git commit -m "chore: bump image tag to sha-${GITHUB_SHA}"
git push
# ArgoCD detects the commit and syncs automatically
```

Or use the ArgoCD CLI for an immediate override without a Git commit:

```bash
argocd app set fraud-detection \
  --helm-set global.imageTag=sha-<commit>
```

## Toggling components

Every service has an `enabled` flag in `values.yaml`:

```yaml
mlflow:
  enabled: false   # exclude MLflow if you use an external tracking server
```

## Adding a dev ArgoCD Application

To run a separate ArgoCD Application pointed at dev, copy `argocd-application.yaml`,
rename it `argocd-application-dev.yaml`, and change:

```yaml
metadata:
  name: fraud-detection-dev
spec:
  source:
    helm:
      valueFiles:
        - values.yaml
        - values-dev.yaml        # ← dev overlay instead of prod
```
