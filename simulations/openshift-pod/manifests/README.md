# OpenShift Manifests

These manifests deploy:
1. A scheduler API `Deployment` (always-on pod).
2. RBAC for creating and monitoring `Job` pods.
3. One shared PVC mounted at `/data` by scheduler and trainer jobs.
4. A `Service` + external `Route`.

## Required variables

Set image references before applying:

```bash
export SCHEDULER_IMAGE="image-registry.openshift-image-registry.svc:5000/hack-europe-team-i/metatune-scheduler:latest"
export TRAINER_IMAGE="image-registry.openshift-image-registry.svc:5000/hack-europe-team-i/metatune-trainer:latest"
```

## Apply order

```bash
oc apply -f simulations/openshift-pod/manifests/scheduler-serviceaccount-role-rolebinding.yaml
oc apply -f simulations/openshift-pod/manifests/shared-pvc.yaml
envsubst < simulations/openshift-pod/manifests/scheduler-configmap.yaml | oc apply -f -
envsubst < simulations/openshift-pod/manifests/scheduler-deployment.yaml | oc apply -f -
oc apply -f simulations/openshift-pod/manifests/scheduler-service.yaml
oc apply -f simulations/openshift-pod/manifests/scheduler-route.yaml
```

## Smoke checks

```bash
oc get pods -n hack-europe-team-i -l app=metatune-scheduler
oc get route -n hack-europe-team-i metatune-scheduler
oc logs -n hack-europe-team-i deploy/metatune-scheduler
```

