from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

DEFAULT_TRAIN_IMAGE = (
    "image-registry.openshift-image-registry.svc:5000/"
    "hack-europe-team-i/train@sha256:02ffe0bfec264b4c434b27dda3ba36ca691a7bbe7443c22a66a8605c9191561c"
)
DEFAULT_SERVICE_ACCOUNT = "green-scheduler-sa"
DEFAULT_ROLE = "green-scheduler-job-runner"
DEFAULT_ROLE_BINDING = "green-scheduler-job-runner-binding"
DEFAULT_GPU_RESOURCE = "nvidia.com/gpu"


def load_cluster_config() -> None:
    """Load Kubernetes auth config for in-cluster or local execution."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def ensure_scheduler_permissions(
    namespace: str,
    service_account_name: str = DEFAULT_SERVICE_ACCOUNT,
    role_name: str = DEFAULT_ROLE,
    role_binding_name: str = DEFAULT_ROLE_BINDING,
) -> None:
    """Create idempotent RBAC objects so scheduler can create/watch Jobs."""
    load_cluster_config()
    core_v1 = client.CoreV1Api()
    rbac_v1 = client.RbacAuthorizationV1Api()

    service_account = client.V1ServiceAccount(
        metadata=client.V1ObjectMeta(name=service_account_name)
    )
    role = client.V1Role(
        metadata=client.V1ObjectMeta(name=role_name),
        rules=[
            client.V1PolicyRule(
                api_groups=["batch"],
                resources=["jobs"],
                verbs=["create", "get", "list", "watch", "delete", "patch"],
            ),
            client.V1PolicyRule(
                api_groups=[""],
                resources=["pods", "pods/log"],
                verbs=["get", "list", "watch"],
            ),
        ],
    )
    role_binding = client.V1RoleBinding(
        metadata=client.V1ObjectMeta(name=role_binding_name),
        role_ref=client.V1RoleRef(
            api_group="rbac.authorization.k8s.io",
            kind="Role",
            name=role_name,
        ),
        subjects=[
            client.V1Subject(
                kind="ServiceAccount",
                name=service_account_name,
                namespace=namespace,
            )
        ],
    )

    for kind, create_fn, body in [
        ("ServiceAccount", core_v1.create_namespaced_service_account, service_account),
        ("Role", rbac_v1.create_namespaced_role, role),
        ("RoleBinding", rbac_v1.create_namespaced_role_binding, role_binding),
    ]:
        try:
            create_fn(namespace=namespace, body=body)
            print(f"   Created {kind} '{body.metadata.name}'")
        except ApiException as e:
            if e.status != 409:
                raise


def _build_tolerations(
    tolerations: list[dict[str, Any]] | None,
) -> list[client.V1Toleration] | None:
    if not tolerations:
        return None
    return [client.V1Toleration(**tol) for tol in tolerations]


def create_training_job(
    *,
    run_id: str,
    dc_id: str,
    namespace: str,
    hp_values: dict[str, float],
    dataset_abs_path: str,
    pvc_name: str,
    cache_mount_path: str = "/mnt/cache",
    service_account_name: str = DEFAULT_SERVICE_ACCOUNT,
    gpu_count: int = 1,
    gpu_resource_name: str = DEFAULT_GPU_RESOURCE,
    job_name: str | None = None,
    image: str = DEFAULT_TRAIN_IMAGE,
    command: list[str] | None = None,
    args: list[str] | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    auto_ensure_permissions: bool = True,
) -> str:
    """Create a single Kubernetes training Job and return the Kubernetes Job name."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    if not pvc_name:
        raise ValueError("pvc_name is required")
    if not namespace:
        raise ValueError("namespace is required")

    if auto_ensure_permissions:
        ensure_scheduler_permissions(
            namespace=namespace,
            service_account_name=service_account_name,
        )

    load_cluster_config()
    batch_v1 = client.BatchV1Api()
    k8s_job_name = (
        job_name
        if job_name
        else f"metatune-train-{run_id.lower().replace('_', '-')[:40]}"
    )

    gpu_resources = client.V1ResourceRequirements(
        requests={gpu_resource_name: str(gpu_count)},
        limits={gpu_resource_name: str(gpu_count)},
    )

    env = [
        client.V1EnvVar(name="LORA_R", value=str(hp_values["lora_r"])),
        client.V1EnvVar(name="LEARNING_RATE", value=str(hp_values["learning_rate"])),
        client.V1EnvVar(name="LORA_DROPOUT", value=str(hp_values["lora_dropout"])),
        client.V1EnvVar(name="DATASET_PATH", value=dataset_abs_path),
        client.V1EnvVar(name="RUN_ID", value=run_id),
        client.V1EnvVar(name="DC_ID", value=dc_id),
    ]

    container = client.V1Container(
        name="trainer",
        image=image,
        command=command,
        args=args,
        env=env,
        resources=gpu_resources,
        volume_mounts=[
            client.V1VolumeMount(name="cache-volume", mount_path=cache_mount_path)
        ],
    )

    labels = {
        "app": "metatune-trainer",
        "run_id": run_id,
        "dc_id": dc_id,
    }

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels),
        spec=client.V1PodSpec(
            restart_policy="Never",
            service_account_name=service_account_name,
            node_selector=node_selector,
            tolerations=_build_tolerations(tolerations),
            containers=[container],
            volumes=[
                client.V1Volume(
                    name="cache-volume",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_name
                    ),
                )
            ],
        ),
    )

    spec = client.V1JobSpec(template=template, backoff_limit=0)
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=k8s_job_name, labels=labels),
        spec=spec,
    )

    batch_v1.create_namespaced_job(body=job, namespace=namespace)
    return k8s_job_name


def get_training_job_status(namespace: str, job_name: str) -> dict[str, Any]:
    """Return a lightweight status view for a Kubernetes Job."""
    load_cluster_config()
    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()

    try:
        job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)
    except ApiException as e:
        if e.status == 404:
            return {"state": "NOT_FOUND", "reason": "Job not found", "pod_name": None}
        raise

    status = job.status or client.V1JobStatus()
    conditions = status.conditions or []
    reason = None
    for cond in conditions:
        if cond.status == "True":
            reason = cond.reason or cond.message

    state = "RUNNING"
    if (status.succeeded or 0) > 0:
        state = "SUCCEEDED"
    elif (status.failed or 0) > 0:
        state = "FAILED"
        if not reason:
            reason = "Job reported failed pods"
    elif (status.active or 0) == 0:
        state = "SUBMITTING"

    pod_name = None
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace, label_selector=f"job-name={job_name}"
        )
        if pods.items:
            pod_name = pods.items[0].metadata.name
    except ApiException:
        pod_name = None

    completion_time = None
    if status.completion_time:
        completion_time = status.completion_time.astimezone(timezone.utc).isoformat()
    elif state in {"SUCCEEDED", "FAILED"}:
        completion_time = datetime.now(timezone.utc).isoformat()

    return {
        "state": state,
        "reason": reason,
        "pod_name": pod_name,
        "completion_time": completion_time,
    }


if __name__ == "__main__":
    ensure_scheduler_permissions(namespace="hack-europe-team-i")
