from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from app.config import AppConfig
from app.utils import safe_json_loads, sanitize_k8s_name


def load_cluster_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def build_training_job_object(run: dict[str, Any], app_config: AppConfig) -> client.V1Job:
    run_id = str(run["run_id"])
    hp = safe_json_loads(str(run["hp_json"]), default={})
    if not isinstance(hp, dict):
        raise ValueError("hp_json must decode to an object")

    desired_name = str(run.get("job_name") or f"{app_config.job_name_prefix}-{run_id}")
    k8s_job_name = sanitize_k8s_name(desired_name)
    callback_url = app_config.callback_url_template.format(run_id=run_id)

    env = [
        client.V1EnvVar(name="RUN_ID", value=run_id),
        client.V1EnvVar(name="MODEL", value=str(run["model"])),
        client.V1EnvVar(name="DATASET_REL_PATH", value=str(run["dataset_rel_path"])),
        client.V1EnvVar(name="HP_JSON", value=json.dumps(hp, ensure_ascii=True, sort_keys=True)),
        client.V1EnvVar(name="TOKEN_BUDGET", value=str(run["token_budget"])),
        client.V1EnvVar(name="SEQ_LEN", value=str(run["seq_len"])),
        client.V1EnvVar(name="SEED", value=str(run["seed"])),
        client.V1EnvVar(name="OUTPUT_ROOT", value=str(app_config.output_root)),
        client.V1EnvVar(name="DATASETS_ROOT", value=str(app_config.datasets_root)),
        client.V1EnvVar(name="SCHEDULER_CALLBACK_URL", value=callback_url),
    ]

    resources = client.V1ResourceRequirements(
        requests={app_config.gpu_resource_name: str(app_config.gpu_count)},
        limits={app_config.gpu_resource_name: str(app_config.gpu_count)},
    )

    container = client.V1Container(
        name="trainer",
        image=app_config.trainer_image,
        image_pull_policy=app_config.trainer_image_pull_policy,
        command=app_config.trainer_command,
        args=app_config.trainer_args,
        env=env,
        resources=resources,
        volume_mounts=[
            client.V1VolumeMount(
                name="shared-data",
                mount_path=str(app_config.data_root),
            )
        ],
    )

    labels = {
        "app": "metatune-trainer",
        "run_id": run_id,
    }

    pod_spec = client.V1PodSpec(
        restart_policy="Never",
        service_account_name=app_config.trainer_service_account_name,
        containers=[container],
        node_selector=app_config.node_selector,
        tolerations=[
            client.V1Toleration(**entry) for entry in (app_config.tolerations or [])
        ]
        or None,
        volumes=[
            client.V1Volume(
                name="shared-data",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=app_config.shared_pvc_name
                ),
            )
        ],
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels),
        spec=pod_spec,
    )
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=0,
    )
    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=k8s_job_name, labels=labels),
        spec=spec,
    )


def submit_training_job(*, run: dict[str, Any], app_config: AppConfig) -> str:
    load_cluster_config()
    batch_v1 = client.BatchV1Api()
    job = build_training_job_object(run, app_config)
    batch_v1.create_namespaced_job(body=job, namespace=app_config.namespace)
    return str(job.metadata.name)


def get_training_job_status(*, namespace: str, job_name: str) -> dict[str, Any]:
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
    for condition in conditions:
        if condition.status == "True":
            reason = condition.reason or condition.message

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
            namespace=namespace,
            label_selector=f"job-name={job_name}",
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

