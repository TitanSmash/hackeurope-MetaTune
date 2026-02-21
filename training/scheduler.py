import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException

def create_training_job(region: str, namespace: str, job_id: int):
    """Creates a single, independent Kubernetes Job routed to a simulated region."""
    
    # 1. Auto-detect Authentication
    try:
        config.load_incluster_config() # If running inside OpenShift
    except config.ConfigException:
        config.load_kube_config()      # If running on your laptop
        
    batch_v1 = client.BatchV1Api()
    job_name = f"green-model-train-{job_id}"
    
    # 2. Define the container (Swap 'python:3.9-slim' with your OpenShift AI image)
    container = client.V1Container(
        name="trainer",
        image="image-registry.openshift-image-registry.svc:5000/hack-europe-team-i/train@sha256:02ffe0bfec264b4c434b27dda3ba36ca691a7bbe7443c22a66a8605c9191561c",
        command=["python", "-c", f"print('Training in {region}...'); import time; time.sleep(20); print('Done!')"]
    )

    # 3. Define the Job with our special region label
    labels = {
        "app": "green-scheduler", 
        "simulated-region": region
    }
    
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=labels),
        spec=client.V1PodSpec(restart_policy="Never", containers=[container])
    )

    spec = client.V1JobSpec(
        template=template, 
        backoff_limit=0 # Don't retry if the demo container crashes
    )

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name, labels=labels),
        spec=spec
    )

    # 4. Dispatch the Job to OpenShift
    try:
        batch_v1.create_namespaced_job(body=job, namespace=namespace)
        print(f"   ✅ Dispatched Job '{job_name}' to {region}!")
    except ApiException as e:
        print(f"   ❌ Failed to create Job: {e}")

if __name__ == '__main__':
    OPENSHIFT_NAMESPACE = "hack-europe-team-i" 
    
    print("🚀 Starting Global Green Energy Scheduler...\n")
