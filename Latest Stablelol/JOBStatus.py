from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize with your IBM Quantum token (save it once with service.save_account() if not already done)
service = QiskitRuntimeService(channel="ibm_quantum", token="your_token_here")

# Retrieve a specific job by ID (replace "job_id_here" with your actual job ID)
job = service.job("job_id_here")
print(job.status())  # e.g., 'COMPLETED', 'RUNNING', 'QUEUED'
print(job.result())  # Get results if completed

# Or list your recent jobs (e.g., last 10)
jobs = service.jobs(limit=10)
for job in jobs:
    print(f"Job ID: {job.job_id()}, Status: {job.status()}, Backend: {job.backend().name}")